"""
Simple data loader for OpenHands Index leaderboard.
Loads JSONL files from local directory or GitHub repository.
Uses pydantic models from openhands-index-results for validation.
"""
import os
import sys
import logging
import pandas as pd
import json
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Pydantic models will be imported after setup_data adds them to path
_schema_models_loaded = False
Metadata = None
ScoreEntry = None


def _ensure_schema_models():
    """Lazily import pydantic schema models from openhands-index-results."""
    global _schema_models_loaded, Metadata, ScoreEntry
    
    if _schema_models_loaded:
        return _schema_models_loaded
    
    try:
        # Try importing from the cloned repo's scripts directory
        from validate_schema import Metadata as _Metadata, ScoreEntry as _ScoreEntry
        Metadata = _Metadata
        ScoreEntry = _ScoreEntry
        _schema_models_loaded = True
        logger.info("Successfully loaded pydantic schema models from openhands-index-results")
    except ImportError as e:
        logger.warning(f"Could not import pydantic schema models: {e}")
        logger.warning("Data will be loaded without schema validation")
        _schema_models_loaded = False
    
    return _schema_models_loaded


def load_and_validate_agent_data(agent_dir: Path) -> tuple[Optional[dict], Optional[list], list[str]]:
    """
    Load and validate agent data using pydantic models if available.
    
    Returns:
        Tuple of (metadata_dict, scores_list, validation_errors)
    """
    errors = []
    metadata_file = agent_dir / "metadata.json"
    scores_file = agent_dir / "scores.json"
    
    if not metadata_file.exists() or not scores_file.exists():
        return None, None, [f"Missing metadata.json or scores.json in {agent_dir}"]
    
    # Load raw JSON
    with open(metadata_file) as f:
        metadata_raw = json.load(f)
    with open(scores_file) as f:
        scores_raw = json.load(f)
    
    # Validate with pydantic if available
    if _ensure_schema_models() and Metadata and ScoreEntry:
        try:
            validated_metadata = Metadata(**metadata_raw)
            # Use mode='json' to serialize enums as strings
            metadata_dict = validated_metadata.model_dump(mode='json')
        except Exception as e:
            errors.append(f"Metadata validation error in {agent_dir.name}: {e}")
            metadata_dict = metadata_raw  # Fall back to raw data
        
        validated_scores = []
        for i, score in enumerate(scores_raw):
            try:
                validated_score = ScoreEntry(**score)
                # Use mode='json' to serialize enums as strings
                validated_dict = validated_score.model_dump(mode='json')
                # Preserve any extra fields from raw data (like full_archive)
                for key, value in score.items():
                    if key not in validated_dict:
                        validated_dict[key] = value
                validated_scores.append(validated_dict)
            except Exception as e:
                errors.append(f"Score entry {i} validation error in {agent_dir.name}: {e}")
                validated_scores.append(score)  # Fall back to raw data
        scores_list = validated_scores
    else:
        # No validation, use raw data
        metadata_dict = metadata_raw
        scores_list = scores_raw
    
    return metadata_dict, scores_list, errors


class SimpleLeaderboardViewer:
    """Simple replacement for agent-eval's LeaderboardViewer."""
    
    def __init__(self, data_dir: str, config: str, split: str):
        """
        Args:
            data_dir: Path to data directory
            config: Config name (e.g., "1.0.0-dev1")
            split: Split name (e.g., "validation" or "test")
        """
        self.data_dir = Path(data_dir)
        self.config = config
        self.split = split
        self.config_path = self.data_dir / config
        
        # Benchmark to category mappings (single source of truth)
        self.benchmark_to_categories = {
            'swe-bench': ['Issue Resolution'],
            'swe-bench-multimodal': ['Frontend'],
            'commit0': ['Greenfield'],
            'swt-bench': ['Testing'],
            'gaia': ['Information Gathering'],
        }
        
        # Build tag map (category -> benchmarks)
        self.tag_map = {}
        for benchmark, categories in self.benchmark_to_categories.items():
            for category in categories:
                if category not in self.tag_map:
                    self.tag_map[category] = []
                if benchmark not in self.tag_map[category]:
                    self.tag_map[category].append(benchmark)
        
    def _load_from_agent_dirs(self):
        """Load data from new agent-centric directory structure (results/YYYYMMDD_model/)."""
        results_dir = self.config_path / "results"
        
        if not results_dir.exists():
            return None  # Fall back to old format
        
        all_records = []
        all_validation_errors = []
        
        # Iterate through each agent directory
        for agent_dir in results_dir.iterdir():
            if not agent_dir.is_dir():
                continue
            
            # Load and validate using pydantic models
            metadata, scores, errors = load_and_validate_agent_data(agent_dir)
            
            if errors:
                all_validation_errors.extend(errors)
            
            if metadata is None or scores is None:
                continue
            
            # Skip entries that are hidden from the leaderboard
            if metadata.get('hide_from_leaderboard', False):
                logger.info(f"Skipping {agent_dir.name}: hide_from_leaderboard is True")
                continue
            
            # Create one record per benchmark (mimicking old JSONL format)
            for score_entry in scores:
                record = {
                    'agent_version': metadata.get('agent_version', 'Unknown'),
                    'llm_base': metadata.get('model', 'unknown'),
                    'openness': metadata.get('openness', 'unknown'),
                    'submission_time': score_entry.get('submission_time', metadata.get('submission_time', '')),
                    'release_date': metadata.get('release_date', ''),  # Model release date
                    'parameter_count_b': metadata.get('parameter_count_b'),  # Total params in billions
                    'active_parameter_count_b': metadata.get('active_parameter_count_b'),  # Active params for MoE
                    'score': score_entry.get('score'),
                    'metric': score_entry.get('metric', 'unknown'),
                    'cost_per_instance': score_entry.get('cost_per_instance'),
                    'average_runtime': score_entry.get('average_runtime'),
                    'tags': [score_entry.get('benchmark')],
                    'full_archive': score_entry.get('full_archive', ''),  # Download URL for trajectories
                    'eval_visualization_page': score_entry.get('eval_visualization_page', ''),  # Laminar visualization URL
                }
                all_records.append(record)
        
        # Log validation errors if any
        if all_validation_errors:
            logger.warning(f"Schema validation errors ({len(all_validation_errors)} total):")
            for error in all_validation_errors[:5]:  # Show first 5
                logger.warning(f"  - {error}")
            if len(all_validation_errors) > 5:
                logger.warning(f"  ... and {len(all_validation_errors) - 5} more")
        
        if not all_records:
            return None  # Fall back to old format
        
        return pd.DataFrame(all_records)
    
    def _load(self):
        """Load data from agent-centric directories and return DataFrame and tag map."""
        df = self._load_from_agent_dirs()
        
        if df is None:
            # Return empty dataframe with error message
            return pd.DataFrame({
                "Message": [f"No data found for split '{self.split}' in results directory"]
            }), {}
        
        # Process the dataframe
        try:
            
            # Transform to expected format for leaderboard
            # Group by agent (version + model combination) to aggregate results across datasets
            transformed_records = []
            
            # Create a unique identifier for each agent (version + model)
            df['agent_id'] = df['agent_version'] + '_' + df['llm_base']
            
            for agent_id in df['agent_id'].unique():
                agent_records = df[df['agent_id'] == agent_id]
                
                # Build a single record for this agent
                first_record = agent_records.iloc[0]
                agent_version = first_record['agent_version']
                
                # Normalize openness to "open" or "closed"
                from aliases import OPENNESS_MAPPING
                raw_openness = first_record['openness']
                normalized_openness = OPENNESS_MAPPING.get(raw_openness, raw_openness)
                
                # All 5 categories for the leaderboard
                ALL_CATEGORIES = ['Issue Resolution', 'Frontend', 'Greenfield', 'Testing', 'Information Gathering']
                
                record = {
                    # Core agent info - use final display names
                    'SDK version': agent_version,  # Will become "SDK Version"
                    'Language model': first_record['llm_base'],  # Will become "Language Model"
                    'openness': normalized_openness,  # Will become "Openness" (simplified to "open" or "closed")
                    'date': first_record['submission_time'],  # Will become "Date"
                    # Model metadata for visualizations
                    'release_date': first_record.get('release_date', ''),  # Model release date
                    'parameter_count_b': first_record.get('parameter_count_b'),  # Total params in billions
                    'active_parameter_count_b': first_record.get('active_parameter_count_b'),  # Active params for MoE
                    # Additional columns expected by the transformer
                    # Use agent_id (version_model) as unique identifier for Pareto frontier calculation
                    'id': agent_id,
                    'source': first_record.get('source', ''),  # Will become "Source"
                    'logs': first_record.get('logs', ''),  # Will become "Logs"
                    'visualization': '',  # Will become "Visualization" - populated below
                }
                
                # Add per-dataset scores and costs
                dataset_scores = []
                dataset_costs = []
                
                # Track category-level data for aggregation
                category_data = {}  # {category: {'scores': [...], 'costs': [], 'runtimes': []}}
                
                for _, row in agent_records.iterrows():
                    tags = row['tags'] if isinstance(row['tags'], list) else [row['tags']]
                    for tag in tags:
                        # Add columns for this specific dataset/benchmark
                        record[f'{tag} score'] = row['score']
                        record[f'{tag} cost'] = row['cost_per_instance']
                        record[f'{tag} runtime'] = row.get('average_runtime')
                        dataset_scores.append(row['score'])
                        dataset_costs.append(row['cost_per_instance'])
                        
                        # Store the full_archive URL for this benchmark (for benchmark-specific download)
                        full_archive_url = row.get('full_archive', '') if hasattr(row, 'get') else row['full_archive'] if 'full_archive' in row.index else ''
                        if full_archive_url:
                            record[f'{tag} download'] = full_archive_url
                        
                        # Store the eval_visualization_page URL for this benchmark (for Laminar visualization)
                        viz_url = row.get('eval_visualization_page', '') if hasattr(row, 'get') else row['eval_visualization_page'] if 'eval_visualization_page' in row.index else ''
                        if viz_url:
                            record[f'{tag} visualization'] = viz_url
                        
                        # Track category-level data for aggregation
                        if tag in self.benchmark_to_categories:
                            for category in self.benchmark_to_categories[tag]:
                                if category not in category_data:
                                    category_data[category] = {'scores': [], 'costs': [], 'runtimes': []}
                                category_data[category]['scores'].append(row['score'])
                                category_data[category]['costs'].append(row['cost_per_instance'])
                                category_data[category]['runtimes'].append(row.get('average_runtime'))
                
                # Calculate category-level aggregates and track average cost/runtime
                all_costs = []
                all_runtimes = []
                categories_with_scores = 0
                for category in ALL_CATEGORIES:
                    if category in category_data and category_data[category]['scores']:
                        data = category_data[category]
                        avg_score = sum(data['scores']) / len(data['scores'])
                        record[f'{category} score'] = avg_score
                        categories_with_scores += 1
                        if data['costs']:
                            valid_costs = [c for c in data['costs'] if c is not None]
                            if valid_costs:
                                avg_cost = sum(valid_costs) / len(valid_costs)
                                record[f'{category} cost'] = avg_cost
                                all_costs.extend(valid_costs)
                        if data['runtimes']:
                            valid_runtimes = [r for r in data['runtimes'] if r is not None]
                            if valid_runtimes:
                                avg_runtime = sum(valid_runtimes) / len(valid_runtimes)
                                record[f'{category} runtime'] = avg_runtime
                                all_runtimes.extend(valid_runtimes)
                    else:
                        # Category not submitted - will show as NA
                        pass
                
                # Calculate average score: always divide by 5 (treating missing categories as 0)
                # This penalizes incomplete submissions
                score_sum = sum(
                    record.get(f'{cat} score', 0) or 0 
                    for cat in ALL_CATEGORIES
                )
                record['average score'] = score_sum / 5
                
                # Average cost per instance across all benchmarks
                record['average cost'] = sum(all_costs) / len(all_costs) if all_costs else None
                
                # Average runtime per instance across all benchmarks
                record['average runtime'] = sum(all_runtimes) / len(all_runtimes) if all_runtimes else None
                
                # Track how many categories were completed
                record['categories_completed'] = categories_with_scores
                
                transformed_records.append(record)
            
            transformed_df = pd.DataFrame(transformed_records)
            
            # Build tag map if not already built
            if not self.tag_map:
                # Create simple tag map from the data
                all_tags = set()
                for _, row in df.iterrows():
                    tags = row['tags'] if isinstance(row['tags'], list) else [row['tags']]
                    all_tags.update(tags)
                
                # Simple mapping: each tag maps to itself
                self.tag_map = {tag: [tag] for tag in sorted(all_tags)}
            
            # DEBUG: Print sample of loaded data
            print(f"[DATA_LOADER] Loaded {len(transformed_df)} agents")
            if len(transformed_df) > 0:
                sample_cols = ['agent_name', 'overall_score', 'overall_cost']
                available_cols = [c for c in sample_cols if c in transformed_df.columns]
                print(f"[DATA_LOADER] Sample row: {transformed_df[available_cols].iloc[0].to_dict()}")
            
            return transformed_df, self.tag_map
        except Exception as e:
            import traceback
            traceback.print_exc()
            return pd.DataFrame({
                "Message": [f"Error loading data: {e}"]
            }), {}
    
    def get_dataframe(self):
        """Get the raw dataframe."""
        df, _ = self._load()
        return df


def load_mock_data_locally(data_dir: str = "mock_results"):
    """
    Load mock data from local directory for testing.
    
    Args:
        data_dir: Path to mock results directory
        
    Returns:
        Dictionary mapping split names to SimpleLeaderboardViewer instances
    """
    viewers = {}
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Warning: Mock data directory '{data_dir}' not found")
        return viewers
    
    # Find all config directories
    for config_dir in data_path.iterdir():
        if config_dir.is_dir():
            config_name = config_dir.name
            
            # Find all JSONL files (each represents a split)
            for jsonl_file in config_dir.glob("*.jsonl"):
                split_name = jsonl_file.stem
                viewer = SimpleLeaderboardViewer(
                    data_dir=str(data_path),
                    config=config_name,
                    split=split_name
                )
                viewers[split_name] = viewer
    
    return viewers
