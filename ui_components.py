import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import os
import base64
import re

from huggingface_hub import HfApi

import aliases
from constants import MARK_BY_CHOICES, MARK_BY_DEFAULT
from simple_data_loader import SimpleLeaderboardViewer
from leaderboard_transformer import (
    DataTransformer,
    transform_raw_dataframe,
    create_pretty_tag_map,
    INFORMAL_TO_FORMAL_NAME_MAP,
    _plot_scatter_plotly,
    format_cost_column,
    format_score_column,
    format_runtime_column,
    format_date_column,
    get_pareto_df,
    clean_llm_base_list,
    get_company_from_model,
    COMPANY_LOGO_MAP,
)
from config import (
    CONFIG_NAME,
    EXTRACTED_DATA_DIR,
    IS_INTERNAL,
    RESULTS_DATASET,
)
from content import (
    create_gradio_anchor_id,
    format_error,
    get_benchmark_description,
    hf_uri_to_web_url,
    hyperlink,
    SCATTER_DISCLAIMER,
)

api = HfApi()
os.makedirs(EXTRACTED_DATA_DIR, exist_ok=True)


def get_company_logo_html(model_name: str) -> str:
    """
    Generates HTML for a company logo based on the model name.
    """
    company_info = get_company_from_model(model_name)
    uri = get_svg_as_data_uri(company_info["path"])
    if uri:
        return f'<img src="{uri}" alt="{company_info["name"]}" title="{company_info["name"]}" style="width:20px; height:20px; vertical-align: middle;">'
    return ""


# Simplified icon map (no tooling distinction, only openness)
# Not actually used since we removed icons from the table, but keeping for potential future use
OPENNESS_ICON_MAP = {
    aliases.CANONICAL_OPENNESS_OPEN: "assets/ellipse-pink.svg",
    aliases.CANONICAL_OPENNESS_CLOSED: "assets/ellipse-yellow.svg",
}

# Add aliases
for canonical_openness, openness_aliases in aliases.OPENNESS_ALIASES.items():
    for openness_alias in openness_aliases:
        OPENNESS_ICON_MAP[openness_alias] = OPENNESS_ICON_MAP[canonical_openness]


OPENNESS_SVG_MAP = {
    aliases.CANONICAL_OPENNESS_OPEN: {
        "path": "assets/ellipse-pink.svg",
        "description": "Open source model"
    },
    aliases.CANONICAL_OPENNESS_CLOSED: {
        "path": "assets/ellipse-yellow.svg",
        "description": "Closed source model"
    },
}

def get_svg_as_data_uri(path: str) -> str:
    """Reads an SVG file and returns it as a base64-encoded data URI."""
    try:
        with open(path, "rb") as svg_file:
            encoded_svg = base64.b64encode(svg_file.read()).decode("utf-8")
            return f"data:image/svg+xml;base64,{encoded_svg}"
    except FileNotFoundError:
        print(f"Warning: SVG file not found at {path}")
        return ""


def build_openness_tooltip_content() -> str:
    """
    Generates the inner HTML for the Model Openness tooltip card using custom SVG lock icons.
    """
    open_uri = get_svg_as_data_uri("assets/lock-open.svg")
    closed_uri = get_svg_as_data_uri("assets/lock-closed.svg")
    html_items = [
        f"""
        <div class="tooltip-legend-item">
            <img src="{open_uri}" alt="Open" style="width: 24px; height: 24px;">
            <div>
                <strong>Open</strong>
                <span>Open source model</span>
            </div>
        </div>
        """,
        f"""
        <div class="tooltip-legend-item">
            <img src="{closed_uri}" alt="Closed" style="width: 24px; height: 24px;">
            <div>
                <strong>Closed</strong>
                <span>Closed source model</span>
            </div>
        </div>
        """
    ]

    joined_items = "".join(html_items)

    return f"""<span class="tooltip-icon-legend">
        ⓘ
        <span class="tooltip-card">
            <h3>Model Openness</h3>
            <p class="tooltip-description">Indicates whether the language model is open source or closed source.</p>
            <div class="tooltip-items-container">{joined_items}</div>
        </span>
    </span>"""


def build_pareto_tooltip_content() -> str:
    """Generates the inner HTML for the Pareto tooltip card with final copy."""
    trophy_uri = get_svg_as_data_uri("assets/trophy.svg")
    trophy_icon_html = f'<img src="{trophy_uri}" style="width: 25px; height: 25px; vertical-align: middle;">'
    return f"""
        <h3>On Pareto Frontier</h3>
        <p class="tooltip-description">The Pareto frontier represents the best balance between score and cost.</p>
        <p class="tooltip-description">Agents on the frontier either:</p>
        <ul class="tooltip-sub-list">
            <li>Offer the lowest cost for a given performance, or</li>
            <li>Deliver the best performance at a given cost.</li>
        </ul>
        <div class="tooltip-description" style="margin-top: 12px; display: flex; align-items: center;">
            <span>These agents are marked with this icon:</span>
            <span>{trophy_icon_html}</span>
        </div>
    """




def build_descriptions_tooltip_content(table) -> str:
    """Generates the inner HTML for the Column Descriptions tooltip card depending on which kind of table."""
    if table == "Overall":
        return """
            <div class="tooltip-description-item"><b>SDK Version:</b> Version of the OpenHands SDK evaluated.</div>
            <div class="tooltip-description-item"><b>Language Model:</b> Language model(s) used by the agent. Hover over ⓘ to view all.</div>
            <div class="tooltip-description-item"><b>Average Score:</b> Sum of category scores divided by 5. Missing categories count as 0.</div>
            <div class="tooltip-description-item"><b>Average Cost:</b> Average cost per instance across all submitted benchmarks, in USD.</div>
            <div class="tooltip-description-item"><b>Issue Resolution Score:</b> Macro-average score across Issue Resolution benchmarks.</div>
            <div class="tooltip-description-item"><b>Issue Resolution Cost:</b> Macro-average cost per instance (USD) across Issue Resolution benchmarks.</div>
            <div class="tooltip-description-item"><b>Frontend Score:</b> Macro-average score across Frontend benchmarks.</div>
            <div class="tooltip-description-item"><b>Frontend Cost:</b> Macro-average cost per instance (USD) across Frontend benchmarks.</div>
            <div class="tooltip-description-item"><b>Greenfield Score:</b> Macro-average score across Greenfield benchmarks.</div>
            <div class="tooltip-description-item"><b>Greenfield Cost:</b> Macro-average cost per instance (USD) across Greenfield benchmarks.</div>
            <div class="tooltip-description-item"><b>Testing Score:</b> Macro-average score across Testing benchmarks.</div>
            <div class="tooltip-description-item"><b>Testing Cost:</b> Macro-average cost per instance (USD) across Testing benchmarks.</div>
            <div class="tooltip-description-item"><b>Information Gathering Score:</b> Macro-average score across Information Gathering benchmarks.</div>
            <div class="tooltip-description-item"><b>Information Gathering Cost:</b> Macro-average cost per instance (USD) across Information Gathering benchmarks.</div>
            <div class="tooltip-description-item"><b>Categories Attempted:</b> Number of core categories with at least one benchmark attempted (out of 5).</div>
            <div class="tooltip-description-item"><b>Logs:</b> View evaluation run logs (e.g., outputs, traces).</div>
            <div class="tooltip-description-item"><b>Visualization:</b> View interactive evaluation visualization (when available).</div>
        """
    elif table in ["Issue Resolution", "Frontend", "Greenfield", "Testing", "Information Gathering"]:
        return f"""
            <div class="tooltip-description-item"><b>SDK Version:</b> Version of the OpenHands agent evaluated.</div>
            <div class="tooltip-description-item"><b>Language Model:</b> Language model(s) used by the agent. Hover over ⓘ to view all.</div>
            <div class="tooltip-description-item"><b>{table} Score:</b> Macro-average score across {table} benchmarks.</div>
            <div class="tooltip-description-item"><b>{table} Cost:</b> Macro-average cost per problem (USD) across {table} benchmarks.</div>
            <div class="tooltip-description-item"><b>Benchmark Score:</b> Average (mean) score on the benchmark.</div>
            <div class="tooltip-description-item"><b>Benchmark Cost:</b> Average (mean) cost per problem (USD) on the benchmark.</div>
            <div class="tooltip-description-item"><b>Benchmarks Attempted:</b> Number of benchmarks attempted in this category (e.g., 3/5).</div>
            <div class="tooltip-description-item"><b>Logs:</b> View evaluation run logs (e.g., outputs, traces).</div>
            <div class="tooltip-description-item"><b>📊 Visualization:</b> View interactive evaluation visualization (when available).</div>
            <div class="tooltip-description-item"><b>Download:</b> Download evaluation trajectories archive.</div>
        """
    else:
        # Fallback for any other table type, e.g., individual benchmarks
        return f"""
            <div class="tooltip-description-item"><b>SDK Version:</b> Version of the OpenHands agent evaluated.</div>
            <div class="tooltip-description-item"><b>Language Model:</b> Language model(s) used by the agent. Hover over ⓘ to view all.</div>
            <div class="tooltip-description-item"><b>Benchmark Attempted:</b> Indicates whether the agent attempted this benchmark.</div>
            <div class="tooltip-description-item"><b>{table} Score:</b> Score achieved by the agent on this benchmark.</div>
            <div class="tooltip-description-item"><b>{table} Cost:</b> Cost incurred by the agent to solve this benchmark (in USD).</div>
            <div class="tooltip-description-item"><b>Logs:</b> View evaluation run logs (e.g., outputs, traces).</div>
            <div class="tooltip-description-item"><b>📊 Visualization:</b> View interactive evaluation visualization (when available).</div>
            <div class="tooltip-description-item"><b>Download:</b> Download evaluation trajectories archive.</div>
        """

# Create HTML for the "Openness" legend items for table using custom SVG lock icons
open_lock_uri = get_svg_as_data_uri("assets/lock-open.svg")
closed_lock_uri = get_svg_as_data_uri("assets/lock-closed.svg")
openness_html_items = [
    f'<div style="display: flex; align-items: center; white-space: nowrap;">'
    f'<img src="{open_lock_uri}" alt="Open" style="width:16px; height:16px; margin-right: 4px;">'
    f'<span>Open</span>'
    f'</div>',
    f'<div style="display: flex; align-items: center; white-space: nowrap;">'
    f'<img src="{closed_lock_uri}" alt="Closed" style="width:16px; height:16px; margin-right: 4px;">'
    f'<span>Closed</span>'
    f'</div>'
]
openness_html = " ".join(openness_html_items)

pareto_tooltip_content = build_pareto_tooltip_content()
openness_tooltip_content = build_openness_tooltip_content()

def create_legend_markdown(which_table: str) -> str:
    """
    Generates the complete HTML for the legend section, including tooltips.
    This is used in the main leaderboard display.
    """
    descriptions_tooltip_content = build_descriptions_tooltip_content(which_table)
    trophy_uri = get_svg_as_data_uri("assets/trophy.svg")
    
    # Add download section for benchmark-specific tables (not Overall or category pages)
    download_section = ""
    if which_table not in ["Overall", "Issue Resolution", "Frontend", "Greenfield", "Testing", "Information Gathering"]:
        download_section = """
        <div> <!-- Container for the Download section -->
            <b>Download</b>
            <div class="table-legend-item">
                <span style="font-size: 16px; margin-right: 4px;">⬇️</span>
                <span>Trajectories</span>
            </div>
        </div>
        """
    
    legend_markdown = f"""
    <div style="display: flex; flex-wrap: wrap; align-items: flex-start; gap: 20px; font-size: 14px; padding-bottom: 8px;">
            
        <div> <!-- Container for the Pareto section -->
            <b>Pareto</b>
            <span class="tooltip-icon-legend">
                ⓘ
                <span class="tooltip-card">{pareto_tooltip_content}</span>
            </span>
            <div class="table-legend-item">
                <img src="{trophy_uri}" alt="On Frontier" style="width:20px; height:20px; margin-right: 4px; flex-shrink: 0;">
                <span>On frontier</span>
            </div>
        </div>
    
        <div> <!-- Container for the Openness section -->
            <b>Model Openness</b>
            {openness_tooltip_content}
            <div class="table-legend-item">{openness_html}</div>
        </div>
        
        {download_section}
        
        <div><!-- Container for the Column Descriptions section -->
            <b>Column Descriptions</b>
            <span class="tooltip-icon-legend">
                ⓘ
                <span class="tooltip-card">
                    <h3>Column Descriptions</h3>
                    <div class="tooltip-items-container">{descriptions_tooltip_content}</div>
                </span>
            </span>
        </div>
    </div>
    """
    return legend_markdown

# Create HTML for plot legend with company logos
company_legend_items = []
# Show a sample of company logos in the legend
sample_companies = [
    ("Anthropic", "assets/logo-anthropic.svg"),
    ("OpenAI", "assets/logo-openai.svg"),
    ("Google", "assets/logo-google.svg"),
    ("Meta", "assets/logo-meta.svg"),
    ("Mistral", "assets/logo-mistral.svg"),
]
for name, path in sample_companies:
    uri = get_svg_as_data_uri(path)
    if uri:
        company_legend_items.append(
            f'<div class="plot-legend-item">'
                f'<img class="plot-legend-item-svg" src="{uri}" alt="{name}" title="{name}" style="width: 20px; height: 20px;">'
                f'<span>{name}</span>'
            f'</div>'
        )

plot_legend_html = f"""
    <div class="plot-legend-container">
        <div id="plot-legend-logo">
            <img src="{get_svg_as_data_uri("assets/logo.svg")}">
        </div>
        <div style="margin-bottom: 16px;">
            <span class="plot-legend-category-heading">Pareto</span>
            <div style="margin-top: 8px;">
                <div class="plot-legend-item">
                    <img id="plot-legend-item-pareto-svg" class="plot-legend-item-svg" src="{get_svg_as_data_uri("assets/pareto.svg")}">
                    <span>On frontier</span>
                </div>
            </div>
        </div>
        <div>
            <span class="plot-legend-category-heading">Company Logos</span>
            <div style="margin-top: 8px;">
                {''.join(company_legend_items)}
            </div>
        </div>
    </div>
""";

# --- Global State for Viewers (simple caching with TTL) ---
CACHED_VIEWERS = {}
CACHED_TAG_MAPS = {}
_cache_lock = __import__('threading').Lock()
_data_version = 0  # Incremented when data is refreshed


def get_data_version():
    """Get the current data version number."""
    global _data_version
    return _data_version


def clear_viewer_cache():
    """
    Clear all cached viewers and tag maps.
    Called when data is refreshed from the background scheduler.
    """
    global CACHED_VIEWERS, CACHED_TAG_MAPS, _data_version
    with _cache_lock:
        CACHED_VIEWERS.clear()
        CACHED_TAG_MAPS.clear()
        _data_version += 1
        print(f"[CACHE] Viewer cache cleared after data refresh (version: {_data_version})")


# Register the cache clear callback with the data refresh system
try:
    from setup_data import register_refresh_callback
    register_refresh_callback(clear_viewer_cache)
except ImportError:
    pass  # setup_data may not be available during import


# Category definitions for Winners by Category section
CATEGORY_DEFINITIONS = [
    {"name": "Issue Resolution", "score_col": "Issue Resolution Score", "icon": "bug-fixing.svg", "emoji": "🐛"},
    {"name": "Greenfield", "score_col": "Greenfield Score", "icon": "app-creation.svg", "emoji": "🌱"},
    {"name": "Frontend", "score_col": "Frontend Score", "icon": "frontend-development.svg", "emoji": "🎨"},
    {"name": "Testing", "score_col": "Testing Score", "icon": "test-generation.svg", "emoji": "🧪"},
    {"name": "Information Gathering", "score_col": "Information Gathering Score", "icon": "information-gathering.svg", "emoji": "🔍"},
]


def get_winners_by_category(df: pd.DataFrame, top_n: int = 5) -> dict:
    """
    Extract the top N models for each category based on their score.
    Deduplicates by model name, keeping only the highest score per model.
    
    Args:
        df: The full leaderboard DataFrame with all scores
        top_n: Number of top models to return per category (default: 5)
        
    Returns:
        Dictionary mapping category name to list of top models with their scores
    """
    winners = {}
    
    for cat_def in CATEGORY_DEFINITIONS:
        cat_name = cat_def["name"]
        score_col = cat_def["score_col"]
        
        if score_col not in df.columns:
            winners[cat_name] = []
            continue
        
        # Filter to rows that have a valid score for this category
        cat_df = df[df[score_col].notna()].copy()
        
        if cat_df.empty:
            winners[cat_name] = []
            continue
        
        # Clean model names for deduplication
        def clean_model_name(model):
            if isinstance(model, list):
                model = model[0] if model else "Unknown"
            return str(model).split('/')[-1]
        
        cat_df['_clean_model'] = cat_df['Language Model'].apply(clean_model_name)
        
        # Deduplicate by model name, keeping highest score
        cat_df = cat_df.sort_values(score_col, ascending=False)
        cat_df = cat_df.drop_duplicates(subset=['_clean_model'], keep='first')
        
        # Take top N after dedup
        cat_df = cat_df.head(top_n)
        
        # Extract relevant info
        top_models = []
        for rank, (_, row) in enumerate(cat_df.iterrows(), 1):
            model_info = {
                "rank": rank,
                "language_model": row.get("Language Model", "Unknown"),
                "score": row[score_col],
            }
            top_models.append(model_info)
        
        winners[cat_name] = top_models
    
    return winners


def create_winners_by_category_html(df: pd.DataFrame, top_n: int = 5) -> str:
    """
    Create a single HTML table displaying the top N winners for each category side by side.
    Format: | [emoji] Category | Score | [emoji] Category | Score | ...
    
    Args:
        df: The full leaderboard DataFrame
        top_n: Number of top models to show per category
        
    Returns:
        HTML string with the winners table
    """
    winners = get_winners_by_category(df, top_n)
    
    # Build a single unified table
    html_parts = ['<div class="winners-by-category-container">']
    html_parts.append('<table class="winners-unified-table">')
    
    # Header row with category emojis and names
    html_parts.append('<thead><tr>')
    for cat_def in CATEGORY_DEFINITIONS:
        cat_name = cat_def["name"]
        cat_emoji = cat_def["emoji"]
        html_parts.append(f'''
            <th class="category-header" colspan="2">
                {cat_emoji} {cat_name}
            </th>
        ''')
    html_parts.append('</tr></thead>')
    
    # Body rows - one row per rank position
    html_parts.append('<tbody>')
    for rank in range(1, top_n + 1):
        html_parts.append('<tr>')
        
        for cat_def in CATEGORY_DEFINITIONS:
            cat_name = cat_def["name"]
            top_models = winners.get(cat_name, [])
            
            # Find the model at this rank
            model_at_rank = None
            for m in top_models:
                if m["rank"] == rank:
                    model_at_rank = m
                    break
            
            if model_at_rank:
                language_model = model_at_rank["language_model"]
                score = model_at_rank["score"]
                
                # Format model name - clean it if it's a list
                if isinstance(language_model, list):
                    language_model = language_model[0] if language_model else "Unknown"
                model_display = str(language_model).split('/')[-1]
                
                # Add medal emoji for top 3 (after model name)
                rank_suffix = ""
                if rank == 1:
                    rank_suffix = " 🥇"
                elif rank == 2:
                    rank_suffix = " 🥈"
                elif rank == 3:
                    rank_suffix = " 🥉"
                
                html_parts.append(f'''
                    <td class="score-cell">{score:.1f}</td>
                    <td class="model-cell">{model_display}{rank_suffix}</td>
                ''')
            else:
                html_parts.append('<td class="score-cell">-</td><td class="model-cell">-</td>')
        
        html_parts.append('</tr>')
    
    html_parts.append('</tbody></table></div>')
    
    return ''.join(html_parts)


class DummyViewer:
    """A mock viewer to be cached on error. It has a ._load() method
       to ensure it behaves like the real LeaderboardViewer."""
    def __init__(self, error_df):
        self._error_df = error_df

    def _load(self):
        # The _load method returns the error DataFrame and an empty tag map
        return self._error_df, {}

def get_leaderboard_viewer_instance(split: str):
    """
    Fetches the LeaderboardViewer for a split, using a thread-safe cache to avoid
    re-downloading data. On error, returns a stable DummyViewer object.
    """
    global CACHED_VIEWERS, CACHED_TAG_MAPS

    with _cache_lock:
        if split in CACHED_VIEWERS:
            # Cache hit: return the cached viewer and tag map
            return CACHED_VIEWERS[split], CACHED_TAG_MAPS.get(split, {"Overall": []})

    # --- Cache miss: try to load data from the source ---
    try:
        # First try to load from extracted data directory (local mock data)
        data_dir = EXTRACTED_DATA_DIR if os.path.exists(EXTRACTED_DATA_DIR) else "mock_results"
        
        print(f"Loading data for split '{split}' from: {data_dir}/{CONFIG_NAME}")
        viewer = SimpleLeaderboardViewer(
            data_dir=data_dir,
            config=CONFIG_NAME,
            split=split
        )

        # Simplify tag map creation
        pretty_tag_map = create_pretty_tag_map(viewer.tag_map, INFORMAL_TO_FORMAL_NAME_MAP)

        # Cache the results for next time (thread-safe)
        with _cache_lock:
            CACHED_VIEWERS[split] = viewer
            CACHED_TAG_MAPS[split] = pretty_tag_map  # Cache the pretty map directly

        return viewer, pretty_tag_map

    except Exception as e:
        # On ANY error, create a consistent error message and cache a DummyViewer
        error_message = f"Error loading data for split '{split}': {e}"
        print(format_error(error_message))

        dummy_df = pd.DataFrame({"Message": [error_message]})
        dummy_viewer = DummyViewer(dummy_df)
        dummy_tag_map = {"Overall": []}

        # Cache the dummy objects so we don't try to fetch again on this run
        with _cache_lock:
            CACHED_VIEWERS[split] = dummy_viewer
            CACHED_TAG_MAPS[split] = dummy_tag_map

        return dummy_viewer, dummy_tag_map


def create_leaderboard_display(
        full_df: pd.DataFrame,
        tag_map: dict,
        category_name: str,
        split_name: str
):
    """
    This UI factory takes pre-loaded data and renders the main DataFrame and Plot
    for a given category (e.g., "Overall" or "Literature Understanding").
    
    The display includes a timer that periodically checks for data updates and
    refreshes the UI when new data is available.
    """
    # Track the data version when this display was created
    initial_data_version = get_data_version()
    
    # 1. Instantiate the transformer and get the specific view for this category.
    # The function no longer loads data itself; it filters the data it receives.
    transformer = DataTransformer(full_df, tag_map)
    df_view_full, plots_dict = transformer.view(tag=category_name, use_plotly=True)
    
    def prepare_df_for_display(df_view):
        """Prepare a DataFrame for display with all formatting applied."""
        df_display = df_view.copy()
        
        # Get Pareto frontier info
        pareto_df = get_pareto_df(df_display)
        trophy_uri = get_svg_as_data_uri("assets/trophy.svg")
        if not pareto_df.empty and 'id' in pareto_df.columns:
            pareto_agent_names = pareto_df['id'].tolist()
        else:
            pareto_agent_names = []

        for col in df_display.columns:
            if "Cost" in col:
                df_display = format_cost_column(df_display, col)

        for col in df_display.columns:
            if "Score" in col:
                df_display = format_score_column(df_display, col)
        
        for col in df_display.columns:
            if "Runtime" in col:
                df_display = format_runtime_column(df_display, col)
        
        # Format Date column to show only date (not time)
        if "Date" in df_display.columns:
            df_display = format_date_column(df_display, "Date")
        
        # Clean the Language Model column first
        df_display['Language Model'] = df_display['Language Model'].apply(clean_llm_base_list)
        
        # Now combine icons with Language Model column
        def format_language_model_with_icons(row):
            icons_html = ''
            
            # Add Pareto trophy if on frontier
            if row['id'] in pareto_agent_names:
                icons_html += f'<img src="{trophy_uri}" alt="On Pareto Frontier" title="On Pareto Frontier" style="width:18px; height:18px;">'
            
            # Add openness lock icon
            openness_val = row.get('Openness', '')
            if openness_val in [aliases.CANONICAL_OPENNESS_OPEN, 'Open', 'Open Source', 'Open Source + Open Weights']:
                lock_uri = get_svg_as_data_uri("assets/lock-open.svg")
                icons_html += f'<img src="{lock_uri}" alt="Open" title="Open source model" style="width:16px; height:16px;">'
            else:
                lock_uri = get_svg_as_data_uri("assets/lock-closed.svg")
                icons_html += f'<img src="{lock_uri}" alt="Closed" title="Closed source model" style="width:16px; height:16px;">'
            
            # Add company logo
            company_html = get_company_logo_html(row['Language Model'])
            if company_html:
                icons_html += company_html
            
            # Format the model name
            model_name = row['Language Model']
            if isinstance(model_name, list):
                if len(model_name) > 1:
                    tooltip_text = "\\n".join(map(str, model_name))
                    model_text = f'<span class="tooltip-icon cell-tooltip-icon" style="cursor: help;" data-tooltip="{tooltip_text}">{model_name[0]} (+ {len(model_name) - 1}) ⓘ</span>'
                elif len(model_name) == 1:
                    model_text = model_name[0]
                else:
                    model_text = str(model_name)
            else:
                model_text = str(model_name)
            
            # Wrap in a flex container to keep icons horizontal
            return f'<div style="display:flex; align-items:center; gap:4px; flex-wrap:nowrap;">{icons_html}<span>{model_text}</span></div>'
        
        df_display['Language Model'] = df_display.apply(format_language_model_with_icons, axis=1)
        
        if 'Source' in df_display.columns:
            df_display['SDK Version'] = df_display.apply(
                lambda row: f"{row['SDK Version']} {row['Source']}" if pd.notna(row['Source']) and row['Source'] else row['SDK Version'],
                axis=1
            )
        
        columns_to_drop = ['id', 'Openness', 'Agent Tooling', 'Source']
        df_display = df_display.drop(columns=columns_to_drop, errors='ignore')
        
        return df_display
    
    # Prepare both complete and all entries versions
    # Complete entries have all 5 categories submitted
    # The 'Categories Attempted' column is formatted as "X/5"
    if 'Categories Attempted' in df_view_full.columns:
        df_view_complete = df_view_full[df_view_full['Categories Attempted'] == '5/5'].copy()
    else:
        df_view_complete = df_view_full.copy()
    
    # Prepare open-only filtered versions (filter before prepare_df_for_display drops Openness column)
    if 'Openness' in df_view_full.columns:
        df_view_open = df_view_full[df_view_full['Openness'].str.lower() == 'open'].copy()
        df_view_complete_open = df_view_complete[df_view_complete['Openness'].str.lower() == 'open'].copy()
    else:
        df_view_open = df_view_full.copy()
        df_view_complete_open = df_view_complete.copy()
    
    df_display_complete = prepare_df_for_display(df_view_complete)
    df_display_all = prepare_df_for_display(df_view_full)
    df_display_open = prepare_df_for_display(df_view_open)
    df_display_complete_open = prepare_df_for_display(df_view_complete_open)
    
    # If no complete entries exist, show all entries by default
    has_complete_entries = len(df_display_complete) > 0
    
    # Determine primary score/cost/runtime columns for scatter plot
    if category_name == "Overall":
        primary_score_col = "Average Score"
        primary_cost_col = "Average Cost"
        primary_runtime_col = "Average Runtime"
    else:
        primary_score_col = f"{category_name} Score"
        primary_cost_col = f"{category_name} Cost"
        primary_runtime_col = f"{category_name} Runtime"
    
    # Function to create cost/performance scatter plot from data
    def create_cost_scatter_plot(df_data, mark_by=MARK_BY_DEFAULT):
        return _plot_scatter_plotly(
            data=df_data,
            x=primary_cost_col if primary_cost_col in df_data.columns else None,
            y=primary_score_col if primary_score_col in df_data.columns else "Average Score",
            agent_col="SDK Version",
            name=category_name,
            plot_type='cost',
            mark_by=mark_by
        )
    
    # Function to create runtime/performance scatter plot from data
    def create_runtime_scatter_plot(df_data, mark_by=MARK_BY_DEFAULT):
        return _plot_scatter_plotly(
            data=df_data,
            x=primary_runtime_col if primary_runtime_col in df_data.columns else None,
            y=primary_score_col if primary_score_col in df_data.columns else "Average Score",
            agent_col="SDK Version",
            name=category_name,
            plot_type='runtime',
            mark_by=mark_by
        )
    
    # Create initial cost scatter plots for all filter combinations
    cost_scatter_complete = create_cost_scatter_plot(df_view_complete) if has_complete_entries else go.Figure()
    cost_scatter_all = create_cost_scatter_plot(df_view_full)
    cost_scatter_open = create_cost_scatter_plot(df_view_open) if len(df_view_open) > 0 else go.Figure()
    cost_scatter_complete_open = create_cost_scatter_plot(df_view_complete_open) if len(df_view_complete_open) > 0 else go.Figure()
    
    # Create initial runtime scatter plots for all filter combinations
    runtime_scatter_complete = create_runtime_scatter_plot(df_view_complete) if has_complete_entries else go.Figure()
    runtime_scatter_all = create_runtime_scatter_plot(df_view_full)
    runtime_scatter_open = create_runtime_scatter_plot(df_view_open) if len(df_view_open) > 0 else go.Figure()
    runtime_scatter_complete_open = create_runtime_scatter_plot(df_view_complete_open) if len(df_view_complete_open) > 0 else go.Figure()
    
    # Now get headers from the renamed dataframe (use all entries to ensure headers are present)
    df_headers = df_display_all.columns.tolist()
    df_datatypes = []
    for col in df_headers:
        if col in ["Logs", "Visualization"] or "Cost" in col or "Score" in col or "Runtime" in col:
            df_datatypes.append("markdown")
        elif col in ["SDK Version", "Language Model"]:
            df_datatypes.append("html")
        else:
            df_datatypes.append("str")
    # Dynamically set widths for the DataFrame columns
    # Order: Language Model, SDK Version, Average Score, Average Cost, Average Runtime, ...
    fixed_start_widths = [280, 100, 100, 90]  # Language Model (with icons), SDK Version, Average Score, Average Runtime
    num_score_cost_runtime_cols = 0
    remaining_headers = df_headers[len(fixed_start_widths):]
    for col in remaining_headers:
        if "Score" in col or "Cost" in col or "Runtime" in col:
            num_score_cost_runtime_cols += 1
    dynamic_widths = [90] * num_score_cost_runtime_cols
    fixed_end_widths = [90, 100, 120, 120]  # Categories Attempted, Date, Logs, Visualization
    # 5. Combine all the lists to create the final, fully dynamic list.
    final_column_widths = fixed_start_widths + dynamic_widths + fixed_end_widths

    # Calculate counts for the checkbox labels
    num_complete = len(df_display_complete)
    num_total = len(df_display_all)
    num_incomplete = num_total - num_complete
    num_open = len(df_display_open)
    num_closed = num_total - num_open
    
    # Add toggle checkboxes and dropdown ABOVE the plot
    with gr.Row():
        with gr.Column(scale=3):
            if has_complete_entries:
                show_incomplete_checkbox = gr.Checkbox(
                    label=f"Show incomplete entries ({num_incomplete} entries with fewer than 5 categories)",
                    value=False,
                    elem_id="show-incomplete-toggle"
                )
            else:
                show_incomplete_checkbox = None
                gr.Markdown(f"*No entries with all 5 categories completed yet. Showing all {num_total} entries.*")
            
            # Add checkbox for open models only (always show this if there are open models)
            if num_open > 0 and num_closed > 0:
                show_open_only_checkbox = gr.Checkbox(
                    label=f"Show only open models ({num_open} open, {num_closed} closed)",
                    value=False,
                    elem_id="show-open-only-toggle"
                )
            else:
                show_open_only_checkbox = None
        
        with gr.Column(scale=1):
            mark_by_dropdown = gr.Dropdown(
                choices=MARK_BY_CHOICES,
                value=MARK_BY_DEFAULT,
                label="Mark systems by",
                elem_id="mark-by-dropdown"
            )

    # Plot components - show complete entries by default if available
    # Cost/Performance plot
    initial_cost_plot = cost_scatter_complete if has_complete_entries else cost_scatter_all
    cost_plot_component = gr.Plot(
        value=initial_cost_plot,
        show_label=False,
    )
    
    # Runtime/Performance plot
    initial_runtime_plot = runtime_scatter_complete if has_complete_entries else runtime_scatter_all
    runtime_plot_component = gr.Plot(
        value=initial_runtime_plot,
        show_label=False,
    )
    gr.Markdown(value=SCATTER_DISCLAIMER, elem_id="scatter-disclaimer")

    # Put table and key into an accordion
    with gr.Accordion("Show / Hide Table View", open=True, elem_id="leaderboard-accordion"):
        # If there are complete entries, show toggle. If not, show all entries.
        if has_complete_entries:
            # Start with complete entries only (default)
            dataframe_component = gr.DataFrame(
                headers=df_headers,
                value=df_display_complete,
                datatype=df_datatypes,
                interactive=False,
                wrap=True,
                column_widths=final_column_widths,
                elem_classes=["wrap-header-df"],
                show_search="search",
                elem_id="main-leaderboard"
            )
            
            # Update function for filters - handles checkboxes and mark_by dropdown
            def update_display(show_incomplete, show_open_only, mark_by):
                # Determine which dataframe to show based on checkbox states
                if show_open_only:
                    df_to_show = df_display_open if show_incomplete else df_display_complete_open
                    view_df = df_view_open if show_incomplete else df_view_complete_open
                else:
                    df_to_show = df_display_all if show_incomplete else df_display_complete
                    view_df = df_view_full if show_incomplete else df_view_complete
                
                # Regenerate plots with current mark_by setting
                cost_plot = create_cost_scatter_plot(view_df, mark_by)
                runtime_plot = create_runtime_scatter_plot(view_df, mark_by)
                return df_to_show, cost_plot, runtime_plot
            
            # Connect checkboxes and dropdown to the update function
            filter_inputs = [show_incomplete_checkbox]
            if show_open_only_checkbox is not None:
                filter_inputs.append(show_open_only_checkbox)
            else:
                # Add a dummy value for show_open_only when checkbox doesn't exist
                filter_inputs = [show_incomplete_checkbox, gr.State(value=False)]
            filter_inputs.append(mark_by_dropdown)
            
            show_incomplete_checkbox.change(
                fn=update_display,
                inputs=filter_inputs,
                outputs=[dataframe_component, cost_plot_component, runtime_plot_component]
            )
            if show_open_only_checkbox is not None:
                show_open_only_checkbox.change(
                    fn=update_display,
                    inputs=filter_inputs,
                    outputs=[dataframe_component, cost_plot_component, runtime_plot_component]
                )
            mark_by_dropdown.change(
                fn=update_display,
                inputs=filter_inputs,
                outputs=[dataframe_component, cost_plot_component, runtime_plot_component]
            )
        else:
            dataframe_component = gr.DataFrame(
                headers=df_headers,
                value=df_display_all,
                datatype=df_datatypes,
                interactive=False,
                wrap=True,
                column_widths=final_column_widths,
                elem_classes=["wrap-header-df"],
                show_search="search",
                elem_id="main-leaderboard"
            )
            
            # Update function for mark_by and optional open_only checkbox
            def update_display_no_complete(show_open_only, mark_by):
                if show_open_only:
                    df_to_show = df_display_open
                    view_df = df_view_open
                else:
                    df_to_show = df_display_all
                    view_df = df_view_full
                cost_plot = create_cost_scatter_plot(view_df, mark_by)
                runtime_plot = create_runtime_scatter_plot(view_df, mark_by)
                return df_to_show, cost_plot, runtime_plot
            
            filter_inputs_no_complete = []
            if show_open_only_checkbox is not None:
                filter_inputs_no_complete.append(show_open_only_checkbox)
            else:
                filter_inputs_no_complete.append(gr.State(value=False))
            filter_inputs_no_complete.append(mark_by_dropdown)
            
            if show_open_only_checkbox is not None:
                show_open_only_checkbox.change(
                    fn=update_display_no_complete,
                    inputs=filter_inputs_no_complete,
                    outputs=[dataframe_component, cost_plot_component, runtime_plot_component]
                )
            mark_by_dropdown.change(
                fn=update_display_no_complete,
                inputs=filter_inputs_no_complete,
                outputs=[dataframe_component, cost_plot_component, runtime_plot_component]
            )
        
        legend_markdown = create_legend_markdown(category_name)
        gr.HTML(value=legend_markdown, elem_id="legend-markdown")
    
    # Add a timer to periodically check for data updates and refresh the UI
    # This runs every 60 seconds to check if new data is available
    def check_and_refresh_data(show_incomplete, show_open_only=False, mark_by=MARK_BY_DEFAULT):
        """Check if data has been refreshed and return updated data if so."""
        current_version = get_data_version()
        if current_version > initial_data_version:
            # Data has been refreshed, reload it
            print(f"[REFRESH] Data version changed from {initial_data_version} to {current_version}, reloading...")
            new_df, new_tag_map = get_full_leaderboard_data(split_name)
            if not new_df.empty:
                new_transformer = DataTransformer(new_df, new_tag_map)
                new_df_view_full, _ = new_transformer.view(tag=category_name, use_plotly=True)
                
                # Prepare both complete and all entries versions
                if 'Categories Attempted' in new_df_view_full.columns:
                    new_df_view_complete = new_df_view_full[new_df_view_full['Categories Attempted'] == '5/5'].copy()
                else:
                    new_df_view_complete = new_df_view_full.copy()
                
                # Prepare open-only versions
                if 'Openness' in new_df_view_full.columns:
                    new_df_view_open = new_df_view_full[new_df_view_full['Openness'].str.lower() == 'open'].copy()
                    new_df_view_complete_open = new_df_view_complete[new_df_view_complete['Openness'].str.lower() == 'open'].copy()
                else:
                    new_df_view_open = new_df_view_full.copy()
                    new_df_view_complete_open = new_df_view_complete.copy()
                
                new_df_display_complete = prepare_df_for_display(new_df_view_complete)
                new_df_display_all = prepare_df_for_display(new_df_view_full)
                new_df_display_open = prepare_df_for_display(new_df_view_open)
                new_df_display_complete_open = prepare_df_for_display(new_df_view_complete_open)
                
                # Create new scatter plots for all combinations (with current mark_by)
                new_cost_scatter_complete = create_cost_scatter_plot(new_df_view_complete, mark_by) if len(new_df_display_complete) > 0 else go.Figure()
                new_cost_scatter_all = create_cost_scatter_plot(new_df_view_full, mark_by)
                new_cost_scatter_open = create_cost_scatter_plot(new_df_view_open, mark_by) if len(new_df_view_open) > 0 else go.Figure()
                new_cost_scatter_complete_open = create_cost_scatter_plot(new_df_view_complete_open, mark_by) if len(new_df_view_complete_open) > 0 else go.Figure()
                
                new_runtime_scatter_complete = create_runtime_scatter_plot(new_df_view_complete, mark_by) if len(new_df_display_complete) > 0 else go.Figure()
                new_runtime_scatter_all = create_runtime_scatter_plot(new_df_view_full, mark_by)
                new_runtime_scatter_open = create_runtime_scatter_plot(new_df_view_open, mark_by) if len(new_df_view_open) > 0 else go.Figure()
                new_runtime_scatter_complete_open = create_runtime_scatter_plot(new_df_view_complete_open, mark_by) if len(new_df_view_complete_open) > 0 else go.Figure()
                
                # Return the appropriate data based on checkbox states
                if show_open_only:
                    if show_incomplete:
                        return new_df_display_open, new_cost_scatter_open, new_runtime_scatter_open
                    else:
                        return new_df_display_complete_open, new_cost_scatter_complete_open, new_runtime_scatter_complete_open
                else:
                    if show_incomplete:
                        return new_df_display_all, new_cost_scatter_all, new_runtime_scatter_all
                    else:
                        return new_df_display_complete, new_cost_scatter_complete, new_runtime_scatter_complete
        
        # No change, return current values based on checkbox states
        if show_open_only:
            if show_incomplete:
                return df_display_open, cost_scatter_open, runtime_scatter_open
            else:
                return df_display_complete_open, cost_scatter_complete_open, runtime_scatter_complete_open
        else:
            if show_incomplete:
                return df_display_all, cost_scatter_all, runtime_scatter_all
            else:
                return df_display_complete, cost_scatter_complete, runtime_scatter_complete
    
    # Create a timer that checks for updates every 60 seconds
    refresh_timer = gr.Timer(value=60)
    
    # Connect the timer to the refresh function
    if show_incomplete_checkbox is not None:
        timer_inputs = [show_incomplete_checkbox]
        if show_open_only_checkbox is not None:
            timer_inputs.append(show_open_only_checkbox)
        timer_inputs.append(mark_by_dropdown)  # Always include mark_by
        refresh_timer.tick(
            fn=check_and_refresh_data,
            inputs=timer_inputs,
            outputs=[dataframe_component, cost_plot_component, runtime_plot_component]
        )
    else:
        # If no incomplete checkbox, always show all data (but still filter by open if needed)
        def check_and_refresh_all(show_open_only=False, mark_by=MARK_BY_DEFAULT):
            current_version = get_data_version()
            if current_version > initial_data_version:
                print(f"[REFRESH] Data version changed, reloading...")
                new_df, new_tag_map = get_full_leaderboard_data(split_name)
                if not new_df.empty:
                    new_transformer = DataTransformer(new_df, new_tag_map)
                    new_df_view_full, _ = new_transformer.view(tag=category_name, use_plotly=True)
                    
                    if show_open_only and 'Openness' in new_df_view_full.columns:
                        new_df_view_full = new_df_view_full[new_df_view_full['Openness'].str.lower() == 'open'].copy()
                    
                    new_df_display_all = prepare_df_for_display(new_df_view_full)
                    new_cost_scatter_all = create_cost_scatter_plot(new_df_view_full, mark_by)
                    new_runtime_scatter_all = create_runtime_scatter_plot(new_df_view_full, mark_by)
                    return new_df_display_all, new_cost_scatter_all, new_runtime_scatter_all
            
            if show_open_only:
                return df_display_open, cost_scatter_open, runtime_scatter_open
            return df_display_all, cost_scatter_all, runtime_scatter_all
        
        if show_open_only_checkbox is not None:
            refresh_timer.tick(
                fn=check_and_refresh_all,
                inputs=[show_open_only_checkbox, mark_by_dropdown],
                outputs=[dataframe_component, cost_plot_component, runtime_plot_component]
            )
        else:
            def check_and_refresh_simple(mark_by=MARK_BY_DEFAULT):
                return check_and_refresh_all(False, mark_by)
            refresh_timer.tick(
                fn=check_and_refresh_simple,
                inputs=[mark_by_dropdown],
                outputs=[dataframe_component, cost_plot_component, runtime_plot_component]
            )

    # Return the filter controls so they can be used to update other sections
    return show_incomplete_checkbox, show_open_only_checkbox, mark_by_dropdown

# # --- Detailed Benchmark Display ---
def create_benchmark_details_display(
        full_df: pd.DataFrame,
        tag_map: dict,
        category_name: str,
        validation: bool = False,
):
    """
    Generates a detailed breakdown for each benchmark within a given category.
    For each benchmark, it creates a title, a filtered table, and a scatter plot.
    Only shows the detailed results if there is more than one benchmark in the category.
    Args:
        full_df (pd.DataFrame): The complete, "pretty" dataframe for the entire split.
        tag_map (dict): The "pretty" tag map to find the list of benchmarks.
        category_name (str): The main category to display details for (e.g., "Literature Understanding").
    """
    # 1. Get the list of benchmarks for the selected category
    benchmark_names = tag_map.get(category_name, [])

    # Only show detailed results if there is more than one benchmark
    if len(benchmark_names) <= 1:
        return

    gr.HTML(f'<h2 class="benchmark-main-subtitle">{category_name} Detailed Benchmark Results</h2>')
    gr.Markdown("---")
    # 2. Loop through each benchmark and create its UI components
    for benchmark_name in benchmark_names:
        anchor_id = create_gradio_anchor_id(benchmark_name, validation)
        gr.HTML(
            f"""
                <h3 class="benchmark-title" id="{anchor_id}">{benchmark_name} Leaderboard <a href="#{anchor_id}" class="header-link-icon">🔗</a></h3>
            <div class="benchmark-description">{get_benchmark_description(benchmark_name, validation)}</div>
            <button onclick="scroll_to_element('page-content-wrapper')" class="primary-link-button">Return to the aggregate {category_name} leaderboard</button>
            """
        )

        # 3. Prepare the data for this specific benchmark's table and plot
        benchmark_score_col = f"{benchmark_name} Score"
        benchmark_cost_col = f"{benchmark_name} Cost"
        benchmark_runtime_col = f"{benchmark_name} Runtime"
        benchmark_download_col = f"{benchmark_name} Download"
        benchmark_visualization_col = f"{benchmark_name} Visualization"

        # Define the columns needed for the detailed table
        table_cols = ['SDK Version','Source','Openness', 'Date', benchmark_score_col, benchmark_cost_col, benchmark_runtime_col, 'Logs', benchmark_visualization_col, benchmark_download_col, 'id', 'Language Model']

        # Filter to only columns that actually exist in the full dataframe
        existing_table_cols = [col for col in table_cols if col in full_df.columns]

        if benchmark_score_col not in existing_table_cols:
            gr.Markdown(f"Score data for {benchmark_name} not available.")
            continue # Skip to the next benchmark if score is missing

        # Create a specific DataFrame for the table view
        benchmark_table_df = full_df[existing_table_cols].copy()
        pareto_df = get_pareto_df(benchmark_table_df)
        # Get the list of agents on the frontier. We'll use this list later.
        trophy_uri = get_svg_as_data_uri("assets/trophy.svg")
        if not pareto_df.empty and 'id' in pareto_df.columns:
            pareto_agent_names = pareto_df['id'].tolist()
        else:
            pareto_agent_names = []

        # Clean the Language Model column first
        benchmark_table_df['Language Model'] = benchmark_table_df['Language Model'].apply(clean_llm_base_list)
        
        # Combine icons with Language Model column
        def format_language_model_with_icons(row):
            icons_html = ''
            
            # Add Pareto trophy if on frontier
            if row['id'] in pareto_agent_names:
                icons_html += f'<img src="{trophy_uri}" alt="On Pareto Frontier" title="On Pareto Frontier" style="width:18px; height:18px;">'
            
            # Add openness lock icon
            openness_val = row.get('Openness', '')
            if openness_val in [aliases.CANONICAL_OPENNESS_OPEN, 'Open', 'Open Source', 'Open Source + Open Weights']:
                lock_uri = get_svg_as_data_uri("assets/lock-open.svg")
                icons_html += f'<img src="{lock_uri}" alt="Open" title="Open source model" style="width:16px; height:16px;">'
            else:
                lock_uri = get_svg_as_data_uri("assets/lock-closed.svg")
                icons_html += f'<img src="{lock_uri}" alt="Closed" title="Closed source model" style="width:16px; height:16px;">'
            
            # Add company logo
            company_html = get_company_logo_html(row['Language Model'])
            if company_html:
                icons_html += company_html
            
            # Format the model name
            model_name = row['Language Model']
            if isinstance(model_name, list):
                if len(model_name) > 1:
                    tooltip_text = "\\n".join(map(str, model_name))
                    model_text = f'<span class="tooltip-icon cell-tooltip-icon" style="cursor: help;" data-tooltip="{tooltip_text}">{model_name[0]} (+ {len(model_name) - 1}) ⓘ</span>'
                elif len(model_name) == 1:
                    model_text = model_name[0]
                else:
                    model_text = str(model_name)
            else:
                model_text = str(model_name)
            
            # Wrap in a flex container to keep icons horizontal
            return f'<div style="display:flex; align-items:center; gap:4px; flex-wrap:nowrap;">{icons_html}<span>{model_text}</span></div>'
        
        benchmark_table_df['Language Model'] = benchmark_table_df.apply(format_language_model_with_icons, axis=1)
        
        # append the repro url to the end of the SDK Version
        if 'Source' in benchmark_table_df.columns:
            benchmark_table_df['SDK Version'] = benchmark_table_df.apply(
                lambda row: f"{row['SDK Version']} {row['Source']}" if row['Source'] else row['SDK Version'],
                axis=1
            )

        # Calculated and add "Benchmark Attempted" column
        def check_benchmark_status(row):
            has_score = pd.notna(row.get(benchmark_score_col))
            has_cost = pd.notna(row.get(benchmark_cost_col))
            if has_score and has_cost:
                return "✅"
            if has_score or has_cost:
                return "⚠️"
            return "🚫 "

        # Apply the function to create the new column
        benchmark_table_df['Attempted Benchmark'] = benchmark_table_df.apply(check_benchmark_status, axis=1)
        # Sort the DataFrame
        if benchmark_score_col in benchmark_table_df.columns:
            benchmark_table_df = benchmark_table_df.sort_values(
                by=benchmark_score_col, ascending=False, na_position='last'
            )
        # 1. Format the cost and score columns
        benchmark_table_df = format_cost_column(benchmark_table_df, benchmark_cost_col)
        benchmark_table_df = format_score_column(benchmark_table_df, benchmark_score_col)
        
        # Format download column as clickable icon
        if benchmark_download_col in benchmark_table_df.columns:
            def format_download_link(url):
                if pd.isna(url) or url == "":
                    return ""
                return f"[⬇️]({url})"
            benchmark_table_df[benchmark_download_col] = benchmark_table_df[benchmark_download_col].apply(format_download_link)
        
        # Format visualization column as clickable icon (bar chart emoji)
        if benchmark_visualization_col in benchmark_table_df.columns:
            def format_visualization_link(url):
                if pd.isna(url) or url == "":
                    return ""
                return f"[📊]({url})"
            benchmark_table_df[benchmark_visualization_col] = benchmark_table_df[benchmark_visualization_col].apply(format_visualization_link)
        
        desired_cols_in_order = [
            'Language Model',
            'SDK Version',
            'Attempted Benchmark',
            benchmark_score_col,
            benchmark_cost_col,
            benchmark_runtime_col,
            'Date',
            'Logs',
            benchmark_visualization_col,
            benchmark_download_col
        ]
        for col in desired_cols_in_order:
            if col not in benchmark_table_df.columns:
                benchmark_table_df[col] = pd.NA # Add as an empty column
        benchmark_table_df = benchmark_table_df[desired_cols_in_order]
        
        # Format the runtime column before renaming
        if benchmark_runtime_col in benchmark_table_df.columns:
            benchmark_table_df = format_runtime_column(benchmark_table_df, benchmark_runtime_col)
        
        # Format Date column to show only date (not time)
        if "Date" in benchmark_table_df.columns:
            benchmark_table_df = format_date_column(benchmark_table_df, "Date")
        
        # Rename columns for a cleaner table display, as requested
        benchmark_table_df.rename(columns={
            benchmark_score_col: 'Score',
            benchmark_cost_col: 'Cost',
            benchmark_runtime_col: 'Runtime',
            benchmark_visualization_col: '📊',  # Visualization column header
            benchmark_download_col: '⬇️',  # Download column header
        }, inplace=True)
        
        # Now get headers from the renamed dataframe
        df_headers = benchmark_table_df.columns.tolist()
        df_datatypes = []
        for col in df_headers:
            if col in ["Logs", "📊", "⬇️"] or "Cost" in col or "Score" in col or "Runtime" in col:
                df_datatypes.append("markdown")
            elif col in ["SDK Version", "Language Model"]:
                df_datatypes.append("html")
            else:
                df_datatypes.append("str")
        
        # Cost/Performance plot
        cost_benchmark_plot = _plot_scatter_plotly(
            data=full_df,
            x=benchmark_cost_col,
            y=benchmark_score_col,
            agent_col="SDK Version",
            name=benchmark_name,
            plot_type='cost'
        )
        gr.Plot(value=cost_benchmark_plot, show_label=False) 
        
        # Runtime/Performance plot
        runtime_benchmark_plot = _plot_scatter_plotly(
            data=full_df,
            x=benchmark_runtime_col,
            y=benchmark_score_col,
            agent_col="SDK Version",
            name=benchmark_name,
            plot_type='runtime'
        )
        gr.Plot(value=runtime_benchmark_plot, show_label=False) 
        gr.Markdown(value=SCATTER_DISCLAIMER, elem_id="scatter-disclaimer")

        # Put table and key into an accordion
        with gr.Accordion("Show / Hide Table View", open=True, elem_id="leaderboard-accordion"):
            gr.DataFrame(
                headers=df_headers,
                value=benchmark_table_df,
                datatype=df_datatypes,
                interactive=False,
                wrap=True,
                column_widths=[200, 80, 40, 80, 80, 80, 100, 120, 40, 40],  # Language Model, SDK Version, Attempted, Score, Cost, Runtime, Date, Logs, Visualization, Download
                show_search="search",
                elem_classes=["wrap-header-df"]
            )
            legend_markdown = create_legend_markdown(benchmark_name)
            gr.HTML(value=legend_markdown, elem_id="legend-markdown")

def get_full_leaderboard_data(split: str) -> tuple[pd.DataFrame, dict]:
    """
    Loads and transforms the complete dataset for a given split.
    This function handles caching and returns the final "pretty" DataFrame and tag map.
    """
    viewer_or_data, raw_tag_map = get_leaderboard_viewer_instance(split)

    if isinstance(viewer_or_data, (SimpleLeaderboardViewer, DummyViewer)):
        raw_df, _ = viewer_or_data._load()
        if raw_df.empty:
            return pd.DataFrame(), {}

        pretty_df = transform_raw_dataframe(raw_df)
        pretty_tag_map = create_pretty_tag_map(raw_tag_map, INFORMAL_TO_FORMAL_NAME_MAP)
        if "Logs" in pretty_df.columns:
            # Format Logs column with download icons using raw_df data
            benchmarks = ['swe-bench', 'swe-bench-multimodal', 'commit0', 'swt-bench', 'gaia']
            def make_download_icons(idx):
                parts = []
                for benchmark in benchmarks:
                    download_col = f"{benchmark} download"
                    if download_col in raw_df.columns:
                        url = raw_df.loc[idx, download_col] if idx in raw_df.index else None
                        if pd.notna(url) and url != "":
                            parts.append(f'<a href="{url}" title="Download {benchmark} results" target="_blank">⬇️</a>')
                return " ".join(parts) if parts else ""
            pretty_df["Logs"] = pretty_df.index.map(make_download_icons)
        
        if "Visualization" in pretty_df.columns:
            # Format Visualization column with bar chart icons using raw_df data
            benchmarks = ['swe-bench', 'swe-bench-multimodal', 'commit0', 'swt-bench', 'gaia']
            def make_visualization_icons(idx):
                parts = []
                for benchmark in benchmarks:
                    viz_col = f"{benchmark} visualization"
                    if viz_col in raw_df.columns:
                        url = raw_df.loc[idx, viz_col] if idx in raw_df.index else None
                        if pd.notna(url) and url != "":
                            parts.append(f'<a href="{url}" title="Visualize {benchmark} results" target="_blank">📊</a>')
                return " ".join(parts) if parts else ""
            pretty_df["Visualization"] = pretty_df.index.map(make_visualization_icons)

        if "Source" in pretty_df.columns:
            def format_source_url_to_html(raw_url):
                # Handle empty or NaN values, returning a blank string.
                if pd.isna(raw_url) or raw_url == "": return ""
                # Assume 'source_url' is already a valid web URL and doesn't need conversion.
                return hyperlink(str(raw_url), "🔗")
            # Apply the function to the "source_url" column.
            pretty_df["Source"] = pretty_df["Source"].apply(format_source_url_to_html)
        return pretty_df, pretty_tag_map

    # Fallback for unexpected types
    return pd.DataFrame(), {}
def create_sub_navigation_bar(tag_map: dict, category_name: str, validation: bool = False) -> gr.HTML:
    """
    Builds the entire sub-navigation bar as a single, self-contained HTML component.
    This bypasses Gradio's layout components, giving us full control.
    """
    benchmark_names = tag_map.get(category_name, [])
    if not benchmark_names:
        # Return an empty HTML component to prevent errors
        return gr.HTML()

    # Start building the list of HTML button elements as strings
    html_buttons = []
    for name in benchmark_names:
        target_id = create_gradio_anchor_id(name, validation)

        # Create a standard HTML button.
        # The onclick attribute calls our global JS function directly.
        # Note the mix of double and single quotes.
        button_str = f"""
            <button
                class="primary-link-button"
                onclick="scroll_to_element('{target_id}')"
            >
                {name}
            </button>
        """
        html_buttons.append(button_str)

    # Join the button strings and wrap them in a single div container
    # This container will be our flexbox row.
    full_html = f"""
        <div class="sub-nav-bar-container">
            <span class="sub-nav-label">Benchmarks in this category:</span>
            {' | '.join(html_buttons)}
        </div>
    """

    # Return the entire navigation bar as one single Gradio HTML component
    return gr.HTML(full_html)
