"""
REST API for OpenHands Index leaderboard data.

This module provides API endpoints that use the same data loading functions
as the Gradio UI, ensuring consistency between the web interface and API responses.
"""

import logging
import math
from datetime import datetime
from typing import Optional, Any

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from simple_data_loader import SimpleLeaderboardViewer
from config import CONFIG_NAME, EXTRACTED_DATA_DIR
from setup_data import _last_fetch_time, CACHE_TTL_SECONDS
import os


def _sanitize_value(val: Any) -> Any:
    """Convert NaN/inf values to None for JSON serialization."""
    if val is None:
        return None
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
    return val


def _sanitize_dict(d: dict) -> dict:
    """Recursively sanitize a dictionary for JSON serialization."""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _sanitize_dict(v)
        elif isinstance(v, list):
            result[k] = [_sanitize_dict(i) if isinstance(i, dict) else _sanitize_value(i) for i in v]
        else:
            result[k] = _sanitize_value(v)
    return result

logger = logging.getLogger(__name__)

# Create FastAPI app for API endpoints
api_app = FastAPI(
    title="OpenHands Index API",
    description="""
REST API for accessing OpenHands Index benchmark results.

The OpenHands Index is a comprehensive benchmark for evaluating AI coding agents 
across real-world software engineering tasks. It assesses models across five categories:

- **Issue Resolution**: Fixing bugs (SWE-Bench)
- **Greenfield**: Building new applications (Commit0)
- **Frontend**: UI development (SWE-Bench Multimodal)
- **Testing**: Test generation (SWT-Bench)
- **Information Gathering**: Research tasks (GAIA)

This API provides the same data that powers the leaderboard UI.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Benchmark to category mappings (same as simple_data_loader.py)
BENCHMARK_TO_CATEGORIES = {
    'swe-bench': ['Issue Resolution'],
    'swe-bench-multimodal': ['Frontend'],
    'commit0': ['Greenfield'],
    'swt-bench': ['Testing'],
    'gaia': ['Information Gathering'],
}

ALL_CATEGORIES = ['Issue Resolution', 'Frontend', 'Greenfield', 'Testing', 'Information Gathering']

CATEGORY_DESCRIPTIONS = {
    "Issue Resolution": "Fixing bugs in real GitHub issues (SWE-Bench)",
    "Greenfield": "Building new applications from scratch (Commit0)",
    "Frontend": "UI development with visual context (SWE-Bench Multimodal)",
    "Testing": "Test generation and quality (SWT-Bench)",
    "Information Gathering": "Research and information retrieval (GAIA)",
}

# Openness mapping (same as aliases.py)
OPENNESS_MAPPING = {
    'open': 'open',
    'open_weights': 'open',
    'open_weights_open_data': 'open',
    'closed': 'closed',
    'closed_api_available': 'closed',
    'closed_api_unavailable': 'closed',
}


def _get_leaderboard_data() -> dict:
    """
    Load leaderboard data using the same SimpleLeaderboardViewer used by the UI.
    This ensures API responses match what's displayed in the Gradio interface.
    """
    try:
        data_dir = EXTRACTED_DATA_DIR if os.path.exists(EXTRACTED_DATA_DIR) else "mock_results"
        viewer = SimpleLeaderboardViewer(
            data_dir=data_dir,
            config=CONFIG_NAME,
            split="test"
        )
        
        raw_df, tag_map = viewer._load()
        
        if raw_df is None or raw_df.empty or "Message" in raw_df.columns:
            return {"entries": [], "error": "No data available"}
        
        entries = []
        for _, row in raw_df.iterrows():
            # Normalize openness
            raw_openness = row.get('openness', 'unknown')
            normalized_openness = OPENNESS_MAPPING.get(raw_openness, raw_openness)
            
            entry = {
                "id": row.get('id'),
                "language_model": row.get('Language model'),
                "sdk_version": row.get('SDK version'),
                "openness": normalized_openness,
                "average_score": row.get('average score'),
                "average_cost": row.get('average cost'),
                "average_runtime": row.get('average runtime'),
                "categories_completed": row.get('categories_completed', 0),
                "release_date": row.get('release_date'),
                "benchmarks": {},
                "categories": {},
            }
            
            # Add benchmark-level data
            for benchmark in BENCHMARK_TO_CATEGORIES.keys():
                score_col = f'{benchmark} score'
                cost_col = f'{benchmark} cost'
                runtime_col = f'{benchmark} runtime'
                download_col = f'{benchmark} download'
                viz_col = f'{benchmark} visualization'
                
                if score_col in row and row[score_col] is not None:
                    entry["benchmarks"][benchmark] = {
                        "score": row.get(score_col),
                        "cost": row.get(cost_col),
                        "runtime": row.get(runtime_col),
                        "download_url": row.get(download_col),
                        "visualization_url": row.get(viz_col),
                    }
            
            # Add category-level data
            for category in ALL_CATEGORIES:
                score_col = f'{category} score'
                cost_col = f'{category} cost'
                runtime_col = f'{category} runtime'
                
                if score_col in row and row[score_col] is not None:
                    entry["categories"][category] = {
                        "score": row.get(score_col),
                        "cost": row.get(cost_col),
                        "runtime": row.get(runtime_col),
                    }
            
            # Sanitize the entry to handle NaN values
            entries.append(_sanitize_dict(entry))
        
        # Sort by average score descending
        entries.sort(key=lambda x: x.get('average_score') or 0, reverse=True)
        
        return {
            "entries": entries,
            "total_count": len(entries),
            "fetched_at": _last_fetch_time.isoformat() if _last_fetch_time else None,
        }
        
    except Exception as e:
        logger.error(f"Error loading leaderboard data: {e}")
        return {"entries": [], "error": str(e)}


@api_app.get("/", tags=["Info"])
async def api_root():
    """API information and available endpoints."""
    return {
        "name": "OpenHands Index API",
        "version": "1.0.0",
        "description": "REST API for accessing OpenHands Index benchmark results",
        "leaderboard_ui": "/",
        "documentation": "/api/docs",
        "endpoints": {
            "/api/": "API information (this page)",
            "/api/health": "Health check endpoint",
            "/api/leaderboard": "Get the full leaderboard with scores and metadata",
            "/api/leaderboard/models": "List all language models in the leaderboard",
            "/api/leaderboard/model/{model_name}": "Get data for a specific model",
            "/api/categories": "List all benchmark categories",
            "/api/benchmarks": "List all benchmarks",
            "/api/docs": "Interactive Swagger UI documentation",
        }
    }


@api_app.get("/health", tags=["Health"])
async def health_check():
    """Check API health status and cache information."""
    cache_age = None
    if _last_fetch_time is not None:
        cache_age = (datetime.now() - _last_fetch_time).total_seconds()
    
    return {
        "status": "healthy",
        "version": "1.0.0",
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "cache_age_seconds": cache_age,
        "last_fetch_time": _last_fetch_time.isoformat() if _last_fetch_time else None,
    }


@api_app.get("/leaderboard", tags=["Leaderboard"])
async def get_leaderboard(
    openness: Optional[str] = Query(None, description="Filter by openness (open/closed)"),
    min_categories: Optional[int] = Query(None, description="Minimum categories completed"),
    sort_by: str = Query("average_score", description="Sort field (average_score, average_cost, average_runtime)"),
    limit: Optional[int] = Query(None, description="Limit number of results"),
):
    """
    Get the full leaderboard with benchmark scores and metadata.
    
    Returns the same data displayed in the OpenHands Index UI leaderboard.
    """
    data = _get_leaderboard_data()
    
    if "error" in data and data.get("entries") == []:
        raise HTTPException(status_code=503, detail=data["error"])
    
    entries = data.get("entries", [])
    
    # Apply filters
    if openness:
        entries = [e for e in entries if e.get("openness") == openness]
    
    if min_categories is not None:
        entries = [e for e in entries if (e.get("categories_completed") or 0) >= min_categories]
    
    # Apply sorting
    reverse = True
    if sort_by in ["average_cost", "average_runtime"]:
        reverse = False  # Lower is better
    
    entries.sort(
        key=lambda x: x.get(sort_by) if x.get(sort_by) is not None else (float('inf') if not reverse else float('-inf')),
        reverse=reverse
    )
    
    # Apply limit
    if limit:
        entries = entries[:limit]
    
    return {
        "entries": entries,
        "total_count": len(entries),
        "categories": ALL_CATEGORIES,
        "benchmarks": list(BENCHMARK_TO_CATEGORIES.keys()),
        "fetched_at": data.get("fetched_at"),
    }


@api_app.get("/leaderboard/models", tags=["Leaderboard"])
async def list_models(
    openness: Optional[str] = Query(None, description="Filter by openness (open/closed)"),
):
    """List all language models available in the leaderboard."""
    data = _get_leaderboard_data()
    entries = data.get("entries", [])
    
    if openness:
        entries = [e for e in entries if e.get("openness") == openness]
    
    models = [
        {
            "language_model": e.get("language_model"),
            "sdk_version": e.get("sdk_version"),
            "openness": e.get("openness"),
            "average_score": e.get("average_score"),
            "categories_completed": e.get("categories_completed"),
        }
        for e in entries
    ]
    
    return {
        "models": models,
        "total_count": len(models),
    }


@api_app.get("/leaderboard/model/{model_name}", tags=["Leaderboard"])
async def get_model(model_name: str):
    """Get detailed data for a specific language model."""
    data = _get_leaderboard_data()
    entries = data.get("entries", [])
    
    # Find entries matching the model name (case-insensitive)
    matching = [e for e in entries if (e.get("language_model") or "").lower() == model_name.lower()]
    
    if not matching:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    return {
        "model_name": model_name,
        "entries": matching,
        "count": len(matching),
    }


@api_app.get("/categories", tags=["Metadata"])
async def list_categories():
    """List all benchmark categories with their associated benchmarks."""
    category_to_benchmarks = {}
    for benchmark, categories in BENCHMARK_TO_CATEGORIES.items():
        for category in categories:
            if category not in category_to_benchmarks:
                category_to_benchmarks[category] = []
            category_to_benchmarks[category].append(benchmark)
    
    return {
        "categories": [
            {
                "name": category,
                "description": CATEGORY_DESCRIPTIONS.get(category, ""),
                "benchmarks": category_to_benchmarks.get(category, [])
            }
            for category in ALL_CATEGORIES
        ]
    }


@api_app.get("/benchmarks", tags=["Metadata"])
async def list_benchmarks():
    """List all benchmarks with their category mappings."""
    return {
        "benchmarks": [
            {
                "name": benchmark,
                "categories": categories
            }
            for benchmark, categories in BENCHMARK_TO_CATEGORIES.items()
        ]
    }
