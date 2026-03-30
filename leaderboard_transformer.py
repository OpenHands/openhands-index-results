import plotly.graph_objects as go
import numpy as np
import pandas as pd
import logging
from typing import Optional
import base64
import html
import os

import aliases
from constants import FONT_FAMILY, FONT_FAMILY_SHORT

logger = logging.getLogger(__name__)

# Company logo mapping for graphs - maps model name patterns to company logo files
COMPANY_LOGO_MAP = {
    "anthropic": {"path": "assets/logo-anthropic.svg", "name": "Anthropic"},
    "claude": {"path": "assets/logo-anthropic.svg", "name": "Anthropic"},
    "openai": {"path": "assets/logo-openai.svg", "name": "OpenAI"},
    "gpt": {"path": "assets/logo-openai.svg", "name": "OpenAI"},
    "o1": {"path": "assets/logo-openai.svg", "name": "OpenAI"},
    "o3": {"path": "assets/logo-openai.svg", "name": "OpenAI"},
    "google": {"path": "assets/logo-google.svg", "name": "Google"},
    "gemini": {"path": "assets/logo-google.svg", "name": "Google"},
    "gemma": {"path": "assets/logo-google.svg", "name": "Google"},
    "meta": {"path": "assets/logo-meta.svg", "name": "Meta"},
    "llama": {"path": "assets/logo-meta.svg", "name": "Meta"},
    "mistral": {"path": "assets/logo-mistral.svg", "name": "Mistral"},
    "mixtral": {"path": "assets/logo-mistral.svg", "name": "Mistral"},
    "codestral": {"path": "assets/logo-mistral.svg", "name": "Mistral"},
    "deepseek": {"path": "assets/logo-deepseek.svg", "name": "DeepSeek"},
    "xai": {"path": "assets/logo-xai.svg", "name": "xAI"},
    "grok": {"path": "assets/logo-xai.svg", "name": "xAI"},
    "cohere": {"path": "assets/logo-cohere.svg", "name": "Cohere"},
    "command": {"path": "assets/logo-cohere.svg", "name": "Cohere"},
    "qwen": {"path": "assets/logo-qwen.svg", "name": "Qwen"},
    "alibaba": {"path": "assets/logo-qwen.svg", "name": "Qwen"},
    "kimi": {"path": "assets/logo-moonshot.svg", "name": "Moonshot"},
    "moonshot": {"path": "assets/logo-moonshot.svg", "name": "Moonshot"},
    "minimax": {"path": "assets/logo-minimax.svg", "name": "MiniMax"},
    "nvidia": {"path": "assets/logo-nvidia.svg", "name": "NVIDIA"},
    "nemotron": {"path": "assets/logo-nvidia.svg", "name": "NVIDIA"},
    "glm": {"path": "assets/logo-zai.svg", "name": "z.ai"},
    "z.ai": {"path": "assets/logo-zai.svg", "name": "z.ai"},
    "zai": {"path": "assets/logo-zai.svg", "name": "z.ai"},
}

# Openness icon mapping
OPENNESS_ICON_MAP = {
    "open": {"path": "assets/lock-open.svg", "name": "Open"},
    "closed": {"path": "assets/lock-closed.svg", "name": "Closed"},
}

# Country flag mapping - maps model name patterns to country flags
COUNTRY_FLAG_MAP = {
    "us": {"path": "assets/flag-us.svg", "name": "United States"},
    "cn": {"path": "assets/flag-cn.svg", "name": "China"},
    "fr": {"path": "assets/flag-fr.svg", "name": "France"},
}

# Model to country mapping (based on company headquarters)
MODEL_COUNTRY_MAP = {
    # US companies
    "anthropic": "us", "claude": "us",
    "openai": "us", "gpt": "us", "o1": "us", "o3": "us",
    "google": "us", "gemini": "us", "gemma": "us",
    "meta": "us", "llama": "us",
    "xai": "us", "grok": "us",
    "cohere": "us", "command": "us",
    "nvidia": "us", "nemotron": "us",
    # China companies
    "deepseek": "cn",
    "qwen": "cn", "alibaba": "cn",
    "kimi": "cn", "moonshot": "cn",
    "minimax": "cn",
    # France companies
    "mistral": "fr", "mixtral": "fr", "codestral": "fr",
}

# OpenHands branding constants
OPENHANDS_LOGO_PATH_LIGHT = "assets/openhands_logo_color_forwhite.png"
OPENHANDS_LOGO_PATH_DARK = "assets/openhands_logo_color_forblack.png"
OPENHANDS_URL = "https://index.openhands.dev"

# URL annotation for bottom right of charts
URL_ANNOTATION = dict(
    text=OPENHANDS_URL,
    xref="paper",
    yref="paper",
    x=1,
    y=-0.15,
    xanchor="right",
    yanchor="bottom",
    showarrow=False,
    font=dict(
        family=FONT_FAMILY,
        size=14,
        color="#82889B",  # neutral-400
    ),
)


def get_openhands_logo_images():
    """Get both light and dark mode OpenHands logos as Plotly image dicts.
    
    Returns two images - one for light mode (forwhite) and one for dark mode (forblack).
    CSS is used to show/hide the appropriate logo based on the current mode.
    """
    images = []
    
    # Light mode logo (visible in light mode, hidden in dark mode)
    if os.path.exists(OPENHANDS_LOGO_PATH_LIGHT):
        try:
            with open(OPENHANDS_LOGO_PATH_LIGHT, "rb") as f:
                logo_data = base64.b64encode(f.read()).decode('utf-8')
            images.append(dict(
                source=f"data:image/png;openhands=lightlogo;base64,{logo_data}",
                xref="paper",
                yref="paper",
                x=0,
                y=-0.15,
                sizex=0.15,
                sizey=0.15,
                xanchor="left",
                yanchor="bottom",
            ))
        except Exception:
            pass
    
    # Dark mode logo (hidden in light mode, visible in dark mode)
    if os.path.exists(OPENHANDS_LOGO_PATH_DARK):
        try:
            with open(OPENHANDS_LOGO_PATH_DARK, "rb") as f:
                logo_data = base64.b64encode(f.read()).decode('utf-8')
            images.append(dict(
                source=f"data:image/png;openhands=darklogo;base64,{logo_data}",
                xref="paper",
                yref="paper",
                x=0,
                y=-0.15,
                sizex=0.15,
                sizey=0.15,
                xanchor="left",
                yanchor="bottom",
            ))
        except Exception:
            pass
    
    return images


def add_branding_to_figure(fig: go.Figure) -> go.Figure:
    """Add OpenHands logo and URL to a Plotly figure."""
    # Add both light and dark mode logo images
    logo_images = get_openhands_logo_images()
    if logo_images:
        existing_images = list(fig.layout.images) if fig.layout.images else []
        existing_images.extend(logo_images)
        fig.update_layout(images=existing_images)
    
    # Add URL annotation
    existing_annotations = list(fig.layout.annotations) if fig.layout.annotations else []
    existing_annotations.append(URL_ANNOTATION)
    fig.update_layout(annotations=existing_annotations)
    
    return fig


def get_company_from_model(model_name: str) -> dict:
    """
    Gets the company info (logo path and name) from a model name.
    Returns default unknown logo if no match found.
    """
    if not model_name:
        return {"path": "assets/logo-unknown.svg", "name": "Unknown"}

    # Handle list of models - use the first one
    if isinstance(model_name, list):
        model_name = model_name[0] if model_name else ""

    model_lower = str(model_name).lower()

    # Check each pattern
    for pattern, company_info in COMPANY_LOGO_MAP.items():
        if pattern in model_lower:
            return company_info

    return {"path": "assets/logo-unknown.svg", "name": "Unknown"}


def get_openness_icon(openness: str) -> dict:
    """
    Gets the openness icon info (path and name) from openness value.
    Returns closed icon as default.
    """
    if not openness:
        return OPENNESS_ICON_MAP["closed"]
    
    openness_lower = str(openness).lower()
    if openness_lower in OPENNESS_ICON_MAP:
        return OPENNESS_ICON_MAP[openness_lower]
    
    return OPENNESS_ICON_MAP["closed"]


def get_country_from_model(model_name: str) -> dict:
    """
    Gets the country flag info (path and name) from a model name.
    Returns US flag as default.
    """
    if not model_name:
        return COUNTRY_FLAG_MAP["us"]

    # Handle list of models - use the first one
    if isinstance(model_name, list):
        model_name = model_name[0] if model_name else ""

    model_lower = str(model_name).lower()

    # Check each pattern
    for pattern, country_code in MODEL_COUNTRY_MAP.items():
        if pattern in model_lower:
            return COUNTRY_FLAG_MAP.get(country_code, COUNTRY_FLAG_MAP["us"])

    return COUNTRY_FLAG_MAP["us"]


def get_marker_icon(model_name: str, openness: str, mark_by: str) -> dict:
    """
    Gets the appropriate icon based on the mark_by selection.
    
    Args:
        model_name: The model name
        openness: The openness value (open/closed)
        mark_by: One of "Company", "Openness", or "Country"
    
    Returns:
        dict with 'path' and 'name' keys
    """
    from constants import MARK_BY_COMPANY, MARK_BY_OPENNESS, MARK_BY_COUNTRY
    
    if mark_by == MARK_BY_OPENNESS:
        return get_openness_icon(openness)
    elif mark_by == MARK_BY_COUNTRY:
        return get_country_from_model(model_name)
    else:  # Default to company
        return get_company_from_model(model_name)


# Standard layout configuration for all charts
STANDARD_LAYOUT = dict(
    template="plotly_white",
    height=572,
    font=dict(
        family=FONT_FAMILY,
        color="#0D0D0F",  # neutral-950
    ),
    hoverlabel=dict(
        bgcolor="#222328",  # neutral-800
        font_size=12,
        font_family=FONT_FAMILY_SHORT,
        font_color="#F7F8FB",  # neutral-50
    ),
    legend=dict(
        bgcolor='#F7F8FB',  # neutral-50
    ),
    margin=dict(b=80),  # Extra margin for logo and URL
)

# Standard font for annotations
STANDARD_FONT = dict(
    size=10,
    color='#0D0D0F',  # neutral-950
    family=FONT_FAMILY_SHORT
)


def create_scatter_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str = "Average Score",
    mark_by: str = None,
    x_type: str = "log",  # "log" or "date"
    pareto_lower_is_better: bool = True,  # For x-axis: True means lower x is better
    model_col: str = None,
    openness_col: str = None,
) -> go.Figure:
    """
    Generic scatter chart with Pareto frontier, marker icons, and consistent styling.
    
    This is the single source of truth for all scatter plots in the application.
    
    Args:
        df: DataFrame with the data to plot
        x_col: Column name for x-axis values
        y_col: Column name for y-axis values (typically score)
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label (default: "Average Score")
        mark_by: One of "Company", "Openness", or "Country" for marker icons
        x_type: "log" for logarithmic scale, "date" for datetime scale
        pareto_lower_is_better: If True, lower x values are better (cost, size);
                                If False, higher x values are better (time evolution)
        model_col: Column name for model names (auto-detected if None)
        openness_col: Column name for openness values (auto-detected if None)
    
    Returns:
        Plotly figure with scatter plot, Pareto frontier, and branding
    """
    from constants import MARK_BY_DEFAULT
    
    if mark_by is None:
        mark_by = MARK_BY_DEFAULT
    
    # Auto-detect column names if not provided
    if model_col is None:
        for col in ['Language Model', 'Language model', 'llm_base']:
            if col in df.columns:
                model_col = col
                break
        if model_col is None:
            model_col = 'Language Model'
    
    if openness_col is None:
        openness_col = 'Openness' if 'Openness' in df.columns else 'openness'
    
    # Prepare data
    plot_df = df.copy()
    
    # Ensure required columns exist
    if x_col not in plot_df.columns or y_col not in plot_df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="Required data columns not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=STANDARD_FONT
        )
        fig.update_layout(**STANDARD_LAYOUT, title=title)
        return fig
    
    # Convert to appropriate types
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors='coerce')
    if x_type == "date":
        plot_df[x_col] = pd.to_datetime(plot_df[x_col], errors='coerce')
    else:
        plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors='coerce')
    
    # Drop rows with missing values
    plot_df = plot_df.dropna(subset=[x_col, y_col])
    
    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No valid data points available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=STANDARD_FONT
        )
        fig.update_layout(**STANDARD_LAYOUT, title=title)
        return fig
    
    fig = go.Figure()
    
    # Calculate axis ranges
    x_values = plot_df[x_col].tolist()
    y_values = plot_df[y_col].tolist()
    
    if x_type == "log":
        min_x = min(x_values)
        max_x = max(x_values)
        x_range_log = [np.log10(min_x * 0.5) if min_x > 0 else -2,
                       np.log10(max_x * 1.5) if max_x > 0 else 2]
    else:
        min_x = min(x_values)
        max_x = max(x_values)
        if x_type == "date":
            x_padding = (max_x - min_x) * 0.1 if max_x != min_x else pd.Timedelta(days=15)
            x_range = [min_x - x_padding, max_x + x_padding]
        else:
            x_range = None
    
    min_y = min(y_values)
    max_y = max(y_values)
    y_range = [min_y - 5 if min_y > 5 else 0, max_y + 5]
    
    # Calculate Pareto frontier
    frontier_rows = []
    if pareto_lower_is_better:
        # Lower x is better (cost, params): sort by x ascending, track max y
        sorted_df = plot_df.sort_values(by=[x_col, y_col], ascending=[True, False])
        max_score = float('-inf')
        for _, row in sorted_df.iterrows():
            if row[y_col] >= max_score:
                frontier_rows.append(row)
                max_score = row[y_col]
    else:
        # Higher x is better (time): sort by x ascending, track max y seen so far
        sorted_df = plot_df.sort_values(by=x_col, ascending=True)
        max_score = float('-inf')
        for _, row in sorted_df.iterrows():
            if row[y_col] > max_score:
                frontier_rows.append(row)
                max_score = row[y_col]
    
    # Draw Pareto frontier line
    if frontier_rows:
        frontier_x = [row[x_col] for row in frontier_rows]
        frontier_y = [row[y_col] for row in frontier_rows]
        fig.add_trace(go.Scatter(
            x=frontier_x,
            y=frontier_y,
            mode='lines',
            name='Pareto Frontier',
            showlegend=False,
            line=dict(color='#FFE165', width=2, dash='dash'),
            hoverinfo='skip'
        ))
    
    # Prepare hover text for all points
    hover_texts = []
    for _, row in plot_df.iterrows():
        model_name = row.get(model_col, 'Unknown')
        if isinstance(model_name, list):
            model_name = model_name[0] if model_name else 'Unknown'
        model_name = str(model_name).split('/')[-1]
        
        h_pad = "   "
        hover_text = f"<br>{h_pad}<b>{model_name}</b>{h_pad}<br>"
        hover_text += f"{h_pad}{x_label}: <b>{row[x_col]}</b>{h_pad}<br>"
        hover_text += f"{h_pad}{y_label}: <b>{row[y_col]:.1f}</b>{h_pad}<br>"
        hover_texts.append(hover_text)
    
    # Add invisible scatter trace for hover detection
    fig.add_trace(go.Scatter(
        x=plot_df[x_col],
        y=plot_df[y_col],
        mode='markers',
        name='Models',
        showlegend=False,
        text=hover_texts,
        hoverinfo='text',
        marker=dict(color='rgba(0,0,0,0)', size=25, opacity=0)
    ))
    
    # Add marker icon images
    layout_images = []
    
    for _, row in plot_df.iterrows():
        x_val = row[x_col]
        y_val = row[y_col]
        model_name = row.get(model_col, '')
        openness = row.get(openness_col, '')
        
        marker_info = get_marker_icon(model_name, openness, mark_by)
        logo_path = marker_info['path']
        
        if os.path.exists(logo_path):
            try:
                with open(logo_path, 'rb') as f:
                    encoded_logo = base64.b64encode(f.read()).decode('utf-8')
                logo_uri = f"data:image/svg+xml;base64,{encoded_logo}"
                
                if x_type == "date":
                    # For date axes, use data coordinates directly
                    layout_images.append(dict(
                        source=logo_uri,
                        xref="x",
                        yref="y",
                        x=x_val,
                        y=y_val,
                        sizex=15 * 24 * 60 * 60 * 1000,  # ~15 days in milliseconds
                        sizey=3,  # score units
                        xanchor="center",
                        yanchor="middle",
                        layer="above"
                    ))
                else:
                    # For log axes, use domain coordinates (0-1 range)
                    if x_type == "log" and x_val > 0:
                        log_x = np.log10(x_val)
                        domain_x = (log_x - x_range_log[0]) / (x_range_log[1] - x_range_log[0])
                    else:
                        domain_x = 0.5
                    
                    domain_y = (y_val - y_range[0]) / (y_range[1] - y_range[0]) if (y_range[1] - y_range[0]) > 0 else 0.5
                    
                    # Clamp to valid range
                    domain_x = max(0, min(1, domain_x))
                    domain_y = max(0, min(1, domain_y))
                    
                    layout_images.append(dict(
                        source=logo_uri,
                        xref="x domain",
                        yref="y domain",
                        x=domain_x,
                        y=domain_y,
                        sizex=0.04,
                        sizey=0.06,
                        xanchor="center",
                        yanchor="middle",
                        layer="above"
                    ))
            except Exception:
                pass
    
    # Add labels for frontier points only
    for row in frontier_rows:
        model_name = row.get(model_col, '')
        if isinstance(model_name, list):
            model_name = model_name[0] if model_name else ''
        model_name = str(model_name).split('/')[-1]
        if len(model_name) > 25:
            model_name = model_name[:22] + '...'
        
        x_val = row[x_col]
        y_val = row[y_col]
        
        # For log scale, annotation x needs to be in log space
        if x_type == "log":
            ann_x = np.log10(x_val) if x_val > 0 else 0
        else:
            ann_x = x_val
        
        fig.add_annotation(
            x=ann_x,
            y=y_val,
            text=model_name,
            showarrow=False,
            yshift=20,
            font=STANDARD_FONT,
            xanchor='center',
            yanchor='bottom'
        )
    
    # Configure layout
    xaxis_config = dict(title=x_label)
    if x_type == "log":
        xaxis_config['type'] = 'log'
        xaxis_config['range'] = x_range_log
    elif x_type == "date":
        xaxis_config['range'] = x_range
    
    layout_config = dict(
        **STANDARD_LAYOUT,
        title=title,
        xaxis=xaxis_config,
        yaxis=dict(title=y_label, range=y_range),
    )
    
    if layout_images:
        layout_config['images'] = layout_images
    
    fig.update_layout(**layout_config)
    
    # Add branding
    add_branding_to_figure(fig)
    
    return fig


INFORMAL_TO_FORMAL_NAME_MAP = {
    # Short Names
    "lit": "Literature Understanding",
    "code": "Code & Execution",
    "data": "Data Analysis",
    "discovery": "End-to-End Discovery",

    # Validation Names
    "arxivdigestables_validation": "ArxivDIGESTables-Clean",
    "ArxivDIGESTables_Clean_validation": "ArxivDIGESTables-Clean",
    "sqa_dev": "ScholarQA-CS2",
    "ScholarQA_CS2_validation": "ScholarQA-CS2",
    "litqa2_validation": "LitQA2-FullText",
    "LitQA2_FullText_validation": "LitQA2-FullText",
    "paper_finder_validation": "PaperFindingBench",
    "PaperFindingBench_validation": "PaperFindingBench",
    "paper_finder_litqa2_validation": "LitQA2-FullText-Search",
    "LitQA2_FullText_Search_validation": "LitQA2-FullText-Search",
    "discoverybench_validation": "DiscoveryBench",
    "DiscoveryBench_validation": "DiscoveryBench",
    "core_bench_validation": "CORE-Bench-Hard",
    "CORE_Bench_Hard_validation": "CORE-Bench-Hard",
    "ds1000_validation": "DS-1000",
    "DS_1000_validation": "DS-1000",
    "e2e_discovery_validation": "E2E-Bench",
    "E2E_Bench_validation": "E2E-Bench",
    "e2e_discovery_hard_validation": "E2E-Bench-Hard",
    "E2E_Bench_Hard_validation": "E2E-Bench-Hard",
    "super_validation": "SUPER-Expert",
    "SUPER_Expert_validation": "SUPER-Expert",
    # Test Names
    "paper_finder_test": "PaperFindingBench",
    "PaperFindingBench_test": "PaperFindingBench",
    "paper_finder_litqa2_test": "LitQA2-FullText-Search",
    "LitQA2_FullText_Search_test": "LitQA2-FullText-Search",
    "sqa_test": "ScholarQA-CS2",
    "ScholarQA_CS2_test": "ScholarQA-CS2",
    "arxivdigestables_test": "ArxivDIGESTables-Clean",
    "ArxivDIGESTables_Clean_test": "ArxivDIGESTables-Clean",
    "litqa2_test": "LitQA2-FullText",
    "LitQA2_FullText_test": "LitQA2-FullText",
    "discoverybench_test": "DiscoveryBench",
    "DiscoveryBench_test": "DiscoveryBench",
    "core_bench_test": "CORE-Bench-Hard",
    "CORE_Bench_Hard_test": "CORE-Bench-Hard",
    "ds1000_test": "DS-1000",
    "DS_1000_test": "DS-1000",
    "e2e_discovery_test": "E2E-Bench",
    "E2E_Bench_test": "E2E-Bench",
    "e2e_discovery_hard_test": "E2E-Bench-Hard",
    "E2E_Bench_Hard_test": "E2E-Bench-Hard",
    "super_test": "SUPER-Expert",
    "SUPER_Expert_test": "SUPER-Expert",
}
ORDER_MAP = {
    'Overall_keys': [
        'lit',
        'code',
        'data',
        'discovery',
    ],
    'Literature Understanding': [
        'PaperFindingBench',
        'LitQA2-FullText-Search',
        'ScholarQA-CS2',
        'LitQA2-FullText',
        'ArxivDIGESTables-Clean'
    ],
    'Code & Execution': [
        'SUPER-Expert',
        'CORE-Bench-Hard',
        'DS-1000'
    ],
    # Add other keys for 'Data Analysis' and 'Discovery' when/if we add more benchmarks in those categories
}


def _safe_round(value, digits=3):
    """Rounds a number if it's a valid float/int, otherwise returns it as is."""
    return round(value, digits) if isinstance(value, (float, int)) and pd.notna(value) else value


def _pretty_column_name(raw_col: str) -> str:
    """
    Takes a raw column name from the DataFrame and returns a "pretty" version.
    Handles three cases:
    1. Fixed names (e.g., 'SDK version' -> 'SDK Version', 'Language model' -> 'Language Model').
    2. Dynamic names (e.g., 'swe_bench_lite score' -> 'SWE-bench Lite Score').
    3. Fallback for any other names.
    """
    # Case 1: Handle fixed, special-case mappings first.
    fixed_mappings = {
        'id': 'id',
        'SDK version': 'SDK Version',
        'Openhands version': 'SDK Version',  # Legacy support
        'Language model': 'Language Model',
        'Agent description': 'Agent Description',
        'Submission date': 'Date',
        'average score': 'Average Score',
        'Overall': 'Average Score',  # Legacy support
        'average cost': 'Average Cost',
        'total cost': 'Average Cost',  # Legacy support
        'Overall cost': 'Average Cost',  # Legacy support
        'average runtime': 'Average Runtime',
        'categories_completed': 'Categories Completed',
        'Logs': 'Logs',
        'Openness': 'Openness',
        'LLM base': 'Model',
        'Source': 'Source',
    }

    if raw_col in fixed_mappings:
        return fixed_mappings[raw_col]

    # Case 2: Handle dynamic names by finding the longest matching base name.
    # We sort by length (desc) to match 'core_bench_validation' before 'core_bench'.
    sorted_base_names = sorted(INFORMAL_TO_FORMAL_NAME_MAP.keys(), key=len, reverse=True)

    for base_name in sorted_base_names:
        if raw_col.startswith(base_name):
            formal_name = INFORMAL_TO_FORMAL_NAME_MAP[base_name]

            # Get the metric part (e.g., ' score' or ' cost 95% CI')
            metric_part = raw_col[len(base_name):].strip()

            # Capitalize the metric part correctly (e.g., 'score' -> 'Score')
            pretty_metric = metric_part.capitalize()
            return f"{formal_name} {pretty_metric}"

    # Case 3: If no specific rule applies, just make it title case.
    return raw_col.title()


def create_pretty_tag_map(raw_tag_map: dict, name_map: dict) -> dict:
    """
    Converts a tag map with raw names into a tag map with pretty, formal names,
    applying a specific, non-alphabetic sort order to the values.
    """
    pretty_map = {}
    # Helper to get pretty name with a fallback
    def get_pretty(raw_name):
        result = name_map.get(raw_name, raw_name.replace("_", " "))
        # Title case the result to match how _pretty_column_name works
        return result.title().replace(' ', '-') if '-' in raw_name else result.title()

    key_order = ORDER_MAP.get('Overall_keys', [])
    sorted_keys = sorted(raw_tag_map.keys(), key=lambda x: key_order.index(x) if x in key_order else len(key_order))
    for raw_key in sorted_keys:
        raw_value_list = raw_tag_map[raw_key]
        pretty_key = get_pretty(raw_key)
        pretty_value_list = [get_pretty(raw_val) for raw_val in raw_value_list]

        # Get the unique values first
        unique_values = list(set(pretty_value_list))
        # Get the custom order for the current key. Fall back to an empty list.
        custom_order = ORDER_MAP.get(pretty_key, [])
        def sort_key(value):
            if value in custom_order:
                return 0, custom_order.index(value)
            else:
                return 1, value
        pretty_map[pretty_key] = sorted(unique_values, key=sort_key)

    print(f"Created pretty tag map: {pretty_map}")
    return pretty_map


def transform_raw_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms a raw leaderboard DataFrame into a presentation-ready format.

    This function performs two main actions:
    1. Rounds all numeric metric values (columns containing 'score' or 'cost').
    2. Renames all columns to a "pretty", human-readable format.
    Args:
        raw_df (pd.DataFrame): The DataFrame with raw data and column names
                               like 'agent_name', 'overall/score', 'tag/code/cost'.
    Returns:
        pd.DataFrame: A new DataFrame ready for display.
    """
    if not isinstance(raw_df, pd.DataFrame):
        raise TypeError("Input 'raw_df' must be a pandas DataFrame.")

    df = raw_df.copy()
    # Create the mapping for pretty column names
    pretty_cols_map = {col: _pretty_column_name(col) for col in df.columns}

    # Rename the columns and return the new DataFrame
    transformed_df = df.rename(columns=pretty_cols_map)
    # Apply safe rounding to all metric columns
    for col in transformed_df.columns:
        if 'Score' in col or 'Cost' in col:
            transformed_df[col] = transformed_df[col].apply(_safe_round)

    logger.info("Raw DataFrame transformed: numbers rounded and columns renamed.")
    return transformed_df


class DataTransformer:
    """
    Visualizes a pre-processed leaderboard DataFrame.

    This class takes a "pretty" DataFrame and a tag map, and provides
    methods to view filtered versions of the data and generate plots.
    """
    def __init__(self, dataframe: pd.DataFrame, tag_map: dict[str, list[str]]):
        """
        Initializes the viewer.
        Args:
            dataframe (pd.DataFrame): The presentation-ready leaderboard data.
            tag_map (dict): A map of formal tag names to formal task names.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input 'dataframe' must be a pandas DataFrame.")
        if not isinstance(tag_map, dict):
            raise TypeError("Input 'tag_map' must be a dictionary.")

        self.data = dataframe
        self.tag_map = tag_map
        logger.info(f"DataTransformer initialized with a DataFrame of shape {self.data.shape}.")


    def view(
            self,
            tag: Optional[str] = "Overall", # Default to "Overall" for clarity
            use_plotly: bool = False,
    ) -> tuple[pd.DataFrame, dict[str, go.Figure]]:
        """
        Generates a filtered view of the DataFrame and a corresponding scatter plot.
        """
        if self.data.empty:
            logger.warning("No data available to view.")
            return self.data, {}

        # --- 1. Determine Primary and Group Metrics Based on the Tag ---
        if tag is None or tag == "Overall":
            # Use "Average" for the primary metric display name
            primary_metric = "Average"
            group_metrics = list(self.tag_map.keys())
        else:
            primary_metric = tag
            # For a specific tag, the group is its list of sub-tasks.
            group_metrics = self.tag_map.get(tag, [])

        # --- 2. Sort the DataFrame by the Primary Score ---
        primary_score_col = f"{primary_metric} Score"
        df_sorted = self.data
        if primary_score_col in self.data.columns:
            df_sorted = self.data.sort_values(primary_score_col, ascending=False, na_position='last')

        df_view = df_sorted.copy()

        # --- 3. Add Columns for Agent Openness ---
        base_cols = ["id","Language Model","SDK Version","Source"]
        new_cols = ["Openness"]
        ending_cols = ["Date", "Logs", "Visualization"]

        # For Overall view, use "Average Cost" and "Average Runtime" (per instance across all benchmarks)
        if tag is None or tag == "Overall":
            primary_cost_col = "Average Cost"
            primary_runtime_col = "Average Runtime"
        else:
            primary_cost_col = f"{primary_metric} Cost"
            primary_runtime_col = f"{primary_metric} Runtime"

        metrics_to_display = [primary_score_col, primary_cost_col, primary_runtime_col]
        for item in group_metrics:
            metrics_to_display.append(f"{item} Score")
            metrics_to_display.append(f"{item} Cost")
            metrics_to_display.append(f"{item} Runtime")

        final_cols_ordered = new_cols + base_cols +  list(dict.fromkeys(metrics_to_display)) + ending_cols

        for col in final_cols_ordered:
            if col not in df_view.columns:
                df_view[col] = pd.NA

        # The final selection will now use the new column structure
        df_view = df_view[final_cols_ordered].reset_index(drop=True)
        cols = len(final_cols_ordered)

        # Calculated and add "Categories Attempted" column
        if tag is None or tag == "Overall":
            def calculate_attempted(row):
                main_categories = ['Issue Resolution', 'Frontend', 'Greenfield', 'Testing', 'Information Gathering']
                count = 0
                for category in main_categories:
                    value = row.get(f"{category} Score")
                    # A score of 0.0 is a valid result - only exclude truly missing values
                    if pd.notna(value):
                        count += 1
                return f"{count}/5"

            # Apply the function row-wise to create the new column
            attempted_column = df_view.apply(calculate_attempted, axis=1)
            # Insert the new column at a nice position (e.g., after "Date")
            df_view.insert((cols - 2), "Categories Attempted", attempted_column)
        else:
            total_benchmarks = len(group_metrics)
            def calculate_benchmarks_attempted(row):
                # Count how many benchmarks in this category have COST data reported
                count = sum(1 for benchmark in group_metrics if pd.notna(row.get(f"{benchmark} Score")))
                return f"{count}/{total_benchmarks}"
            # Insert the new column, for example, after "Date"
            df_view.insert((cols - 2), "Benchmarks Attempted", df_view.apply(calculate_benchmarks_attempted, axis=1))

        # --- 4. Generate the Scatter Plot for the Primary Metric ---
        plots: dict[str, go.Figure] = {}
        if use_plotly:
            # primary_cost_col is already set above (Average Cost for Overall, or {metric} Cost otherwise)
            # Check if the primary score and cost columns exist in the FINAL view
            if primary_score_col in df_view.columns and primary_cost_col in df_view.columns:
                fig = _plot_scatter_plotly(
                    data=df_view,
                    x=primary_cost_col,
                    y=primary_score_col,
                    agent_col="SDK Version",
                    name=primary_metric
                ) if use_plotly else go.Figure()
                # Use a consistent key for easy retrieval later
                plots['scatter_plot'] = fig
            else:
                logger.warning(
                    f"Skipping plot for '{primary_metric}': score column '{primary_score_col}' "
                    f"or cost column '{primary_cost_col}' not found."
                )
                # Add an empty figure to avoid downstream errors
                plots['scatter_plot'] = go.Figure()
        return df_view, plots

DEFAULT_Y_COLUMN = "Average Score"
DUMMY_X_VALUE_FOR_MISSING_COSTS = 0

def _plot_scatter_plotly(
        data: pd.DataFrame,
        x: Optional[str],
        y: str,
        agent_col: str = 'Agent',
        name: Optional[str] = None,
        plot_type: str = 'cost',  # 'cost' or 'runtime'
        mark_by: Optional[str] = None  # 'Company', 'Openness', or 'Country'
) -> go.Figure:
    from constants import MARK_BY_DEFAULT
    if mark_by is None:
        mark_by = MARK_BY_DEFAULT

    # --- Section 1: Define Mappings ---
    # Map openness to colors (simplified: open vs closed)
    color_map = {
        aliases.CANONICAL_OPENNESS_OPEN: "deeppink",
        aliases.CANONICAL_OPENNESS_CLOSED: "yellow",
    }
    for canonical_openness, openness_aliases in aliases.OPENNESS_ALIASES.items():
        for openness_alias in openness_aliases:
            color_map[openness_alias] = color_map[canonical_openness]
    # Only keep one name per color for the legend.
    colors_for_legend = set(aliases.OPENNESS_ALIASES.keys())
    category_order = list(color_map.keys())

    # Use consistent marker shape (no tooling distinction)
    default_shape = 'circle'

    x_col_to_use = x
    y_col_to_use = y
    llm_base = data["Language Model"] if "Language Model" in data.columns else "Language Model"

    # --- Section 2: Data Preparation---
    required_cols = [y_col_to_use, agent_col, "Openness"]
    if not all(col in data.columns for col in required_cols):
        logger.error(f"Missing one or more required columns for plotting: {required_cols}")
        return go.Figure()

    data_plot = data.copy()
    data_plot[y_col_to_use] = pd.to_numeric(data_plot[y_col_to_use], errors='coerce')

    # Set axis labels based on plot type
    if plot_type == 'runtime':
        x_axis_label = f"Average runtime per problem (seconds)" if x else "Runtime (Data N/A)"
    else:
        x_axis_label = f"Average cost per problem (USD)" if x else "Cost (Data N/A)"
    max_reported_cost = 0
    divider_line_x = 0

    if x and x in data_plot.columns:
        data_plot[x_col_to_use] = pd.to_numeric(data_plot[x_col_to_use], errors='coerce')

        # --- Separate data into two groups ---
        valid_cost_data = data_plot[data_plot[x_col_to_use].notna()].copy()
        missing_cost_data = data_plot[data_plot[x_col_to_use].isna()].copy()

        # Hardcode for all missing costs for now, but ideally try to fallback
        # to the max cost in the same figure in another split, if that one has data...
        max_reported_cost = valid_cost_data[x_col_to_use].max() if not valid_cost_data.empty else 10

        # ---Calculate where to place the missing data and the divider line ---
        divider_line_x = max_reported_cost + (max_reported_cost/10)
        new_x_for_missing = max_reported_cost + (max_reported_cost/5)
        if not missing_cost_data.empty:
            missing_cost_data[x_col_to_use] = new_x_for_missing

        if not valid_cost_data.empty:
            if not missing_cost_data.empty:
                # --- Combine the two groups back together ---
                data_plot = pd.concat([valid_cost_data, missing_cost_data])
            else:
                data_plot = valid_cost_data # No missing data, just use the valid set
        else:
            # ---Handle the case where ALL costs are missing ---
            if not missing_cost_data.empty:
                data_plot = missing_cost_data
            else:
                data_plot = pd.DataFrame()
    else:
        # Handle case where x column is not provided at all
        data_plot[x_col_to_use] = 0

    # Clean data based on all necessary columns
    data_plot.dropna(subset=[y_col_to_use, x_col_to_use, "Openness"], inplace=True)

    # --- Section 3: Initialize Figure ---
    fig = go.Figure()
    if data_plot.empty:
        logger.warning(f"No valid data to plot after cleaning.")
        return fig

    # --- Section 4: Calculate and Draw Pareto Frontier ---
    frontier_rows = []  # Store entire rows for frontier points to access model names
    if x_col_to_use and y_col_to_use:
        sorted_data = data_plot.sort_values(by=[x_col_to_use, y_col_to_use], ascending=[True, False])
        frontier_points = []
        max_score_so_far = float('-inf')

        for _, row in sorted_data.iterrows():
            score = row[y_col_to_use]
            if score >= max_score_so_far:
                frontier_points.append({'x': row[x_col_to_use], 'y': score})
                frontier_rows.append(row)
                max_score_so_far = score

        if frontier_points:
            frontier_df = pd.DataFrame(frontier_points)
            fig.add_trace(go.Scatter(
                x=frontier_df['x'],
                y=frontier_df['y'],
                mode='lines',
                name='Efficiency Frontier',
                showlegend=False,
                line=dict(color='#FFE165', width=2, dash='dash'),  # primary yellow
                hoverinfo='skip'
            ))

    # --- Section 5: Prepare for Marker Plotting ---
    def format_hover_text(row, agent_col, x_axis_label, x_col, y_col, divider_line_x, is_runtime=False):
        """
        Builds the complete HTML string for the plot's hover tooltip.
        Format: {lm_name} (SDK {version})
                Average Score: {score}
                Average Cost/Runtime: {value}
                Openness: {openness}
        """
        h_pad = "   "
        parts = ["<br>"]
        
        # Get and clean the language model name
        llm_base_value = row.get('Language Model', '')
        llm_base_value = clean_llm_base_list(llm_base_value)
        if isinstance(llm_base_value, list) and llm_base_value:
            lm_name = llm_base_value[0]
        else:
            lm_name = str(llm_base_value) if llm_base_value else 'Unknown'
        
        # Get SDK version
        sdk_version = row.get('SDK Version', row.get(agent_col, 'Unknown'))
        
        # Title line: {lm_name} (SDK {version})
        parts.append(f"{h_pad}<b>{lm_name}</b> (SDK {sdk_version}){h_pad}<br>")
        
        # Average Score
        parts.append(f"{h_pad}Average Score: <b>{row[y_col]:.3f}</b>{h_pad}<br>")
        
        # Average Cost or Runtime
        if is_runtime:
            if divider_line_x > 0 and row[x_col] >= divider_line_x:
                parts.append(f"{h_pad}Average Runtime: <b>Missing</b>{h_pad}<br>")
            else:
                parts.append(f"{h_pad}Average Runtime: <b>{row[x_col]:.0f}s</b>{h_pad}<br>")
        else:
            if divider_line_x > 0 and row[x_col] >= divider_line_x:
                parts.append(f"{h_pad}Average Cost: <b>Missing</b>{h_pad}<br>")
            else:
                parts.append(f"{h_pad}Average Cost: <b>${row[x_col]:.2f}</b>{h_pad}<br>")
        
        # Openness
        parts.append(f"{h_pad}Openness: <b>{row['Openness']}</b>{h_pad}")
        
        # Add final line break for padding
        parts.append("<br>")
        return ''.join(parts)
    # Pre-generate hover text and shapes for each point
    data_plot['hover_text'] = data_plot.apply(
        lambda row: format_hover_text(
            row,
            agent_col=agent_col,
            x_axis_label=x_axis_label,
            x_col=x_col_to_use,
            y_col=y_col_to_use,
            divider_line_x=divider_line_x,
            is_runtime=(plot_type == 'runtime')
        ),
        axis=1
    )
    # Use consistent shape for all points (no tooling distinction)
    data_plot['shape_symbol'] = default_shape

    # --- Section 6: Plot Company Logo Images as Markers (replacing open/closed distinction) ---
    # Collect layout images for company logos
    layout_images = []
    
    # Add invisible markers for hover functionality (all points together, no color distinction)
    fig.add_trace(go.Scatter(
        x=data_plot[x_col_to_use],
        y=data_plot[y_col_to_use],
        mode='markers',
        name='Models',
        showlegend=False,
        text=data_plot['hover_text'],
        hoverinfo='text',
        marker=dict(
            color='rgba(0,0,0,0)',  # Invisible markers
            size=25,  # Large enough for hover detection
            opacity=0
        )
    ))
    
    # Add company logo images for each data point
    # Using domain coordinates (0-1 range) to work correctly with log scale x-axis
    # Calculate axis ranges for coordinate conversion
    min_cost = data_plot[x_col_to_use].min()
    max_cost = data_plot[x_col_to_use].max()
    min_score = data_plot[y_col_to_use].min()
    max_score = data_plot[y_col_to_use].max()
    
    # For log scale, we need log10 of the range bounds
    # Add padding to the range
    x_min_log = np.log10(min_cost * 0.5) if min_cost > 0 else -2
    x_max_log = np.log10(max_cost * 1.3) if max_cost > 0 else 1
    y_min = min_score - 5 if min_score > 5 else 0
    y_max = max_score + 5
    
    for _, row in data_plot.iterrows():
        model_name = row.get('Language Model', '')
        openness = row.get('Openness', '')
        marker_info = get_marker_icon(model_name, openness, mark_by)
        logo_path = marker_info['path']
        
        # Read the SVG file and encode as base64 data URI
        if os.path.exists(logo_path):
            try:
                with open(logo_path, 'rb') as f:
                    encoded_logo = base64.b64encode(f.read()).decode('utf-8')
                    logo_uri = f"data:image/svg+xml;base64,{encoded_logo}"
                    
                    x_val = row[x_col_to_use]
                    y_val = row[y_col_to_use]
                    
                    # Convert to domain coordinates (0-1 range)
                    # For log scale x: domain_x = (log10(x) - x_min_log) / (x_max_log - x_min_log)
                    if x_val > 0:
                        log_x = np.log10(x_val)
                        domain_x = (log_x - x_min_log) / (x_max_log - x_min_log)
                    else:
                        domain_x = 0
                    
                    # For linear y: domain_y = (y - y_min) / (y_max - y_min)
                    domain_y = (y_val - y_min) / (y_max - y_min) if (y_max - y_min) > 0 else 0.5
                    
                    # Clamp to valid range
                    domain_x = max(0, min(1, domain_x))
                    domain_y = max(0, min(1, domain_y))
                    
                    layout_images.append(dict(
                        source=logo_uri,
                        xref="x domain",  # Use domain coordinates for log scale compatibility
                        yref="y domain",
                        x=domain_x,
                        y=domain_y,
                        sizex=0.04,  # Size as fraction of plot width
                        sizey=0.06,  # Size as fraction of plot height
                        xanchor="center",
                        yanchor="middle",
                        layer="above"
                    ))
            except Exception as e:
                logger.warning(f"Could not load logo {logo_path}: {e}")

    # --- Section 7: Add Model Name Labels to Frontier Points ---
    if frontier_rows:
        frontier_labels_data = []
        
        for row in frontier_rows:
            x_val = row[x_col_to_use]
            y_val = row[y_col_to_use]
            
            # Get the model name for the label
            model_name = row.get('Language Model', '')
            if isinstance(model_name, list):
                model_name = model_name[0] if model_name else ''
            # Clean the model name (remove path prefixes)
            model_name = str(model_name).split('/')[-1]
            # Truncate long names
            if len(model_name) > 25:
                model_name = model_name[:22] + '...'
            
            frontier_labels_data.append({
                'x': x_val,
                'y': y_val,
                'label': model_name
            })
        
        # Add annotations for each frontier label
        # For log scale x-axis, annotations need log10(x) coordinates (Plotly issue #2580)
        for item in frontier_labels_data:
            x_val = item['x']
            y_val = item['y']
            label = item['label']
            
            # Transform x to log10 for annotation positioning on log scale
            if x_val > 0:
                x_log = np.log10(x_val)
            else:
                x_log = x_min_log
            
            fig.add_annotation(
                x=x_log,
                y=y_val,
                text=label,
                showarrow=False,
                yshift=25,  # Move label higher above the icon
                font=dict(
                    size=10,
                    color='#0D0D0F',  # neutral-950
                    family=FONT_FAMILY_SHORT
                ),
                xanchor='center',
                yanchor='bottom'
            )

    # --- Section 8: Configure Layout  ---
    # Use the same axis ranges as calculated for domain coordinates
    xaxis_config = dict(
        title=x_axis_label,
        type="log",
        range=[x_min_log, x_max_log]  # Match domain coordinate calculation
    )

    # Set title based on plot type
    if plot_type == 'runtime':
        plot_title = f"OpenHands Index {name} Runtime/Performance"
    else:
        plot_title = f"OpenHands Index {name} Cost/Performance"

    # Build layout configuration - colors aligned with OpenHands brand
    layout_config = dict(
        template="plotly_white",
        title=plot_title,
        xaxis=xaxis_config,
        yaxis=dict(title="Average score", range=[y_min, y_max]),  # Match domain calculation
        legend=dict(
            bgcolor='#F7F8FB',  # neutral-50
        ),
        height=572,
        font=dict(
            family=FONT_FAMILY,
            color="#0D0D0F",  # neutral-950
        ),
        hoverlabel=dict(
            bgcolor="#222328",  # neutral-800
            font_size=12,
            font_family=FONT_FAMILY_SHORT,
            font_color="#F7F8FB",  # neutral-50
        ),
        # Add margin at bottom for logo and URL
        margin=dict(b=80),
    )
    
    # Add company logo images to the layout if any were collected
    if layout_images:
        layout_config['images'] = layout_images
    
    fig.update_layout(**layout_config)
    
    # Add OpenHands branding (logo and URL)
    add_branding_to_figure(fig)

    return fig


def format_cost_column(df: pd.DataFrame, cost_col_name: str) -> pd.DataFrame:
    """
    Applies custom formatting to a cost column based on its corresponding score column.
    - If cost is not null, it remains unchanged.
    - If cost is null but score is not, it becomes "Missing Cost".
    - If both cost and score are null, it becomes "Not Attempted".
    Args:
        df: The DataFrame to modify.
        cost_col_name: The name of the cost column to format (e.g., "Average Cost").
    Returns:
        The DataFrame with the formatted cost column.
    """
    # Find the corresponding score column by replacing "Cost" with "Score"
    score_col_name = cost_col_name.replace("Cost", "Score")

    # Ensure the score column actually exists to avoid errors
    if score_col_name not in df.columns:
        return df # Return the DataFrame unmodified if there's no matching score

    def apply_formatting_logic(row):
        cost_value = row[cost_col_name]
        score_value = row[score_col_name]
        status_color = "#ec4899"

        if pd.notna(cost_value) and isinstance(cost_value, (int, float)):
            return f"${cost_value:.2f}"
        elif pd.notna(score_value):
            return f'<span style="color: {status_color};">Missing</span>'  # Score exists, but cost is missing
        else:
            return f'<span style="color: {status_color};">Not Submitted</span>'  # Neither score nor cost exists

    # Apply the logic to the specified cost column and update the DataFrame
    df[cost_col_name] = df.apply(apply_formatting_logic, axis=1)

    return df

def format_score_column(df: pd.DataFrame, score_col_name: str) -> pd.DataFrame:
    """
    Applies custom formatting to a score column for display.
    - If a score is 0 or NaN, it's displayed as a colored "0".
    - Other scores are formatted to two decimal places.
    - Average Score values are displayed in bold.
    """
    status_color = "#ec4899"  # The same color as your other status text
    is_average_score = (score_col_name == "Average Score")

    def apply_formatting(score_value):
        # Explicitly handle missing values without turning them into zeros
        if pd.isna(score_value):
            return f'<span style="color: {status_color};">Not Submitted</span>'
        # Show true zero distinctly
        if isinstance(score_value, (int, float)) and score_value == 0:
            formatted = f'<span style="color: {status_color};">0.0</span>'
        elif isinstance(score_value, (int, float)):
            formatted = f"{score_value:.3f}"
        else:
            formatted = str(score_value)
        
        # Make Average Score bold
        if is_average_score and score_value != 0:
            return f"<strong>{formatted}</strong>"
        return formatted

    # Apply the formatting and return the updated DataFrame
    return df.assign(**{score_col_name: df[score_col_name].apply(apply_formatting)})


def format_runtime_column(df: pd.DataFrame, runtime_col_name: str) -> pd.DataFrame:
    """
    Applies custom formatting to a runtime column based on its corresponding score column.
    - If runtime is not null, formats as time with 's' suffix.
    - If runtime is null but score is not, it becomes "Missing".
    - If both runtime and score are null, it becomes "Not Submitted".
    Args:
        df: The DataFrame to modify.
        runtime_col_name: The name of the runtime column to format (e.g., "Average Runtime").
    Returns:
        The DataFrame with the formatted runtime column.
    """
    # Find the corresponding score column by replacing "Runtime" with "Score"
    score_col_name = runtime_col_name.replace("Runtime", "Score")

    # Ensure the score column actually exists to avoid errors
    if score_col_name not in df.columns:
        return df  # Return the DataFrame unmodified if there's no matching score

    def apply_formatting_logic(row):
        runtime_value = row[runtime_col_name]
        score_value = row[score_col_name]
        status_color = "#ec4899"

        if pd.notna(runtime_value) and isinstance(runtime_value, (int, float)):
            return f"{runtime_value:.0f}s"
        elif pd.notna(score_value):
            return f'<span style="color: {status_color};">Missing</span>'  # Score exists, but runtime is missing
        else:
            return f'<span style="color: {status_color};">Not Submitted</span>'  # Neither score nor runtime exists

    # Apply the logic to the specified runtime column and update the DataFrame
    df[runtime_col_name] = df.apply(apply_formatting_logic, axis=1)

    return df


def format_date_column(df: pd.DataFrame, date_col_name: str = "Date") -> pd.DataFrame:
    """
    Formats a date column to show only the date part (YYYY-MM-DD), removing the time.
    
    Args:
        df: The DataFrame to modify.
        date_col_name: The name of the date column to format (default: "Date").
    
    Returns:
        The DataFrame with the formatted date column.
    """
    if date_col_name not in df.columns:
        return df  # Return the DataFrame unmodified if the column doesn't exist
    
    def apply_date_formatting(date_value):
        if pd.isna(date_value) or date_value == '':
            return ''
        
        # Handle ISO format strings like "2025-11-24T19:56:00.092865"
        if isinstance(date_value, str):
            # Extract just the date part (before the 'T')
            if 'T' in date_value:
                return date_value.split('T')[0]
            # If it's already in date format, return as-is
            return date_value[:10] if len(date_value) >= 10 else date_value
        
        # Handle pandas Timestamp or datetime objects
        try:
            return pd.to_datetime(date_value).strftime('%Y-%m-%d')
        except (ValueError, TypeError):
            return str(date_value)
    
    df[date_col_name] = df[date_col_name].apply(apply_date_formatting)
    return df


def get_pareto_df(data, cost_col=None, score_col=None):
    """
    Calculate the Pareto frontier for the given data.
    
    Args:
        data: DataFrame with cost and score columns
        cost_col: Specific cost column to use (default: 'Average Cost')
        score_col: Specific score column to use (default: 'Average Score')
    
    Returns:
        DataFrame containing only the rows on the Pareto frontier
    """
    # Use Average Cost/Score by default for the Overall leaderboard
    if cost_col is None:
        cost_col = 'Average Cost' if 'Average Cost' in data.columns else None
        if cost_col is None:
            cost_cols = [c for c in data.columns if 'Cost' in c]
            cost_col = cost_cols[0] if cost_cols else None
    
    if score_col is None:
        score_col = 'Average Score' if 'Average Score' in data.columns else None
        if score_col is None:
            score_cols = [c for c in data.columns if 'Score' in c]
            score_col = score_cols[0] if score_cols else None
    
    if cost_col is None or score_col is None:
        return pd.DataFrame()

    frontier_data = data.dropna(subset=[cost_col, score_col]).copy()
    frontier_data[score_col] = pd.to_numeric(frontier_data[score_col], errors='coerce')
    frontier_data[cost_col] = pd.to_numeric(frontier_data[cost_col], errors='coerce')
    frontier_data.dropna(subset=[cost_col, score_col], inplace=True)
    if frontier_data.empty:
        return pd.DataFrame()

    # Sort by cost ascending, then by score descending
    frontier_data = frontier_data.sort_values(by=[cost_col, score_col], ascending=[True, False])

    pareto_points = []
    max_score_at_cost = -np.inf

    for _, row in frontier_data.iterrows():
        if row[score_col] >= max_score_at_cost:
            pareto_points.append(row)
            max_score_at_cost = row[score_col]

    return pd.DataFrame(pareto_points)


def clean_llm_base_list(model_list):
    """
    Cleans a list of model strings by keeping only the text after the last '/'.
    For example: "models/gemini-2.5-flash-preview-05-20" becomes "gemini-2.5-flash-preview-05-20".
    """
    # Return the original value if it's not a list, to avoid errors.
    if not isinstance(model_list, list):
        return model_list

    # Use a list comprehension for a clean and efficient transformation.
    return [str(item).split('/')[-1] for item in model_list]
