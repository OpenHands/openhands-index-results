"""
Additional visualizations for the OpenHands Index leaderboard.

These functions use the generic create_scatter_chart() from leaderboard_transformer
as the single source of truth for scatter plot styling and behavior.
"""
import pandas as pd
import plotly.graph_objects as go
import aliases

# Import the generic scatter chart function - single source of truth
from leaderboard_transformer import create_scatter_chart, STANDARD_LAYOUT, STANDARD_FONT


def _find_column(df: pd.DataFrame, candidates: list, default: str = None) -> str:
    """Find the first matching column name from candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    return default


def create_evolution_over_time_chart(df: pd.DataFrame, mark_by: str = None) -> go.Figure:
    """
    Create a chart showing model performance evolution over release dates.
    
    Args:
        df: DataFrame with release_date and score columns
        mark_by: One of "Company", "Openness", or "Country" for marker icons
    
    Returns:
        Plotly figure showing score evolution over time
    """
    # Find the release date column
    release_date_col = _find_column(df, ['release_date', 'Release_Date', 'Release Date'])
    
    if df.empty or release_date_col is None:
        fig = go.Figure()
        fig.add_annotation(
            text="No release date data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=STANDARD_FONT
        )
        fig.update_layout(**STANDARD_LAYOUT, title="Model Performance Evolution Over Time")
        return fig
    
    # Find score column
    score_col = _find_column(df, ['Average Score', 'average score', 'Average score'])
    if score_col is None:
        # Try to find any column with 'score' and 'average'
        for col in df.columns:
            if 'score' in col.lower() and 'average' in col.lower():
                score_col = col
                break
    
    if score_col is None:
        fig = go.Figure()
        fig.add_annotation(
            text="No score data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=STANDARD_FONT
        )
        fig.update_layout(**STANDARD_LAYOUT, title="Model Performance Evolution Over Time")
        return fig
    
    # Use the generic scatter chart
    return create_scatter_chart(
        df=df,
        x_col=release_date_col,
        y_col=score_col,
        title="Model Performance Evolution Over Time",
        x_label="Model Release Date",
        y_label="Average Score",
        mark_by=mark_by,
        x_type="date",
        pareto_lower_is_better=False,  # Later dates with higher scores are better
    )


def create_accuracy_by_size_chart(df: pd.DataFrame, mark_by: str = None) -> go.Figure:
    """
    Create a scatter plot showing accuracy vs parameter count for open-weights models.
    
    Args:
        df: DataFrame with parameter_count and score columns
        mark_by: One of "Company", "Openness", or "Country" for marker icons
    
    Returns:
        Plotly figure showing accuracy vs model size
    """
    # Find parameter count column
    param_col = _find_column(df, ['parameter_count_b', 'Parameter_Count_B', 'Parameter Count B'])
    
    if df.empty or param_col is None:
        fig = go.Figure()
        fig.add_annotation(
            text="No parameter count data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=STANDARD_FONT
        )
        fig.update_layout(**STANDARD_LAYOUT, title="Open Model Accuracy by Size")
        return fig
    
    # Filter to only open-weights models
    open_aliases = [aliases.CANONICAL_OPENNESS_OPEN] + list(
        aliases.OPENNESS_ALIASES.get(aliases.CANONICAL_OPENNESS_OPEN, [])
    )
    openness_col = 'Openness' if 'Openness' in df.columns else 'openness'
    
    plot_df = df[
        (df[param_col].notna()) & 
        (df[openness_col].isin(open_aliases))
    ].copy()
    
    if plot_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No open-weights models with parameter data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=STANDARD_FONT
        )
        fig.update_layout(**STANDARD_LAYOUT, title="Open Model Accuracy by Size")
        return fig
    
    # Find score column
    score_col = _find_column(plot_df, ['Average Score', 'average score', 'Average score'])
    if score_col is None:
        for col in plot_df.columns:
            if 'score' in col.lower() and 'average' in col.lower():
                score_col = col
                break
    
    if score_col is None:
        fig = go.Figure()
        fig.add_annotation(
            text="No score data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=STANDARD_FONT
        )
        fig.update_layout(**STANDARD_LAYOUT, title="Open Model Accuracy by Size")
        return fig
    
    # Use the generic scatter chart
    return create_scatter_chart(
        df=plot_df,
        x_col=param_col,
        y_col=score_col,
        title="Open Model Accuracy by Size",
        x_label="Parameters (Billions)",
        y_label="Average Score",
        mark_by=mark_by,
        x_type="log",
        pareto_lower_is_better=True,  # Smaller models with higher scores are better
    )
