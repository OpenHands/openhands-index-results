import matplotlib
matplotlib.use('Agg')
import gradio as gr
import pandas as pd


from ui_components import (
    create_leaderboard_display,
    get_full_leaderboard_data,
    create_winners_by_category_html,
)

from content import (
    CITATION_BUTTON_LABEL,
    CITATION_BUTTON_TEXT,
    INTRO_PARAGRAPH
)

from visualizations import (
    create_evolution_over_time_chart,
    create_accuracy_by_size_chart
)

from constants import MARK_BY_DEFAULT

# --- Global State for Viewers (simple caching) ---
CACHED_VIEWERS = {}
CACHED_TAG_MAPS = {}


def filter_complete_entries(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    category_score_columns = [
        'Issue Resolution Score',
        'Frontend Score',
        'Greenfield Score',
        'Testing Score',
        'Information Gathering Score',
    ]

    if all(column in df.columns for column in category_score_columns):
        return df[df[category_score_columns].notna().all(axis=1)].copy()

    if 'Categories Completed' in df.columns:
        categories_completed = pd.to_numeric(df['Categories Completed'], errors='coerce')
        return df[categories_completed >= 5].copy()

    if 'Categories Attempted' in df.columns:
        return df[df['Categories Attempted'] == '5/5'].copy()

    return df.copy()


def build_page():
    with gr.Row(elem_id="intro-row"):
        with gr.Column(scale=1):
            gr.HTML(INTRO_PARAGRAPH, elem_id="intro-paragraph")

    # --- Leaderboard Display Section ---
    gr.Markdown("---")
    CATEGORY_NAME = "Overall"
    gr.HTML(f'<h2>OpenHands Index {CATEGORY_NAME} Leaderboard <span style="font-weight: normal; color: inherit;">(Aggregate)</span></h2>', elem_id="main-header")

    test_df, test_tag_map = get_full_leaderboard_data("test")
    if not test_df.empty:
        show_incomplete_checkbox, show_open_only_checkbox, mark_by_dropdown = create_leaderboard_display(
            full_df=test_df,
            tag_map=test_tag_map,
            category_name=CATEGORY_NAME,
            split_name="test"
        )

        test_df_complete = filter_complete_entries(test_df)
        has_complete_entries = len(test_df_complete) > 0

        if 'Openness' in test_df.columns:
            test_df_open = test_df[test_df['Openness'].str.lower() == 'open'].copy()
        else:
            test_df_open = test_df.copy()
        test_df_complete_open = filter_complete_entries(test_df_open)

        initial_df = test_df_complete if has_complete_entries else test_df

        # --- Winners by Category Section ---
        gr.Markdown("---")
        gr.HTML('<h2>Winners by Category</h2>', elem_id="winners-header")
        gr.Markdown("Top 5 performing systems in each benchmark category.")

        winners_component = gr.HTML(
            create_winners_by_category_html(initial_df, top_n=5),
            elem_id="winners-by-category",
        )

        # --- New Visualization Sections ---
        gr.Markdown("---")

        # Evolution Over Time Section
        gr.HTML('<h2>Evolution Over Time</h2>', elem_id="evolution-header")
        gr.Markdown("Track how model performance has improved over time based on release dates.")

        evolution_component = gr.Plot(
            value=create_evolution_over_time_chart(initial_df, MARK_BY_DEFAULT),
            elem_id="evolution-chart",
        )

        gr.Markdown("---")

        # Open Model Accuracy by Size Section (always shows open models only by design)
        gr.HTML('<h2>Open Model Accuracy by Size</h2>', elem_id="size-accuracy-header")
        gr.Markdown("Compare open-weights model performance against their parameter count.")

        size_component = gr.Plot(
            value=create_accuracy_by_size_chart(initial_df, MARK_BY_DEFAULT),
            elem_id="size-accuracy-chart",
        )

        def update_extra_sections(show_incomplete, show_open_only, mark_by):
            include_incomplete = show_incomplete or not has_complete_entries
            base_df = test_df if include_incomplete else test_df_complete
            base_df_open = test_df_open if include_incomplete else test_df_complete_open
            winners_df = base_df_open if show_open_only else base_df

            winners_html = create_winners_by_category_html(winners_df, top_n=5)
            evolution_fig = create_evolution_over_time_chart(winners_df, mark_by)
            size_fig = create_accuracy_by_size_chart(base_df, mark_by)

            return winners_html, evolution_fig, size_fig

        show_incomplete_input = show_incomplete_checkbox if show_incomplete_checkbox is not None else gr.State(value=True)
        show_open_only_input = show_open_only_checkbox if show_open_only_checkbox is not None else gr.State(value=False)
        extra_section_inputs = [show_incomplete_input, show_open_only_input, mark_by_dropdown]

        if show_incomplete_checkbox is not None:
            show_incomplete_checkbox.change(
                fn=update_extra_sections,
                inputs=extra_section_inputs,
                outputs=[winners_component, evolution_component, size_component]
            )

        if show_open_only_checkbox is not None:
            show_open_only_checkbox.change(
                fn=update_extra_sections,
                inputs=extra_section_inputs,
                outputs=[winners_component, evolution_component, size_component]
            )

        if mark_by_dropdown is not None:
            mark_by_dropdown.change(
                fn=update_extra_sections,
                inputs=extra_section_inputs,
                outputs=[winners_component, evolution_component, size_component]
            )
        
    else:
        gr.Markdown("No data available.")

if __name__ == "__main__":
    demo.launch()