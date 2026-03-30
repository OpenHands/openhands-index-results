import gradio as gr
import pandas as pd

# Import our UI factories and the data loader
from ui_components import create_leaderboard_display, create_benchmark_details_display, get_full_leaderboard_data, create_sub_navigation_bar

def build_category_page(CATEGORY_NAME, PAGE_DESCRIPTION):
    with gr.Column(elem_id="page-content-wrapper"):
        test_df, test_tag_map = get_full_leaderboard_data("test")
        
        gr.HTML(f'<h2>OpenHands Index {CATEGORY_NAME} Leaderboard <span style="font-weight: normal; color: inherit;">(Aggregate)</span></h2>', elem_id="main-header")
        with gr.Column(elem_id="test_nav_container", visible=True) as test_nav_container:
            create_sub_navigation_bar(test_tag_map, CATEGORY_NAME)

        gr.Markdown(PAGE_DESCRIPTION, elem_id="intro-category-paragraph")

        if not test_df.empty:
            create_leaderboard_display(
                full_df=test_df,
                tag_map=test_tag_map,
                category_name=CATEGORY_NAME,
                split_name="test"
            )
            create_benchmark_details_display(
                full_df=test_df,
                tag_map=test_tag_map,
                category_name=CATEGORY_NAME,
                validation=False,
            )
        else:
            gr.Markdown("No data available.")

    return test_nav_container