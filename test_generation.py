from content import TEST_GENERATION_DESCRIPTION
from category_page_builder import build_category_page

# Define the category for this page
CATEGORY_NAME = "Testing"

def build_page():
    build_category_page(CATEGORY_NAME, TEST_GENERATION_DESCRIPTION)
