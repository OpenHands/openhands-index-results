from content import BUG_FIXING_DESCRIPTION
from category_page_builder import build_category_page

# Define the category for this page
CATEGORY_NAME = "Issue Resolution"

def build_page():
    build_category_page(CATEGORY_NAME, BUG_FIXING_DESCRIPTION)
