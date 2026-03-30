from content import FRONTEND_DEVELOPMENT_DESCRIPTION
from category_page_builder import build_category_page

# Define the category for this page
CATEGORY_NAME = "Frontend"

def build_page():
    build_category_page(CATEGORY_NAME, FRONTEND_DEVELOPMENT_DESCRIPTION)
