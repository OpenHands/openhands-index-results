from content import APP_CREATION_DESCRIPTION
from category_page_builder import build_category_page

# Define the category for this page
CATEGORY_NAME = "Greenfield"

def build_page():
    build_category_page(CATEGORY_NAME, APP_CREATION_DESCRIPTION)
