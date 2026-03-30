from content import INFORMATION_GATHERING_DESCRIPTION
from category_page_builder import build_category_page

# Define the category for this page
CATEGORY_NAME = "Information Gathering"

def build_page():
    build_category_page(CATEGORY_NAME, INFORMATION_GATHERING_DESCRIPTION)
