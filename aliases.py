# Define constants for openness categories
CANONICAL_OPENNESS_OPEN = "open"
CANONICAL_OPENNESS_CLOSED = "closed"

# Map raw openness values to simplified display values
OPENNESS_MAPPING = {
    # Open-weights models (publicly available weights)
    "open_weights": CANONICAL_OPENNESS_OPEN,
    "open_source_open_weights": CANONICAL_OPENNESS_OPEN,
    "open_source_closed_weights": CANONICAL_OPENNESS_OPEN,
    "open": CANONICAL_OPENNESS_OPEN,
    # Closed models (API-only or no public access)
    "closed_api_available": CANONICAL_OPENNESS_CLOSED,
    "closed_ui_only": CANONICAL_OPENNESS_CLOSED,
    "closed": CANONICAL_OPENNESS_CLOSED,
}

OPENNESS_ALIASES = {
    CANONICAL_OPENNESS_OPEN: {"Open", "Open Source", "Open Source + Open Weights"},
    CANONICAL_OPENNESS_CLOSED: {"Closed", "API Available"}
}
