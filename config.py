import os

LOCAL_DEBUG = not (os.environ.get("system") == "spaces")
CONFIG_NAME = os.getenv("HF_CONFIG", "1.0.0-dev1") # This corresponds to 'config' in LeaderboardViewer
IS_INTERNAL = os.environ.get("IS_INTERNAL", "false").lower() == "true"

# OpenHands Index datasets
if IS_INTERNAL:
    RESULTS_DATASET = f"OpenHands/openhands-index-internal-results"
    LEADERBOARD_PATH = f"OpenHands/openhands-index-internal-leaderboard"
else:
    RESULTS_DATASET = f"OpenHands/openhands-index-results"
    LEADERBOARD_PATH = f"OpenHands/openhands-index"

DATA_DIR = "/tmp/oh_index/data/" + CONFIG_NAME
EXTRACTED_DATA_DIR = os.path.join(DATA_DIR, "extracted")
