"""
Setup script to fetch data from GitHub repository or use mock data as fallback.
This runs before the app starts to ensure data is available.
Supports periodic refresh with caching.
"""
import os
import shutil
import subprocess
import sys
import threading
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from config import DATA_DIR, EXTRACTED_DATA_DIR, CONFIG_NAME

logger = logging.getLogger(__name__)

GITHUB_REPO = "https://github.com/OpenHands/openhands-index-results.git"
# Keep the full repo clone so we can use pydantic models from scripts/
REPO_CLONE_DIR = Path("/tmp/openhands-index-results")

# Cache management
_last_fetch_time = None
_fetch_lock = threading.Lock()
_refresh_callbacks = []  # Callbacks to call after data refresh

# Cache TTL can be configured via environment variable (default: 15 minutes = 900 seconds)
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", 900))


def get_repo_clone_dir() -> Path:
    """Get the path to the cloned openhands-index-results repository."""
    return REPO_CLONE_DIR


def fetch_data_from_github():
    """
    Fetch data from the openhands-index-results GitHub repository.
    Keeps the full repo clone for access to pydantic schema models.
    Returns True if successful, False otherwise.
    """
    clone_dir = REPO_CLONE_DIR
    target_dir = Path(EXTRACTED_DATA_DIR) / CONFIG_NAME
    
    try:
        # Remove existing clone directory if it exists
        if clone_dir.exists():
            shutil.rmtree(clone_dir)
        
        print(f"Attempting to clone data from {GITHUB_REPO}...")
        
        # Set environment to prevent git from prompting for credentials
        env = os.environ.copy()
        env['GIT_TERMINAL_PROMPT'] = '0'
        
        # Clone the repository (keep full repo for pydantic models)
        result = subprocess.run(
            ["git", "clone", "--depth", "1", GITHUB_REPO, str(clone_dir)],
            capture_output=True,
            text=True,
            timeout=30,
            stdin=subprocess.DEVNULL,
            env=env
        )
        
        if result.returncode != 0:
            print(f"Git clone failed: {result.stderr}")
            return False
        
        # Look for data files in the cloned repository
        results_source = clone_dir / "results"
        
        if not results_source.exists():
            print(f"Results directory not found in repository")
            return False
        
        # Check if there are any agent result directories
        result_dirs = list(results_source.iterdir())
        if not result_dirs:
            print(f"No agent results found in {results_source}")
            return False
        
        print(f"Found {len(result_dirs)} agent result directories")
        
        # Create target directory and copy the results structure
        os.makedirs(target_dir.parent, exist_ok=True)
        if target_dir.exists():
            shutil.rmtree(target_dir)
        
        # Copy the entire results directory
        target_results = target_dir / "results"
        shutil.copytree(results_source, target_results)
        
        print(f"Successfully fetched data from GitHub. Files: {list(target_dir.glob('*'))}")
        
        # Verify data integrity by checking a sample agent
        sample_agents = list(target_results.glob("*/scores.json"))
        if sample_agents:
            import json
            with open(sample_agents[0]) as f:
                sample_data = json.load(f)
                print(f"Sample data from {sample_agents[0].parent.name}: {sample_data[0] if sample_data else 'EMPTY'}")
        
        # Add the repo's scripts directory to Python path for pydantic models
        scripts_dir = clone_dir / "scripts"
        if scripts_dir.exists() and str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
            print(f"Added {scripts_dir} to Python path for schema imports")
        
        return True
        
    except subprocess.TimeoutExpired:
        print("Git clone timed out")
        return False
    except Exception as e:
        print(f"Error fetching data from GitHub: {e}")
        return False

def copy_mock_data():
    """Copy mock data to the expected extraction directory."""
    # Try both relative and absolute paths
    mock_source = Path("mock_results") / CONFIG_NAME
    if not mock_source.exists():
        # Try absolute path in case we're in a different working directory
        mock_source = Path("/app/mock_results") / CONFIG_NAME
    
    target_dir = Path(EXTRACTED_DATA_DIR) / CONFIG_NAME
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Looking for mock data at: {mock_source.absolute()}")
    
    if not mock_source.exists():
        print(f"ERROR: Mock data directory {mock_source} not found!")
        print(f"Directory contents: {list(Path('.').glob('*'))}")
        return False
    
    # Create target directory
    os.makedirs(target_dir.parent, exist_ok=True)
    
    # Copy mock data
    print(f"Using mock data from {mock_source}")
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(mock_source, target_dir)
    
    # Verify the copy
    copied_files = list(target_dir.glob('*'))
    print(f"Mock data copied successfully. Files: {copied_files}")
    print(f"Target directory: {target_dir.absolute()}")
    return True

def register_refresh_callback(callback):
    """
    Register a callback to be called after data is refreshed.
    The callback should clear any caches that depend on the data.
    """
    global _refresh_callbacks
    if callback not in _refresh_callbacks:
        _refresh_callbacks.append(callback)


def _notify_refresh_callbacks():
    """Notify all registered callbacks that data has been refreshed."""
    global _refresh_callbacks
    for callback in _refresh_callbacks:
        try:
            callback()
        except Exception as e:
            logger.warning(f"Error in refresh callback: {e}")


def get_last_fetch_time():
    """Get the timestamp of the last successful data fetch."""
    global _last_fetch_time
    return _last_fetch_time


def is_cache_stale():
    """Check if the cache has expired (older than CACHE_TTL_SECONDS)."""
    global _last_fetch_time
    if _last_fetch_time is None:
        return True
    return datetime.now() - _last_fetch_time > timedelta(seconds=CACHE_TTL_SECONDS)


def refresh_data_if_needed(force: bool = False) -> bool:
    """
    Refresh data from GitHub if the cache is stale or if forced.
    
    Args:
        force: If True, refresh regardless of cache age
        
    Returns:
        True if data was refreshed, False otherwise
    """
    global _last_fetch_time, _fetch_lock
    
    if not force and not is_cache_stale():
        return False
    
    # Use lock to prevent concurrent refreshes
    if not _fetch_lock.acquire(blocking=False):
        logger.info("Another refresh is in progress, skipping...")
        return False
    
    try:
        logger.info("Refreshing data from GitHub...")
        
        # Remove old data to force re-fetch
        target_dir = Path(EXTRACTED_DATA_DIR) / CONFIG_NAME
        if target_dir.exists():
            shutil.rmtree(target_dir)
        
        # Fetch new data
        if fetch_data_from_github():
            _last_fetch_time = datetime.now()
            logger.info(f"✓ Data refreshed successfully at {_last_fetch_time}")
            _notify_refresh_callbacks()
            return True
        else:
            # If GitHub fails, try mock data as fallback
            logger.warning("GitHub fetch failed, trying mock data...")
            if copy_mock_data():
                _last_fetch_time = datetime.now()
                logger.info(f"✓ Using mock data (refreshed at {_last_fetch_time})")
                _notify_refresh_callbacks()
                return True
            logger.error("Failed to refresh data from any source")
            return False
    finally:
        _fetch_lock.release()


def setup_mock_data():
    """
    Setup data for the leaderboard.
    First tries to fetch from GitHub, falls back to mock data if unavailable.
    """
    global _last_fetch_time
    
    print("=" * 60)
    print("STARTING DATA SETUP")
    print("=" * 60)
    
    target_dir = Path(EXTRACTED_DATA_DIR) / CONFIG_NAME
    
    # Check if data already exists and cache is not stale
    if target_dir.exists():
        results_dir = target_dir / "results"
        has_results = results_dir.exists() and any(results_dir.iterdir())
        has_jsonl = any(target_dir.glob("*.jsonl"))
        
        if has_results or has_jsonl:
            if not is_cache_stale():
                print(f"Data already exists at {target_dir} and cache is fresh")
                return
            print(f"Data exists but cache is stale, will refresh...")
    
    # Try to fetch from GitHub first
    print("\n--- Attempting to fetch from GitHub ---")
    try:
        if fetch_data_from_github():
            _last_fetch_time = datetime.now()
            print("✓ Successfully using data from GitHub repository")
            return
    except Exception as e:
        print(f"GitHub fetch failed with error: {e}")
    
    # Fall back to mock data
    print("\n--- GitHub data not available, falling back to mock data ---")
    try:
        if copy_mock_data():
            _last_fetch_time = datetime.now()
            print("✓ Successfully using mock data")
            return
    except Exception as e:
        print(f"Mock data copy failed with error: {e}")
    
    print("\n" + "!" * 60)
    print("ERROR: No data available! Neither GitHub nor mock data could be loaded.")
    print("!" * 60)


# Background refresh scheduler
_scheduler_thread = None
_scheduler_stop_event = threading.Event()


def _background_refresh_loop():
    """Background thread that periodically refreshes data."""
    global _scheduler_stop_event
    
    logger.info(f"Background refresh scheduler started (interval: {CACHE_TTL_SECONDS}s)")
    
    while not _scheduler_stop_event.is_set():
        # Wait for the TTL interval (or until stopped)
        if _scheduler_stop_event.wait(timeout=CACHE_TTL_SECONDS):
            break  # Stop event was set
        
        # Try to refresh data
        try:
            logger.info("Background refresh triggered")
            refresh_data_if_needed(force=True)
        except Exception as e:
            logger.error(f"Error in background refresh: {e}")
    
    logger.info("Background refresh scheduler stopped")


def start_background_refresh():
    """Start the background refresh scheduler."""
    global _scheduler_thread, _scheduler_stop_event
    
    if _scheduler_thread is not None and _scheduler_thread.is_alive():
        logger.info("Background refresh scheduler already running")
        return
    
    _scheduler_stop_event.clear()
    _scheduler_thread = threading.Thread(target=_background_refresh_loop, daemon=True)
    _scheduler_thread.start()
    logger.info("Background refresh scheduler started")


def stop_background_refresh():
    """Stop the background refresh scheduler."""
    global _scheduler_thread, _scheduler_stop_event
    
    if _scheduler_thread is None:
        return
    
    _scheduler_stop_event.set()
    _scheduler_thread.join(timeout=5)
    _scheduler_thread = None
    logger.info("Background refresh scheduler stopped")


if __name__ == "__main__":
    setup_mock_data()
