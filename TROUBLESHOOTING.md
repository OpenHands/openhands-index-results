# Troubleshooting Guide for OpenHands Index

## Issue: Model Missing from GAIA Plot but Present in Table

### Symptoms
- A model (e.g., claude-sonnet-4-5) appears in the benchmark table on the leaderboard
- The same model is missing from the corresponding scatter plot
- Data exists correctly in this repository

### Root Cause
The [OpenHands Index HuggingFace Space](https://huggingface.co/spaces/OpenHands/openhands-index) uses two data sources:
1. **Primary**: Fetches data directly from this GitHub repository (`openhands-index-results`)
2. **Fallback**: Uses local `mock_results` directory when GitHub fetch fails

When the HuggingFace Space falls back to mock results, it may use outdated data that doesn't include recently added models.

### Solution
There are two ways to fix this:

#### Option 1: Update HuggingFace Space Mock Results (Immediate Fix)
1. Clone the HuggingFace Space repository:
   ```bash
   git clone https://huggingface.co/spaces/OpenHands/openhands-index
   cd openhands-index
   ```

2. Copy the missing model data from this repository:
   ```bash
   cp -r /path/to/openhands-index-results/results/<model-name> \
         mock_results/1.0.0-dev1/results/
   ```

3. Commit and push to HuggingFace:
   ```bash
   git add mock_results/
   git commit -m "Add <model-name> to mock results"
   git push origin main
   ```

#### Option 2: Force Data Refresh on HuggingFace Space (Preferred)
The HuggingFace Space should automatically fetch fresh data from this repository. To force a refresh:

1. Wait for the cache TTL to expire (default: 15 minutes)
2. Or restart the HuggingFace Space to clear the cache
3. Or manually trigger a refresh through the Space's UI (if available)

### Prevention
To prevent this issue in the future:
1. Ensure the HuggingFace Space's data fetching is working correctly
2. Keep the mock_results directory synchronized with the latest data
3. Consider adding a GitHub Action to automatically sync mock data to HuggingFace Space

### Example: Fixing claude-sonnet-4-5 Missing from GAIA Plot
This issue was resolved by adding the claude-sonnet-4-5 data to the HuggingFace Space's mock_results directory:

```bash
# The data already existed in this repository at:
# results/claude-sonnet-4-5/metadata.json
# results/claude-sonnet-4-5/scores.json

# It was copied to the HuggingFace Space at:
# mock_results/1.0.0-dev1/results/claude-sonnet-4-5/
```

The GAIA benchmark data for claude-sonnet-4-5:
- Score: 58.8
- Cost per instance: $0.38
- Average runtime: 126s

### Related Files
- HuggingFace Space data loader: `simple_data_loader.py`
- HuggingFace Space plot generator: `leaderboard_transformer.py` (function: `_plot_scatter_plotly`)
- HuggingFace Space data setup: `setup_data.py`
