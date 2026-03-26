# Issue #694: Empty output.jsonl for qwen3-coder-480b on commit0

## Summary

All 16 instances in the commit0 benchmark evaluation for `qwen3-coder-480b` (via LiteLLM proxy with Fireworks AI) failed immediately during runtime initialization, resulting in empty `output.jsonl` files and a score of 0.0.

## Affected Evaluations

- Run 1: https://results.eval.all-hands.dev/commit0/litellm_proxy-fireworks_ai-qwen3-coder-480b-a35b-instruct/22104205466/results.tar.gz
- Run 2: https://results.eval.all-hands.dev/commit0/litellm_proxy-fireworks_ai-qwen3-coder-480b-a35b-instruct/22925541086/results.tar.gz

## Analysis

### Key Findings

1. **100% Failure Rate**: All 16 instances failed across multiple retry attempts
2. **Consistent Error Pattern**: Every instance failed with "Remote conversation ended with error"
3. **Timing**: Failures occurred immediately after `conversation.run()` was triggered
4. **Infrastructure OK**: Runtime pods started successfully, health checks passed, repositories were cloned

### Diagnostic Results

Using the `diagnose_eval_failure.py` script on the evaluation results:

```
📊 REPORT SUMMARY:
  Total Instances: 16
  Submitted: 0
  Completed: 0
  Resolved: 0
  Errors: 0
  Total Tests: 0

📄 OUTPUT.JSONL STATUS:
  ❌ EMPTY (0 bytes)

📋 LOG ANALYSIS:
  Total Log Files: 32
  Failure Patterns:
    - runtime_init_failure: 16
    - conversation_run_failed: 16
    - remote_conversation_error: 16
```

### Timeline of Failure (Example from babel instance)

1. ✅ Runtime pod created and became healthy (21:45:47)
2. ✅ Repository cloned: commit-0/babel (21:45:48)
3. ✅ Branch created: openhands (21:45:48)
4. ✅ pytest installed and verified (21:46:36)
5. ✅ Runtime allocated successfully
6. ✅ `conversation.run()` triggered: HTTP 200 OK (21:46:37)
7. ❌ **FAILURE**: "Remote conversation ended with error" (21:46:38)
8. 🔄 Retry attempts with increased resources (factors: 2x, 4x, 8x)
9. ❌ All retries failed with the same error

## Root Cause Analysis

### Hypothesis: LLM API Configuration Issue

The failure pattern strongly indicates an issue with the LLM model endpoint rather than the evaluation infrastructure:

**Evidence supporting LLM API issue:**
- Infrastructure components worked correctly (pods, networking, storage)
- Setup phase completed successfully (cloning, installation, verification)
- Failure occurred when the agent attempted to use the LLM
- 100% consistent failure rate across all instances and retries
- Error occurred immediately, not after timeout

**Possible specific causes:**

1. **Model Endpoint Unavailable**
   - The qwen3-coder-480b-a35b-instruct model may not be available via Fireworks AI
   - Endpoint may be temporarily down or deprecated

2. **Authentication/Authorization Failure**
   - API key may not have access to this specific model
   - Fireworks AI may require different authentication for qwen3-coder-480b

3. **Invalid Response Format**
   - Model may be returning responses in an unexpected format
   - Response format may be incompatible with LiteLLM proxy or OpenHands SDK expectations

4. **Rate Limiting**
   - Less likely since failures were immediate and consistent across time

5. **Model Configuration Mismatch**
   - Model name in configuration may not match Fireworks AI's model identifier
   - Required parameters may be missing from LiteLLM configuration

## Comparison with Other Benchmarks

Interestingly, the same model configuration succeeded on other benchmarks:

- **gaia**: 33.9% accuracy ✅
- **swe-bench**: 62.4% accuracy ✅
- **swt-bench**: 34.9% accuracy ✅
- **swe-bench-multimodal**: 23.5% accuracy ✅
- **commit0**: 0.0% accuracy ❌

This suggests the issue may be specific to the commit0 benchmark setup or the timing of when the evaluation was run (model availability may have changed).

## Recommendations

### Immediate Actions

1. **Verify Model Availability**
   ```bash
   # Check if qwen3-coder-480b-a35b-instruct is available via Fireworks AI
   curl -X POST "https://api.fireworks.ai/inference/v1/chat/completions" \
     -H "Authorization: Bearer $FIREWORKS_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "accounts/fireworks/models/qwen3-coder-480b-a35b-instruct",
       "messages": [{"role": "user", "content": "Hello"}]
     }'
   ```

2. **Check LiteLLM Configuration**
   - Review litellm_proxy configuration for qwen3-coder-480b
   - Verify model name mapping is correct
   - Ensure all required parameters are set

3. **Review Agent Server Logs**
   - Check agent-server logs for more detailed error messages
   - Look for LLM API response errors or exceptions

4. **Test with Alternative Configuration**
   - Try different model endpoint if available
   - Test with direct Fireworks AI API (bypass LiteLLM if possible)
   - Verify with a simple test conversation before full evaluation

### Long-term Solutions

1. **Add Pre-flight Checks**
   - Test LLM connectivity before starting evaluations
   - Validate model availability and response format
   - Implement health checks for LLM endpoints

2. **Improve Error Reporting**
   - Capture and log LLM API responses for failed conversations
   - Include model-specific error details in evaluation reports
   - Add structured error categorization (auth, availability, format, etc.)

3. **Add Diagnostic Tooling**
   - Use `scripts/diagnose_eval_failure.py` to analyze failed evaluations
   - Create automated alerts for 100% failure rate scenarios
   - Build dashboard to track model availability and success rates

## How to Use the Diagnostic Script

To diagnose similar issues in the future:

```bash
# Analyze a failed evaluation
python scripts/diagnose_eval_failure.py <results_tar_gz_url>

# Example
python scripts/diagnose_eval_failure.py \
  https://results.eval.all-hands.dev/commit0/litellm_proxy-fireworks_ai-qwen3-coder-480b-a35b-instruct/22925541086/results.tar.gz

# Get JSON output for programmatic processing
python scripts/diagnose_eval_failure.py --json <results_tar_gz_url>
```

## Related Issues

- GitHub Actions runs mentioned in comments:
  - https://github.com/OpenHands/software-agent-sdk/actions/runs/22925519709
  - Evaluation workflow: https://github.com/OpenHands/evaluation/actions/runs/22923425530

## Status

- **Issue Opened**: 2026-03-10
- **Re-evaluation Attempts**: 2 (both failed with same symptoms)
- **Root Cause**: LLM API configuration or availability issue (hypothesis)
- **Impact**: commit0 benchmark score incorrectly reported as 0.0% for qwen3-coder-480b
- **Resolution**: Pending - requires investigation of LiteLLM/Fireworks AI configuration

## Next Steps

1. Coordinate with infrastructure team to check LiteLLM proxy configuration
2. Verify qwen3-coder-480b model availability with Fireworks AI
3. Re-run evaluation after configuration is confirmed working
4. Update this analysis with findings and resolution
