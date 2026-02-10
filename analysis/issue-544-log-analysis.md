# Performance Regression Analysis: v1.8.3 vs v1.11.0 on SWE-bench (Qwen3-Coder)

## Executive Summary

This analysis investigates the performance regression reported in Issue #544, where SWE-bench accuracy dropped from 62.4% (v1.8.3) to 47.2% (v1.11.0). After thorough log analysis, we identified **three key factors** contributing to this regression:

1. **CRITICAL: Different Models Used** - The evaluations used different model sizes (480B vs 30B parameters)
2. **Infrastructure Issues** - v1.11.0 experienced significant runtime/network issues
3. **Configuration Changes** - The critic mode changed between versions

## 1. Infrastructure/Network Issues Analysis

### v1.8.3 Error Analysis
- **Total errors**: 3 instances (0.6% error rate)
- **Error type**: `LLMBadRequestError: litellm.BadRequestError: Error code: 400`
- **Cause**: LLM API request errors (likely malformed requests or context length issues)

### v1.11.0 Error Analysis
- **Total errors**: 2 instances (0.4% error rate)
- **Error type**: `Remote conversation not found (404). The runtime may have been deleted.`
- **Infrastructure issues in eval.log**: 3,159 occurrences of transient errors including:
  - `502 Bad Gateway`
  - `503 Service Unavailable`
  - `Server disconnected without sending a response`
  - `StatusCode.UNAVAILABLE` (trace export failures)

**Conclusion**: While v1.11.0 had more infrastructure instability, the error rates were similar (0.4-0.6%). Infrastructure issues alone do not explain the 15.2 percentage point drop.

## 2. Instance Comparison: django__django-11265

This instance was resolved in v1.8.3 but failed in v1.11.0.

| Metric | v1.8.3 | v1.11.0 |
|--------|--------|---------|
| Patch Size | 3,168 bytes | 14,095 bytes |
| Accumulated Cost | $3.10 | $0.00* |
| Approach | Copy `_filtered_relations` to subquery | Copy `annotations` to subquery |
| Result | ✅ Passed | ❌ Failed |

*v1.11.0 cost data appears to be missing/zero

**Key Observation**: The v1.8.3 patch was smaller and more focused, while v1.11.0 produced a larger patch with a different (incorrect) approach.

## 3. Software-Agent-SDK Changes (v1.8.3 → v1.10.0 → v1.11.0)

### Key Changes in v1.9.0
- Added experimental critic model support

### Key Changes in v1.10.0
- Multi-summary views for condenser (#1721)
- Plugin loading support (#1651, #1676)
- Tool immutability enforcement on conversation restore (#1788)

### Key Changes in v1.11.0
- **Cap event history scanned by StuckDetector** (#1829) - May affect stuck detection
- **Hard context reset on unrecoverable error** (#1596) - Could cause context loss
- **Truncate terminal outputs before persisting events** (#1823) - May lose important output
- **Avoid materializing full event history in Agent.init_state** (#1840) - Memory optimization

### Configuration Difference
```json
// v1.8.3
"critic": {"kind": "AgentFinishedCritic"}

// v1.11.0
"critic": {"mode": "finish_and_message", "kind": "AgentFinishedCritic"}
```

The addition of `"mode": "finish_and_message"` may affect how the agent decides when to stop working on a problem.

## 4. CRITICAL FINDING: Different Models Used

**This is the most significant finding and likely the primary cause of the regression.**

| Version | Model | Parameters | Active Parameters |
|---------|-------|------------|-------------------|
| v1.8.3 | `qwen3-coder-480b-a35b-instruct` | 480B | 35B |
| v1.11.0 | `Qwen3-Coder-30B-A3B-Instruct` | 30B | 3B |

The v1.11.0 evaluation used a model with:
- **16x fewer total parameters** (480B → 30B)
- **~12x fewer active parameters** (35B → 3B)

This massive difference in model capability is the most likely explanation for:
- The 15.2 percentage point accuracy drop
- Larger but less effective patches
- Different (incorrect) fix approaches

## Recommendations

1. **Re-run v1.11.0 evaluation with the same model** (qwen3-coder-480b-a35b-instruct) to get a fair comparison

2. **Investigate the critic mode change** - The `finish_and_message` mode may be causing premature termination

3. **Monitor infrastructure stability** - The high number of transient errors in v1.11.0 suggests infrastructure improvements may be needed

4. **Document model requirements** - Ensure evaluations use consistent models for valid comparisons

## Data Sources

- v1.8.3 archive: `eval-20979851181-qwen-3-cod_litellm_proxy-fireworks_ai-qwen3-coder-480b-a35b-instruct_26-01-14-09-02.tar.gz`
- v1.11.0 archive: `swebench/litellm_proxy-Qwen3-Coder-30B-A3B-Instruct/21688863397/results.tar.gz`

## Appendix: Detailed Error Logs

### v1.8.3 Error Instances
1. `sphinx-doc__sphinx-7757` - LLMBadRequestError
2. `sphinx-doc__sphinx-9281` - LLMBadRequestError
3. `pytest-dev__pytest-7792` - LLMBadRequestError

### v1.11.0 Error Instances
1. `django__django-13012` - Remote conversation not found (404)
2. `django__django-15368` - Remote conversation not found (404)
