# Qwen3.6-Plus Metadata Sources

This document lists the sources for each metadata field value for Qwen3.6-Plus, as requested in issue #778.

## model: "Qwen3.6-Plus"
**Source:** https://qwen.ai/blog?id=qwen3.6  
**Quote:** "Following the release of the Qwen3.5 series in February, we are thrilled to announce the official launch of Qwen3.6-Plus."

## release_date: "2026-04-01"
**Source:** https://qwen.ai/blog?id=qwen3.6  
**Quote:** From the blog post header: "2026/04/01 · 31 minute · 6222 words · QwenTeam丨Translations:简体中文"

## supports_vision: true
**Source:** https://qwen.ai/blog?id=qwen3.6  
**Quote:** "Qwen3.6-Plus marks a steady progress in multimodal capabilities, evolving across three core dimensions: advanced reasoning, enhanced applicability, and ability to execute complex tasks."

**Additional source:** https://www.alibabacloud.com/help/en/model-studio/models  
**Quote:** "Qwen3.6-Plus supports text, image, and video inputs."

## openness: "closed_api_available"
**Source:** https://qwen.ai/blog?id=qwen3.6  
**Quote:** "Qwen3.6-Plus is now generally available through our official API via Alibaba Cloud Model Studio."

**Additional source:** https://www.alibabacloud.com/help/en/model-studio/models  
**Note:** Model is only available via API, not as downloadable weights, confirming closed_api_available status.

## country: "cn"
**Source:** https://qwen.ai/blog?id=qwen3.6  
**Quote:** "Qwen3.6-Plus is now generally available through our official API via Alibaba Cloud Model Studio."

**Additional context:** Alibaba Cloud and the Qwen team are based in China, and the model is developed and hosted by Alibaba.

## input_price: 0.5 (per 1M tokens in USD)
**Source:** https://www.alibabacloud.com/help/en/model-studio/models  
**Quote:** From the "Flagship models" pricing table for International service deployment:
"Min input price (per 1M tokens): $0.5"

## output_price: 3.0 (per 1M tokens in USD)
**Source:** https://www.alibabacloud.com/help/en/model-studio/models  
**Quote:** From the "Flagship models" pricing table for International service deployment:
"Min output price (per 1M tokens): $3"

## parameter_count_b: null
**Note:** Parameter count is not publicly disclosed for this closed API model. The Qwen blog post and official documentation do not specify the exact parameter count for Qwen3.6-Plus.

## active_parameter_count_b: null
**Note:** Active parameter count is not publicly disclosed. While the model may use a Mixture-of-Experts (MoE) architecture based on related Qwen models, specific active parameter counts are not documented in the official sources.

## cache_read_price: null
**Note:** Cache pricing is not mentioned in the official pricing documentation at https://www.alibabacloud.com/help/en/model-studio/models

## cache_write_price: null
**Note:** Cache pricing is not mentioned in the official pricing documentation at https://www.alibabacloud.com/help/en/model-studio/models

## Context Window
**Source:** https://www.alibabacloud.com/help/en/model-studio/models  
**Quote:** "Max context window (tokens): 1,000,000"

**Additional source:** https://qwen.ai/blog?id=qwen3.6  
**Quote:** "Qwen3.6-Plus is the hosted model available via Alibaba Cloud Model Studio, featuring: a 1M context window by default"
