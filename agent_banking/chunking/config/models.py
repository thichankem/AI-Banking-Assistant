"""Chunking model configurations for text extraction, vision analysis, and visual review."""

TEXT_MODELS = [
    {
        "name": "primary",
        "model_env": "MODEL",
        "model_default": "",
        "url_env": "BASE_URL",
        "api_key_env": "API_KEY",
        "timeout": 240,
        "attempts": 3,
    },
]

VISION_MODELS = [
    {
        "name": "primary",
        "model_env": ("CHUNKING_VISION_MODEL", "VISION_MODEL"),
        "model_default": "Qwen3.6-27B",
        "url_env": ("CHUNKING_VISION_BASE_URL", "BASE_URL"),
        "api_key_env": ("CHUNKING_VISION_API_KEY", "API_KEY"),
        "timeout": 240,
        "attempts": 3,
    },
]

# The FPT Cloud defaults use two different vision model families.
VISION_REVIEW_MODELS = [
    {
        "name": "vision-review-primary",
        "model_env": ("CHUNKING_VISION_REVIEW_MODEL", "VISION_REVIEW_MODEL"),
        "model_default": "Azure:gpt-5.6-luna",
        "url_env": ("CHUNKING_VISION_REVIEW_BASE_URL", "BASE_URL"),
        "api_key_env": ("CHUNKING_VISION_REVIEW_API_KEY", "API_KEY"),
        "timeout": 240,
        "attempts": 3,
    },
    {
        "name": "vision-review-fallback",
        "model_env": "CHUNKING_VISION_REVIEW_FALLBACK_MODEL",
        "model_default": "gemma-4-31B-it",
        "url_env": ("CHUNKING_VISION_REVIEW_BASE_URL", "BASE_URL"),
        "api_key_env": ("CHUNKING_VISION_REVIEW_API_KEY", "API_KEY"),
        "timeout": 240,
        "attempts": 2,
    },
]

DECISION_MODELS = [
    {
        "name": "primary",
        "model_env": "DECISION_MODEL",
        "model_default": "",
        "url_env": ("DECISION_BASE_URL", "BASE_URL"),
        "url_default": "",
        "api_key_env": ("DECISION_API_KEY", "API_KEY"),
        "timeout": 8,
        "attempts": 3,
    },
]
