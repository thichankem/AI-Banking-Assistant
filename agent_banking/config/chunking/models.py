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
        "model_env": "CHUNKING_VISION_MODEL",
        "model_default": "",
        "url_env": "BASE_URL",
        "api_key_env": "API_KEY",
        "timeout": 240,
        "attempts": 3,
    },
]

# Independent visual review; selected separately from the extraction model.
# These scoped overrides keep generic VISION_MODEL settings for other modules.
VISION_REVIEW_MODELS = [
    {
        "name": "vision-review-primary",
        "model_env": "CHUNKING_VISION_REVIEW_MODEL",
        "model_default": "",
        "url_env": "BASE_URL",
        "api_key_env": "API_KEY",
        "timeout": 240,
        "attempts": 3,
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
