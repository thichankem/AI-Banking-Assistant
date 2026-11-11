"""Decision model configuration for routing and verification."""

MODELS = [
    {
        "name": "primary",
        "model_env": "DECISION_MODEL",
        "model_default": "",
        "url_env": ("DECISION_BASE_URL", "BASE_URL"),
        "url_default": "",
        "api_key_env": ("DECISION_API_KEY", "API_KEY"),
        "timeout": 8,
        "attempts": 1,
    },
]
