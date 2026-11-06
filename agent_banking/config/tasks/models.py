"""Models in descending priority; credentials are environment references."""
MODELS = [
    {
        "name": "fpt-primary",
        "model_env": "MODEL",
        "model_default": "DeepSeek-V4-Flash",
        "url_env": "BASE_URL",
        "api_key_env": "API_KEY",
        "timeout": 45,
        "attempts": 1,
    },
]
