"""Models in descending priority; credentials are environment references."""

MODELS = [
    {
        "name": "primary",
        "model_env": "MODEL",
        "model_default": "",
        "url_env": "BASE_URL",
        "api_key_env": "API_KEY",
        "timeout": 45,
        "attempts": 1,
    },
]
