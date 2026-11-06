MODELS = [
    {
        "name": "jev-primary",
        "model_env": "JEV_MODEL",
        "model_default": "jev-latest",
        "url_env": "JEV_BASE_URL",
        "url_default": "https://api.typesafe.ai/v1",
        "api_key_env": ("JEV_API_KEY", "TYPESAFE_API_KEY"),
        "timeout": 8,
        "attempts": 1,
    },
]
