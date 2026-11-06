MODELS = [
    {
        "name": "jev-primary",
        "model_env": "DECISION_MODEL",
        "model_default": "decision-model",
        "url_env": "DECISION_BASE_URL",
        "url_default": "https://api.example.com/v1",
        "api_key_env": ("DECISION_API_KEY", "DECISION_API_KEY"),
        "timeout": 8,
        "attempts": 1,
    },
]
