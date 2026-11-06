TEXT_MODELS = [
    {
        "name": "fpt-primary",
        "model_env": "MODEL",
        "model_default": "Language-Model",
        "url_env": "BASE_URL",
        "api_key_env": "API_KEY",
        "timeout": 240,
        "attempts": 3,
    },
]

VISION_MODELS = [
    {
        "name": "fpt-primary",
        "model_env": "VISION_MODEL",
        "model_default": "Qwen2.5-VL-7B-Instruct",
        "url_env": "BASE_URL",
        "api_key_env": "API_KEY",
        "timeout": 240,
        "attempts": 3,
    },
]

DECISION_MODELS = [
    {
        "name": "jev-primary",
        "model_env": "DECISION_MODEL",
        "model_default": "decision-model",
        "url_env": "DECISION_BASE_URL",
        "url_default": "https://api.example.com/v1",
        "api_key_env": ("DECISION_API_KEY", "DECISION_API_KEY"),
        "timeout": 8,
        "attempts": 3,
    },
]
