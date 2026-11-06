MODELS = [
    {
        "name": "fpt-reranker-primary",
        "model_env": "RERANKER_MODEL",
        "model_default": "reranker-model",
        "url_env": "RERANK_URL",
        "base_url_env": "BASE_URL",
        "base_url_default": "https://api.example.com/v1",
        "path": "/rerank",
        "api_key_env": "API_KEY",
        "timeout": 20,
        "attempts": 1,
    },
]
