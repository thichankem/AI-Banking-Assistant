"""Reranking model configuration; models in descending priority."""

MODELS = [
    {
        "name": "primary",
        "model_env": "RERANKER_MODEL",
        "model_default": "",
        "url_env": "RERANK_URL",
        "base_url_env": "BASE_URL",
        "base_url_default": "",
        "path": "/rerank",
        "api_key_env": "API_KEY",
        "timeout": 20,
        "attempts": 1,
    },
]
