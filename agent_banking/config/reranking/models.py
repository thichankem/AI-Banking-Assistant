MODELS = [
    {
        "name": "fpt-reranker-primary",
        "model_env": "RERANKER_MODEL",
        "model_default": "bge-reranker-v2-m3",
        "url_env": "RERANK_URL",
        "base_url_env": "BASE_URL",
        "base_url_default": "https://mkp-api.fptcloud.com/v1",
        "path": "/rerank",
        "api_key_env": "API_KEY",
        "timeout": 20,
        "attempts": 1,
    },
]
