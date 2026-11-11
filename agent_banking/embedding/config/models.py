"""Only equivalent embedding spaces may fail over against an existing index."""

MODELS = [
    {
        "name": "primary",
        "model_env": "EMBEDDING_MODEL",
        "model_default": "",
        "url_env": "EMBEDDING_URL",
        "base_url_env": "BASE_URL",
        "path": "/embeddings",
        "api_key_env": ("API_KEY", "LLM_API_KEY"),
        "dimension": 1024,
        "timeout": 60,
        "attempts": 1,
    },
]
