"""Only equivalent embedding spaces may fail over against an existing index."""
MODELS = [
    {
        "name": "fpt-e5-primary",
        "model_env": "EMBEDDING_MODEL",
        "model_default": "embedding-model",
        "url_env": "EMBEDDING_URL",
        "api_key_env": ("API_KEY", "LLM_API_KEY"),
        "dimension": 1024,
        "timeout": 60,
        "attempts": 1,
    },
]
