import os

class TracingConfig:
    # 9. Cấu hình Tracing & Observability (Arize Phoenix & Local Tracing)
    ENABLE_TRACING = os.getenv("ENABLE_TRACING", "true").lower() == "true"

    # Cấu hình Arize Phoenix (RAG OpenTelemetry Tracing)
    ENABLE_PHOENIX = os.getenv("ENABLE_PHOENIX", "true").lower() == "true"
    PHOENIX_COLLECTOR_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "")
    PHOENIX_PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME", "")

