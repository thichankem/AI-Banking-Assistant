
class ChunkingConfig:
    # 2. Cấu hình Chunking (Layout-Aware)
    CHUNK_MAX_SIZE = 800          # Kích thước ký tự tối đa cho 1 text chunk
    KEEP_TABLE_ATOMIC = True      # Giữ nguyên bảng biểu làm 1 đơn vị dữ liệu độc lập

