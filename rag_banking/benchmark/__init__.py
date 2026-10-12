"""
Module Benchmark & Tuning cho Hệ thống AI Banking Assistant
Bao gồm:
- test_full_chunking.py: Kiểm thử toàn diện quá trình Chunking trên toàn bộ dataset
- test_full_embedding.py: Kiểm thử toàn diện quá trình sinh Embedding và đo lường throughput
- evaluation_dataset.py: Tập dữ liệu 20 câu hỏi kiểm định ground-truth chuẩn ngân hàng
- tuner.py: Thuật toán Grid Search Tuning tối ưu hóa siêu tham số (Chunking, BM25, RRF, Hybrid)
"""
