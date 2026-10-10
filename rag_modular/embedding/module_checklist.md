# Embedding Checklist

## 1. Mô hình Embedding Nơ-ron Thực tế (Production Neural Models)
- [x] **Enterprise Cloud Vietnamese_Embedding** (`CloudEmbedding` - 1024d, 8K context, SOTA Vietnamese Retrieval)
- [x] **Enterprise Cloud Multilingual-E5-Large** (`CloudEmbedding` - 1024d, Microsoft Multilingual Transformer)
- [x] **OpenAI text-embedding-3-small** (`OpenAIEmbedding` - 1536d, Cloud API)
- [x] **OpenAI text-embedding-3-large** (`OpenAIEmbedding` - 3072d, High Precision Cloud API)
- [x] **BGE-M3 / BKAI-BiEncoder / Vietnamese-SBERT** (`SentenceTransformerEmbedding` - On-Premise / Local PyTorch)

## 2. Mô hình Baseline & Testing Fallback
- [x] **Deterministic Dense Projection** (`DeterministicDenseEmbedding` - 128d/1024d, Offline Mock & Unit Testing)
- [x] **TF-IDF / Hashing Sparse Vector** (`HashingTFIDFEmbedding` - 256d Sparse Baseline)

## 3. Trạng thái Deprecated (Loại bỏ)
- [~] ~~**Banking Concept Semantic Domain-Guided Embedding**~~ (ĐÃ KHAI TỬ & THAY THẾ: Phương pháp đếm từ khóa thô sơ cũ đã được thay thế 100% bằng `CloudEmbedding`)

## 4. Nghiên cứu Nâng cao trong tương lai
- [ ] Cohere Embed v3 (Multilingual Compression API)
- [ ] Matryoshka Representation Learning (MRL Adaptive Dimensions 256/512/1024)
