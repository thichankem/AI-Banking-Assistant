# Retrieval & Fusion Checklist

- [ ] Dense Only Retrieval (K-NN Cosine)
- [ ] Sparse Only Retrieval (BM25 Lexical)
- [ ] Weighted Linear Hybrid ($Score = \alpha \cdot S_{dense} + (1-\alpha) \cdot S_{sparse}$)
- [ ] Reciprocal Rank Fusion (RRF, $k=60$)
- [ ] Reciprocal Rank Fusion (RRF, $k=20$)
- [ ] Threshold-Filtered Hybrid (Score / Distance Gating)
- [ ] Relative Score Fusion (RSF / Min-Max Normalization)
- [ ] Auto-Routing Retrieval (Intent-based Strategy Selection)
- [ ] Contextual Compression Retrieval
