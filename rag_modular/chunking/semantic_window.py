from pathlib import Path
import numpy as np
import requests
import os
import re
from dotenv import load_dotenv

load_dotenv()
API_KEY=os.getenv("API_KEY")
EMBEDDING_URL=os.getenv("EMBEDDING_URL")

def get_embeddings(texts, batch_size=16):
    if not texts:
        return np.empty((0, 1024))

    headers={
        "Content-type":"application/json",
        "Authorization":f"Bearer {API_KEY}"
    }
    
    embeddings=[]

    for i in range(0, len(texts), batch_size):
        batch=texts[i:i+batch_size]
        payload={
            "model":"Vietnamese_Embedding",
            "input":batch,
            "dimensions":1024,
            "encoding_format":"float",
            "input_text_truncate":"none",
            "input_type":"passage"
        }

        response=requests.post(EMBEDDING_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()

        data=response.json().get("data", [])
        sorted_data=sorted(data, key=lambda x:x["index"])

        batch_embeddings=[item['embedding'] for item in sorted_data]

        embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)




def load_document():
    documents=[]
    path_dir=Path("dataset/retrieval/retrieval_dataset")
    for file_path in path_dir.glob("*.md"):
        with open(file_path, "r", encoding="utf-8") as f:
            documents.append({
                "source":file_path.name,
                "content":f.read()
            })

    return documents


def split_sentences(text):
    sentences=re.split(r'(?<=[.?!])\s+|\n', text)
    return [s.strip() for s in sentences if s.strip()]

def compute_cosine_distance(vec_a, vec_b):
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    similarity = np.dot(vec_a, vec_b) / (norm_a * norm_b)
    
    return float(1.0 - similarity)


def semantic_window_chunking(
    text,
    threshold_percentile=80.0,
    buffer_size=1,
    chunk_size=512
):
    sentences=split_sentences(text)
    if not sentences:
        return []
    if len(sentences)==1:
        return [{"chunk_id":0, "content":sentences[0], "shift_score":0.0}]

    buffered_sentences=[]

    for i in range(len(sentences)):
        start=max(0, i-buffer_size)
        end=min(len(sentences), i+buffer_size+1)
        buffered_sentences.append(" ".join(sentences[start:end]))

    embeddings=get_embeddings(buffered_sentences, batch_size=16)

    distances=[]

    for i in range(len(embeddings)-1):
            dist=compute_cosine_distance(embeddings[i], embeddings[i+1])
            distances.append(dist)

    breakpoint_threshold=np.percentile(distances, threshold_percentile)

    chunks=[]

    current_chunk=[sentences[0]]
    chunk_id=0

    for i in range(len(distances)):
        dist=distances[i]
        next_sentence=sentences[i+1]

        current_len=sum(len(s) for s in current_chunk)
        is_semantic_shift=dist>=breakpoint_threshold
        is_length_overflow=(current_len+len(next_sentence))>chunk_size

        if is_semantic_shift or is_length_overflow:
            chunks.append({
                "chunk_id":chunk_id,
                "content":" ".join(current_chunk),
                "shift_score":round(dist, 4)
            })
            current_chunk=[next_sentence]
            chunk_id+=1

        else:
            current_chunk.append(next_sentence)

    if current_chunk:
        chunks.append({
            "chunk_id":chunk_id,
            "content":" ".join(current_chunk),
            "shift_score":0
        })

    return chunks
    
