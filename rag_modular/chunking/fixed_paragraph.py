from pathlib import Path

def load_document():
    dataset_dir=Path("dataset/retrieval/retrieval_dataset")
    documents=[]

    for file_path in dataset_dir.glob("*.md"):
        with open(file_path, "r", encoding="utf-8") as f:
            documents.append({
                "source":file_path.name,
                "content":f.read()
            })
    return documents

def fixed_paragraph_chunking(text, max_token_chunk=512):
    chunks=[]

    paragraphs=[p.strip() for p in text.split("\n\n") if p.strip() != ""]

    current_chunk=""

    chunk_id=0

    for p in paragraphs:
        if(len(current_chunk)+len(p)<max_token_chunk):
            current_chunk += " " + p
        else:
            if current_chunk!="":
                chunks.append({
                    "chunk_id":chunk_id,
                    "content":current_chunk.strip()
                })
                chunk_id+=1
            current_chunk=p

    if current_chunk!="":
            chunks.append({
                "chunk_id":chunk_id,
                "content":current_chunk
            })

    return chunks


