from pathlib import Path

# lay dataset de tien hanh chunking
def load_document():

    dataset_dir=Path("dataset/retrieval/retrieval_dataset")
    documents=[]

    for file_path in dataset_dir.glob("*.md"):
        with open(file_path, "r", encoding="utf-8") as f:
            documents.append({
                "source": file_path.name, 
                "content": f.read()
            })

    return documents

# bat dau chunking theo fixed_window

def fixed_window_chunking(text, chunk_size=512):

    chunks=[]

    for i in range(0, len(text), chunk_size):
        chunk=text[i: i+chunk_size]
        chunks.append({
            "chunk_id": i/chunk_size, 
            "content": chunk
        })

    return chunks







    



