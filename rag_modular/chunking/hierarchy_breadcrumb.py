from pathlib import Path
import re

def load_document() -> List:
    path_dir=Path("dataset/retrieval/retrieval_dataset")
    documents=[]
    
    for file_path in path_dir.glob("*.md"):
        with open(file_path, "r", encoding="utf-8") as f:
            documents.append({
            "source":file_path.name,
            "content":f.read()
            })

    return documents

def hierarchy_breadcrumb_chunking(text, chunk_size=512):
    chunks=[]

    lines=text.split('\n')

    current_header={}
    current_content=[]
    chunk_id=0

    def get_breadcrumb():
        levels=sorted(current_header.keys())
        titles=[current_header[lvl] for lvl in levels]
        return "> ".join(titles)

    def save_chunk(content_lines):
        nonlocal chunk_id
        text_block="\n".join(content_lines).strip()
        if not text_block:
            return
        
        breadcrumb=get_breadcrumb()

        full_content=(
            f'{breadcrumb}\n{text_block}' if breadcrumb else text_block
        )

        chunks.append({
            "chunk_id":chunk_id,
            "breadcrumb":breadcrumb,
            "content":full_content
        })
        chunk_id+=1
    
    for line in lines:
        header_match=re.match(r"^(#{1,6})\s+(.*)", line.strip())

        if header_match:
            if current_content:
                save_chunk(current_content)
                current_content=[]
            
            level=len(header_match.group(1))
            title=header_match.group(2).strip()

            current_header={k: v for k, v in current_header.items() if k<level}
            current_header[level]=title


        else:
            if line.strip():
                temp_text="\n".join(current_content+[line])
                if len(temp_text)>chunk_size and current_content:
                    save_chunk(current_content)
                    current_content=[line]

                else:
                    current_content.append(line)
        
    if current_content:
        save_chunk(current_content)

    return chunks
