"""
Module Tu Dong Phan Doan Tai Lieu (Auto Chunking Node with Deduplication).
==========================================================================
Dac diem:
1. Doc toan bo file .docx trong chunking/input/.
2. Tinh toan SHA-256 hash cua tung file.
3. Kiem tra StateTracker: File nao da chay va khong doi thi bo qua (SKIP).
4. Chi phan doan file moi hoac file bi sua doi.
5. Dong bo dau ra: {stem}.json, chunks_per_file/{stem}.json, all_chunks.json.

NGUYEN TAC BAT BUOC:
1. KHONG DUNG __INIT__.PY.
2. KHONG DUNG THU VIEN TYPING.
3. ZERO TERNARY: if/else tuong minh.
4. 0% REGEX LAW.
"""

from __future__ import annotations
import sys
import json
import time
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_banking.chunking.unified_chunk.layout_aware_chunker import LayoutAwareChunker
from agent_banking.state_tracker import StateTracker


def run_auto_chunking(
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    force: bool = False,
    max_files: int | None = None
) -> list[dict]:
    """
    Tu dong phan doan tu input_dir sang output_dir kem co che chong trung lap.
    """
    if input_dir is not None:
        in_path = Path(input_dir)
    else:
        in_path = PROJECT_ROOT / "agent_banking" / "chunking" / "input"

    if output_dir is not None:
        out_path = Path(output_dir)
    else:
        out_path = PROJECT_ROOT / "agent_banking" / "chunking" / "output"

    in_path.mkdir(parents=True, exist_ok=True)
    out_path.mkdir(parents=True, exist_ok=True)
    chunks_per_file_dir = out_path / "chunks_per_file"
    chunks_per_file_dir.mkdir(parents=True, exist_ok=True)

    state_file = out_path / ".state_tracker.json"
    tracker = StateTracker(state_file)
    if force:
        tracker.clear()

    # Quet toan bo file .docx
    files = []
    for ext in (".docx", ".DOCX"):
        files.extend(list(in_path.rglob(f"*{ext}")))
    files = sorted(list(set(files)))

    if max_files is not None:
        if max_files > 0:
            files = files[:max_files]

    print(f"\n{'='*75}")
    print(f"▶ [MODULE 1: CHUNKING] Input: {in_path} -> Output: {out_path}")
    print(f"[*] Tong so file tim thay: {len(files)} file")
    print(f"{'='*75}")

    if not files:
        print("⚠️ Khong co file .docx nao trong input/. Vui long sao chep file .docx vao day.")
        return []

    # Kiem tra nhanh neu toan bo files da chay roi thi bo qua toan bo node
    all_chunks_file = out_path / "all_chunks.json"
    if not force and all_chunks_file.exists():
        files_hash_map = {}
        for fpath in files:
            rel_name = str(fpath.relative_to(in_path)).replace("\\", "/")
            files_hash_map[rel_name] = tracker.compute_file_hash(fpath)
        if tracker.are_all_items_processed(files_hash_map):
            try:
                with open(all_chunks_file, "r", encoding="utf-8") as fp:
                    cached_all = json.load(fp)
                if cached_all:
                    print(f"  ⏭️ [SKIP ALL] Tat ca {len(files)} file docx da duoc phan doan truoc do. Bo qua toan bo trong 0.0s!")
                    print(f"✅ [Chunking Hoan Tat] Tong: {len(cached_all)} chunks (Bo qua: {len(files)}, Moi: 0) trong 0.0s")
                    return cached_all
            except Exception:
                pass

    chunker = LayoutAwareChunker()
    all_chunks: list[dict] = []
    skipped_count = 0
    new_processed_count = 0
    start_total = time.time()
    manifest_records: list[dict] = []

    for idx, fpath in enumerate(files, start=1):
        rel_name = str(fpath.relative_to(in_path)).replace("\\", "/")
        stem = fpath.stem
        f_hash = tracker.compute_file_hash(fpath)
        doc_json_file = out_path / f"{stem}.json"

        # Kiem tra xem file da chay truoc do chua
        is_already_done = tracker.is_processed(rel_name, f_hash)
        if is_already_done:
            if doc_json_file.exists():
                try:
                    with open(doc_json_file, "r", encoding="utf-8") as fp:
                        cached_chunks = json.load(fp)
                    all_chunks.extend(cached_chunks)
                    skipped_count += 1
                    manifest_records.append({
                        "file_name": fpath.name,
                        "rel_path": rel_name,
                        "status": "SKIPPED_ALREADY_PROCESSED",
                        "chunk_count": len(cached_chunks)
                    })
                    print(f"  [{idx:02d}/{len(files)}] [SKIP] {fpath.name[:45]:<45} -> {len(cached_chunks)} chunks (Da xu ly)")
                    continue
                except Exception:
                    pass

        # Xu ly file moi hoac file bi sua doi
        t0 = time.time()
        try:
            file_chunks = chunker.chunk_docx(fpath)
            dt = round(time.time() - t0, 2)

            with open(doc_json_file, "w", encoding="utf-8") as fp:
                json.dump(file_chunks, fp, ensure_ascii=False, indent=2)

            single_p = chunks_per_file_dir / f"{stem}.json"
            with open(single_p, "w", encoding="utf-8") as fp:
                json.dump(file_chunks, fp, ensure_ascii=False, indent=2)

            tracker.record_processed(rel_name, f_hash, {
                "chunk_count": len(file_chunks),
                "duration_sec": dt,
                "file_size": fpath.stat().st_size
            })

            all_chunks.extend(file_chunks)
            new_processed_count += 1
            manifest_records.append({
                "file_name": fpath.name,
                "rel_path": rel_name,
                "status": "SUCCESS",
                "chunk_count": len(file_chunks),
                "duration_sec": dt
            })
            print(f"  [{idx:02d}/{len(files)}] [NEW]  {fpath.name[:45]:<45} -> {len(file_chunks)} chunks ({dt}s)")
        except Exception as e:
            print(f"  [{idx:02d}/{len(files)}] [ERR]  {fpath.name}: {e}")
            manifest_records.append({
                "file_name": fpath.name,
                "rel_path": rel_name,
                "status": f"FAILED: {e}",
                "chunk_count": 0
            })

    tracker.save()
    total_elapsed = round(time.time() - start_total, 2)

    # Ghi file tong hop all_chunks.json
    all_chunks_file = out_path / "all_chunks.json"
    with open(all_chunks_file, "w", encoding="utf-8") as fp:
        json.dump(all_chunks, fp, ensure_ascii=False, indent=2)

    # Ghi manifest
    manifest_file = out_path / "chunking_manifest.json"
    summary_data = {
        "module": "chunking",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_files": len(files),
        "skipped_files": skipped_count,
        "new_files": new_processed_count,
        "total_chunks": len(all_chunks),
        "total_elapsed_sec": total_elapsed,
        "records": manifest_records
    }
    with open(manifest_file, "w", encoding="utf-8") as fp:
        json.dump(summary_data, fp, ensure_ascii=False, indent=2)

    print(f"✅ [Chunking Hoan Tat] Tong: {len(all_chunks)} chunks (Bo qua: {skipped_count}, Moi: {new_processed_count}) trong {total_elapsed}s")
    return all_chunks


if __name__ == "__main__":
    run_auto_chunking()
