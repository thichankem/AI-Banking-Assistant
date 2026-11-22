"""Backfill cropped visual assets for published documents and sync Qdrant image payloads.

RULES:
- 0% REGEX LAW: Strictly string methods only (no re).
- ZERO TERNARY LAW: Explicit if/else statements only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_banking.chunking.publication import _load_snapshot, publication_asset_path, use_relative_asset_paths
from agent_banking.chunking.publication_io import atomic_bytes, atomic_json, file_hash, publication_lock
from agent_banking.chunking.universal.extraction.pdf.visual_cropper import (
    collect_page_step_numbers,
    extract_page_visual_crops,
    match_chunk_visual_assets,
)
from agent_banking.vector_db.config.settings import VectorDBConfig


def _preserve_legacy_multimodal_images(images):
    kept = []
    if not isinstance(images, list):
        return kept
    for item in images:
        if isinstance(item, dict) and item.get("status") in ("ocr_transcribed", "text_context_supported"):
            kept.append(dict(item))
    return kept


def _copy_asset_to_shared_dir(src_path, shared_dir):
    if not src_path.is_file():
        return False
    shared_dir.mkdir(parents=True, exist_ok=True)
    dst_path = shared_dir / src_path.name
    if dst_path.is_file() and dst_path.stat().st_size == src_path.stat().st_size:
        return False
    tmp_path = shared_dir / (src_path.name + ".tmp")
    shutil.copyfile(src_path, tmp_path)
    tmp_path.replace(dst_path)
    return True


def _backfill_single_publication(pub_dir, shared_assets_dir):
    manifest_path = pub_dir / "chunking_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("records") or []
    if len(records) != 1:
        raise ValueError("Expected single-document publication: " + str(pub_dir))
    rec = records[0]
    if rec.get("status") == "FAILED":
        return None

    rel_path = str(rec.get("rel_path") or "")
    chunks_path = pub_dir / rec["output_file"]
    artifact_path = pub_dir / rec["artifact_file"]
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    doc = artifact["document"]
    assets_map = doc.setdefault("assets", {})
    pdf_crops_created = 0
    region_counts = {}

    if rel_path.lower().endswith(".pdf"):
        # Remove previously generated cropped_region entries so re-running stays clean.
        stale_keys = []
        for asset_key, asset_val in assets_map.items():
            if isinstance(asset_val, dict) and asset_val.get("kind") == "cropped_region":
                stale_keys.append(asset_key)
        for stale_key in stale_keys:
            assets_map.pop(stale_key, None)

        page_assets = []
        for asset_val in list(assets_map.values()):
            if isinstance(asset_val, dict) and asset_val.get("kind") == "page":
                page_assets.append(asset_val)
        page_assets.sort(key=lambda item: int(item.get("page_index", 0)))

        for page_asset in page_assets:
            page_file = publication_asset_path(pub_dir, page_asset["path"])
            if not page_file.is_file():
                continue
            pixels = page_file.read_bytes()
            page_index = int(page_asset.get("page_index", 0))
            page_sha = str(page_asset.get("sha256") or "")
            crops = extract_page_visual_crops(
                pixels, page_index, page_sha, pub_dir / "assets", source_name=rel_path
            )
            for crop in crops:
                crop_sha = crop["sha256"]
                assets_map[crop_sha] = crop
                pdf_crops_created += 1
                rtype = str(crop.get("region_type") or "figure")
                region_counts[rtype] = region_counts.get(rtype, 0) + 1

    page_step_numbers = collect_page_step_numbers(doc.get("blocks", []))
    chunks_with_images = 0
    total_chunk_image_links = 0
    copied_assets = 0

    for chunk in chunks:
        chunk["images"] = _preserve_legacy_multimodal_images(chunk.get("images"))
        matched = match_chunk_visual_assets(chunk, assets_map, page_step_numbers=page_step_numbers)
        chunk["images"] = matched
        if matched:
            chunks_with_images += 1
            total_chunk_image_links += len(matched)

    use_relative_asset_paths(pub_dir, chunks, artifact)

    for chunk in chunks:
        for img in chunk.get("images", []):
            if not isinstance(img, dict) or not img.get("path"):
                continue
            src_asset = publication_asset_path(pub_dir, img["path"])
            if _copy_asset_to_shared_dir(src_asset, shared_assets_dir):
                copied_assets += 1

    with publication_lock(pub_dir):
        atomic_json(chunks_path, chunks)
        atomic_json(artifact_path, artifact)

        rec["output_sha256"] = file_hash(chunks_path)
        rec["artifact_sha256"] = file_hash(artifact_path)
        rec["chunk_count"] = len(chunks)

        aggregate_bytes = json.dumps(chunks, ensure_ascii=False, indent=2).encode("utf-8")
        aggregate_sha = hashlib.sha256(aggregate_bytes).hexdigest()
        snapshot_rel = manifest.get("active_aggregate_file")
        if snapshot_rel:
            atomic_bytes(pub_dir / snapshot_rel, aggregate_bytes)
        manifest["aggregate_sha256"] = aggregate_sha

        all_chunks_path = pub_dir / "all_chunks.json"
        if all_chunks_path.is_file():
            atomic_bytes(all_chunks_path, aggregate_bytes)

        doc_alias_path = pub_dir / (str(doc.get("document_id") or "") + ".json")
        if doc_alias_path.is_file():
            atomic_json(doc_alias_path, chunks)

        atomic_json(manifest_path, manifest)
        last_run_path = pub_dir / "chunking_last_run.json"
        if last_run_path.is_file():
            atomic_json(last_run_path, manifest)

        tracker_path = pub_dir / ".state_tracker.json"
        if tracker_path.is_file():
            tracker = json.loads(tracker_path.read_text(encoding="utf-8"))
            tracker_recs = tracker.get("records")
            if isinstance(tracker_recs, dict) and rel_path in tracker_recs:
                tracker_recs[rel_path]["output_sha256"] = rec["output_sha256"]
                tracker_recs[rel_path]["artifact_sha256"] = rec["artifact_sha256"]
                tracker_recs[rel_path]["chunk_count"] = len(chunks)
                atomic_json(tracker_path, tracker)

    verified_chunks = _load_snapshot(pub_dir, manifest, require_ready=True)
    if len(verified_chunks) != len(chunks):
        raise ValueError("Integrity check chunk count mismatch for " + str(pub_dir))

    return {
        "rel_path": rel_path,
        "chunks": chunks,
        "total_chunks": len(chunks),
        "chunks_with_images": chunks_with_images,
        "total_chunk_image_links": total_chunk_image_links,
        "pdf_crops_created": pdf_crops_created,
        "region_counts": region_counts,
        "copied_assets": copied_assets,
    }


def _sync_qdrant_images(index_dir, collection_name, chunk_image_map, chunk_id_fallback):
    from qdrant_client import QdrantClient

    client = QdrantClient(path=str(index_dir))
    updated_points = 0
    non_empty_points = 0
    total_points = 0
    try:
        collections = [c.name for c in client.get_collections().collections]
        if collection_name not in collections:
            raise ValueError("Collection not found in Qdrant: " + str(collection_name))

        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=collection_name,
                limit=256,
                offset=offset,
                with_payload=["chunk_id", "source", "images"],
                with_vectors=False,
            )
            for point in points:
                total_points += 1
                payload = point.payload or {}
                chunk_id = str(payload.get("chunk_id") or "")
                source = str(payload.get("source") or "").replace("\\", "/")
                target_images = chunk_image_map.get((source, chunk_id))
                if target_images is None:
                    target_images = chunk_id_fallback.get(chunk_id, [])
                if target_images:
                    non_empty_points += 1
                current_images = payload.get("images")
                if not isinstance(current_images, list):
                    current_images = []
                if current_images != target_images:
                    client.set_payload(
                        collection_name=collection_name,
                        payload={"images": target_images},
                        points=[point.id],
                        wait=False,
                    )
                    updated_points += 1
            if offset is None:
                break
    finally:
        client.close()

    return {
        "total_points": total_points,
        "non_empty_points": non_empty_points,
        "updated_points": updated_points,
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--publications",
        type=Path,
        default=PROJECT_ROOT / "agent_banking" / "outputs" / "tuning_20261117" / "knowledge",
    )
    parser.add_argument(
        "--index",
        type=Path,
        nargs="*",
        default=[PROJECT_ROOT / "agent_banking" / "vector_db" / "qdrant_db"],
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=PROJECT_ROOT / "agent_banking" / "chunking" / "output" / "assets",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=VectorDBConfig.QDRANT_COLLECTION,
    )
    args = parser.parse_args()

    manifests = sorted(args.publications.glob("*/chunking_manifest.json"))
    if not manifests:
        raise FileNotFoundError("No publication manifests found under " + str(args.publications))

    chunk_image_map = {}
    chunk_id_fallback = {}
    ambiguous_ids = set()

    total_docs = 0
    total_chunks = 0
    total_chunks_with_images = 0
    total_pdf_crops = 0
    total_copied = 0
    region_totals = {}

    for manifest_path in manifests:
        pub_dir = manifest_path.parent
        outcome = _backfill_single_publication(pub_dir, args.assets_dir)
        if outcome is None:
            continue
        total_docs += 1
        total_chunks += outcome["total_chunks"]
        total_chunks_with_images += outcome["chunks_with_images"]
        total_pdf_crops += outcome["pdf_crops_created"]
        total_copied += outcome["copied_assets"]
        for rtype, cnt in outcome["region_counts"].items():
            region_totals[rtype] = region_totals.get(rtype, 0) + cnt

        for chunk in outcome["chunks"]:
            src = str(chunk.get("source") or "").replace("\\", "/")
            cid = str(chunk.get("chunk_id") or "")
            imgs = chunk.get("images") or []
            chunk_image_map[(src, cid)] = imgs
            if cid in ambiguous_ids:
                continue
            if cid in chunk_id_fallback:
                ambiguous_ids.add(cid)
                chunk_id_fallback.pop(cid, None)
            else:
                chunk_id_fallback[cid] = imgs

    print(
        f"Publications verified: {total_docs}/{len(manifests)}, "
        f"chunks: {total_chunks}, chunks_with_images: {total_chunks_with_images}, "
        f"pdf_crops: {total_pdf_crops} {region_totals}, shared_assets_copied: {total_copied}"
    )

    for index_dir in args.index:
        if not index_dir.exists():
            print("Skipped missing index:", index_dir)
            continue
        sync_stats = _sync_qdrant_images(
            index_dir, args.collection, chunk_image_map, chunk_id_fallback
        )
        print(f"Synced Qdrant [{index_dir}]: {sync_stats}")


if __name__ == "__main__":
    main()
