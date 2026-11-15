"""Verify per-occurrence source preservation before publishing chunks."""
from ..enrichment.image_context import IMAGE_POLICY, excluded_image
from .source_spans import is_furniture_block, normalize


def validate_coverage(planner):
    validate_exact_locations(planner)
    ids = {chunk["chunk_id"] for chunk in planner.chunks}
    if len(ids) != len(planner.chunks):
        raise AssertionError("Duplicate chunk IDs")
    table_parts = {}
    images = set()
    image_occurrences = set()
    for chunk in planner.chunks:
        if chunk["parent_id"] not in planner.parents:
            raise AssertionError("Dangling parent ID")
        provenance = chunk["business_metadata"]["provenance"]
        if any(planner.block_map[identifier]["section_id"] != chunk["parent_id"] for identifier in provenance["block_ids"]):
            raise AssertionError("Chunk contains source blocks from another section")
        if chunk["table_data"]:
            data = chunk["table_data"]
            table_parts.setdefault(data["table_id"], []).append(data)
        for image in chunk["images"]:
            images.add(image["sha256"])
            image_block_id = provenance["image_block_id"]
            block = planner.block_map[image_block_id]
            if block["type"] != "image" or block["image"]["sha256"] != image["sha256"]:
                raise RuntimeError("Image occurrence does not match its source asset")
            image_occurrences.add(image_block_id)
    table_values = {}
    for table_id, pieces in table_parts.items():
        values = set()
        for data in pieces:
            values.update(value for row in data["rows"] for value in row)
            values.update(data.get("headers", []))
            values.update(value for row in data.get("header_grid", []) for value in row)
        table_values[table_id] = {normalize(value) for value in values}
    skipped = {block["id"]: block for block in planner.ir["blocks"]
               if block["type"] == "image" and (excluded_image(block) or is_furniture_block(block))}
    excluded_units = []
    missing = []
    for unit in planner.ir["units"]:
        block = planner.block_map[unit["block_id"]]
        if is_furniture_block(block):
            excluded_units.append({"unit_id": unit["id"], "block_id": block["id"],
                                   "reason": "document_furniture_header_footer"})
            continue
        if block["type"] == "image":
            if block["id"] in skipped:
                excluded_units.append({"unit_id": unit["id"], "block_id": block["id"],
                                       "reason": block["image"]["status"]})
                continue
            if normalize(unit["text"]) not in normalize(block["text"]):
                missing.append(unit["id"])
        elif block["type"] != "table":
            # Exact source spans have already proved reconstruction per
            # occurrence. A global substring search is order-dependent and
            # can both hide duplicate loss and falsely fail split blocks.
            if normalize(unit["text"]) not in normalize(block["text"]):
                missing.append(unit["id"])
        elif "row" in unit and "column" in unit:
            row, column = unit["row"], unit["column"]
            grid = block["table_data"]["grid"]
            if not (0 <= row < len(grid) and 0 <= column < len(grid[row]) and grid[row][column] == unit["text"]):
                missing.append(unit["id"])
        else:
            values = table_values.get(block["id"], set())
            expected = normalize(unit["text"])
            if expected not in values and not any(expected in value for value in values):
                missing.append(unit["id"])
    source_image_ids = {block["id"] for block in planner.ir["blocks"] if block["type"] == "image"}
    expected_occurrences = source_image_ids - set(skipped)
    expected_assets = {planner.block_map[identifier]["image"]["sha256"] for identifier in expected_occurrences}
    missing_images = expected_assets - images
    missing_occurrences = expected_occurrences - image_occurrences
    if missing or missing_images or missing_occurrences:
        raise RuntimeError("Source coverage failed: " + str(len(missing)) + " text units / " + str(len(missing_images)) + " assets / " + str(len(missing_occurrences)) + " image occurrences")
    return {"source_units": len(planner.ir["units"]), "covered_units": len(planner.ir["units"]) - len(excluded_units),
            "accounted_units": len(planner.ir["units"]), "policy_excluded_units": excluded_units,
            "image_policy": planner.ir.get("image_policy", IMAGE_POLICY), "preserved_assets": len(planner.ir["assets"]),
            "source_image_occurrences": len(source_image_ids),
            "skipped_image_occurrences": [{"block_id": identifier, "sha256": block["image"]["sha256"],
                                            "reason": block["image"].get("status", "document_furniture_header_footer")} for identifier, block in skipped.items()],
            "missing_units": [], "linked_assets": len(images), "missing_assets": [],
            "linked_image_occurrences": len(image_occurrences), "missing_image_occurrences": [],
            "oversized_passages": 0,
            "source_preserved_over_budget": sum("passage_content" in c["business_metadata"]["retrieval"] for c in planner.chunks),
            "foreign_elements": sum(element["type"] == "foreign_language"
                                    for chunk in planner.chunks for element in chunk["business_metadata"].get("elements", [])),
            "status": "verified_source_reconstruction",
            "assurance": "IR_to_chunks_exact; visual_accuracy_requires_external_ground_truth"}

def validate_exact_locations(planner):
    """Validate occurrences and coordinates, not a global substring search.

    Identical text elsewhere must never conceal a missing paragraph or cell.
    This proves preservation of the IR, not correctness of visual extraction.
    """
    spans = {}
    tables = {}
    for chunk in planner.chunks:
        for element in chunk["business_metadata"].get("elements", []):
            identifier = element["block_id"]
            if element["original_type"] == "table_cell":
                table = planner.block_map[identifier]["table_data"]
                location = element["location"]
                cell = table["grid"][location["row"]][location["column"]]
                begin, end = element["source_span"]
                if not 0 <= begin < end <= len(cell) or cell[begin:end] != element["text"]:
                    raise RuntimeError("Attached table element does not match its source cell")
                continue
            source = planner.block_map[identifier]["text"]
            begin, end = element["source_span"]
            if not 0 <= begin < end <= len(source) or source[begin:end] != element["text"]:
                raise RuntimeError("Attached source element does not match its origin")
            spans.setdefault(identifier, []).append((begin, end))
        for span in chunk["business_metadata"]["provenance"].get("source_spans", []):
            identifier = span["block_id"]
            source = planner.block_map[identifier]["text"]
            begin, end = span["start"], span["end"]
            chunk_start, chunk_end = span["chunk_start"], span["chunk_end"]
            if (not 0 <= begin < end <= len(source) or not 0 <= chunk_start < chunk_end <= len(chunk["content"])
                    or source[begin:end] != chunk["content"][chunk_start:chunk_end]):
                raise RuntimeError("Source span mismatch")
            spans.setdefault(identifier, []).append((begin, end))
        if chunk["table_data"]:
            table = chunk["table_data"]
            tables.setdefault(table["table_id"], []).append(table)
    for block in planner.ir["blocks"]:
        if is_furniture_block(block):
            continue
        text_block = block["type"] in ("heading", "paragraph", "caption")
        if block["type"] == "image" and not excluded_image(block):
            text_block = True
        if text_block and block["text"]:
            cursor = 0
            for begin, end in sorted(spans.get(block["id"], [])):
                if begin != cursor:
                    raise RuntimeError("Missing or overlapping text occurrence: " + block["id"])
                cursor = end
            if cursor != len(block["text"]):
                raise RuntimeError("Missing source text occurrence: " + block["id"])
        if block["type"] != "table":
            continue
        original = block["table_data"]
        coordinates = {}
        parts = tables.get(block["id"], [])
        if not parts:
            raise RuntimeError("Missing table occurrence")
        for part in parts:
            if part.get("header_grid") != original["grid"][:original["header_row_count"]]:
                raise RuntimeError("Table header grid changed")
            if part["row_end"] != part["row_start"] + len(part["rows"]):
                raise RuntimeError("Invalid table row range")
            for row_index, row in enumerate(part["rows"], part["row_start"]):
                if row_index in coordinates:
                    raise RuntimeError("Overlapping table row occurrence")
                coordinates[row_index] = row
        expected = dict(enumerate(original["rows"], original["header_row_count"]))
        if coordinates != expected:
            raise RuntimeError("Missing or changed table row occurrence")
