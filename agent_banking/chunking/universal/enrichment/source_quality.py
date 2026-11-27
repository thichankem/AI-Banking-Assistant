"""Project source extraction uncertainty onto the passages it affects."""


def annotate_page_quality(chunks, document):
    uncertain = {review["page_index"] for review in document.get("page_reviews", [])
                 if review.get("incomplete") or review.get("content") == "unavailable"}
    uncertain_images = {review.get("asset_sha256") for review in document.get("docx_image_reviews", [])
                        if review.get("incomplete")}
    result = []
    for chunk in chunks:
        metadata = chunk.get("business_metadata", {})
        locations = metadata.get("provenance", {}).get("locations", [])
        pages = {location.get("page_index") for location in locations}
        pages.update(number - 1 for number in metadata.get("citation", {}).get("page_numbers", ()) if type(number) is int)
        affected = sorted(page + 1 for page in uncertain.intersection(pages))
        image_affected = any(location.get("image_asset_sha256") in uncertain_images
                             for location in locations if location.get("image_asset_sha256"))
        if affected or image_affected:
            quality = {**metadata.get("quality", {})}
            if affected:
                quality["ocr_uncertain_pages"] = affected
            if image_affected:
                quality["ocr_uncertain_images"] = True
            chunk = {**chunk, "business_metadata": {**metadata, "quality": quality}}
        result.append(chunk)
    return result
