"""Language normalization for search keys, separate from displayed or cited text."""

from unicodedata import category, normalize


def search_key(value: str) -> str:
    decomposed = normalize("NFKD", value.casefold()).replace("đ", "d")
    plain = "".join(character for character in decomposed if category(character) != "Mn")
    return " ".join("".join(character if character.isalnum() else " " for character in plain).split())
