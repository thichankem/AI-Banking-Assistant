"""Language normalization for search keys, separate from displayed or cited text."""

from unicodedata import category, normalize

from agent_banking.core.config.settings import CoreConfig

_PROCEDURE_PHRASES = tuple(CoreConfig.CORE_SEARCH_TEXT["procedure_query_phrases"])


def search_key(value: str) -> str:
    decomposed = normalize("NFKD", value.casefold()).replace("đ", "d")
    chars = []
    for character in decomposed:
        if category(character) != "Mn":
            if character.isalnum():
                chars.append(character)
            else:
                chars.append(" ")
    return " ".join("".join(chars).split())


def is_procedure_query(value: str) -> bool:
    key = " " + search_key(value) + " "
    return any(phrase in key for phrase in _PROCEDURE_PHRASES)
