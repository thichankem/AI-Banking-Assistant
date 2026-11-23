"""Read the ZIP parts and typed cells of an Excel workbook."""

from collections import namedtuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import PurePosixPath
import posixpath
from urllib.parse import unquote, urlsplit
from zipfile import ZipFile

from lxml import etree


_MAX_PACKAGE_BYTES = 512 * 1024 * 1024
_MAX_COLUMN = 16384
_MAX_ROW = 1048576
_BUILTIN_FORMATS = {
    0: "General", 1: "0", 2: "0.00", 3: "#,##0", 4: "#,##0.00",
    5: "$#,##0", 6: "$#,##0", 7: "$#,##0.00", 8: "$#,##0.00",
    9: "0%", 10: "0.00%", 14: "m/d/yy", 15: "d-mmm-yy", 16: "d-mmm",
    17: "mmm-yy", 18: "h:mm AM/PM", 19: "h:mm:ss AM/PM", 20: "h:mm",
    21: "h:mm:ss", 22: "m/d/yy h:mm", 27: "yyyy/m/d", 28: "yyyy/m/d",
    29: "yyyy/m/d", 30: "m/d/yy", 31: "yyyy/m/d", 32: "h:mm", 33: "h:mm:ss",
    34: "yyyy/m", 35: "m/d", 36: "yyyy/m", 37: "#,##0", 38: "#,##0",
    39: "#,##0.00", 40: "#,##0.00", 45: "mm:ss", 46: "[h]:mm:ss",
    47: "mm:ss.0", 49: "@", 50: "yyyy/m/d", 51: "m/d/yy", 52: "yyyy/m",
    53: "m/d", 54: "yyyy/m", 55: "h:mm", 56: "h:mm:ss", 57: "yyyy/m", 58: "m/d",
}
_CURRENCIES = {"₫": "VND", "đ": "VND", "VND": "VND", "$": "USD",
               "USD": "USD", "€": "EUR", "EUR": "EUR", "£": "GBP", "GBP": "GBP",
               "¥": "¥", "JPY": "JPY", "CNY": "CNY", "₩": "KRW", "KRW": "KRW",
               "AUD": "AUD", "CAD": "CAD", "SGD": "SGD", "HKD": "HKD",
               "CHF": "CHF", "THB": "THB"}
_CURRENCY_SYMBOLS = "$₫€£¥₩"


class InvalidWorkbook(ValueError):
    """A workbook package cannot be read as XLSX."""


def local_name(node):
    if isinstance(node.tag, str):
        return node.tag.rsplit("}", 1)[-1]
    return ""


def children(node, name):
    return (child for child in node if local_name(child) == name)


def child(node, name):
    return next(children(node, name), None)


def attribute(node, name):
    if node is not None:
        return next((value for key, value in node.attrib.items() if key.rsplit("}", 1)[-1] == name), None)
    return None


def text_runs(node):
    if node is None:
        return ""
    return "".join(part.text or "" for part in node.iter()
                   if local_name(part) == "t" and local_name(part.getparent()) != "rPh")


def address(value):
    label = str(value or "").upper().lstrip("$")
    end = 0
    while end < len(label) and "A" <= label[end] <= "Z":
        end += 1
    letters = label[:end]
    row = label[end:].removeprefix("$")
    if not 1 <= len(letters) <= 3 or not 1 <= len(row) <= 7 or not row.isascii() or not row.isdecimal() or row.startswith("0"):
        raise ValueError("Invalid Excel cell address")
    column = 0
    for letter in letters:
        column = column * 26 + ord(letter) - 64
    row = int(row)
    if column > _MAX_COLUMN or row > _MAX_ROW:
        raise ValueError("Excel cell address is out of bounds")
    return row, column


def column_name(number):
    letters = []
    while number:
        number, letter = divmod(number - 1, 26)
        letters.append(chr(65 + letter))
    return "".join(reversed(letters))


def a1(row, column):
    return f"{column_name(column)}{row}"


class CellRange(namedtuple("_CellRange", "top left bottom right")):
    __slots__ = ()

    @classmethod
    def parse(cls, value):
        start, _, end = str(value or "").partition(":")
        top, left = address(start)
        bottom, right = address(end or start)
        if bottom < top or right < left:
            raise ValueError("Reversed Excel range")
        return cls(top, left, bottom, right)

    def contains(self, row, column):
        return self.top <= row <= self.bottom and self.left <= column <= self.right

    def overlaps(self, other):
        return (self.top <= other.bottom and other.top <= self.bottom
                and self.left <= other.right and other.left <= self.right)

    def label(self):
        start = a1(self.top, self.left)
        end = a1(self.bottom, self.right)
        if start == end:
            return start
        return f"{start}:{end}"


@dataclass(frozen=True)
class Cell:
    text: str
    kind: str
    value: object = None
    unit: str = ""
    style: int = 0
    number_format: str = ""


@dataclass(frozen=True)
class Relationship:
    kind: str
    target: str
    external: bool


class Package:
    def __init__(self, stream):
        self.archive = ZipFile(stream)
        try:
            parts = self.archive.infolist()
            self.parts = {part.filename: part for part in parts}
            if len(parts) != len(self.parts) or sum(part.file_size for part in parts) > _MAX_PACKAGE_BYTES:
                raise InvalidWorkbook("XLSX ZIP exceeds the package limit or contains duplicate parts")
            self.workbook = next((rel.target for rel in self.relationships("").values()
                                  if rel.kind == "officeDocument" and not rel.external), "")
            types = self.xml("[Content_Types].xml")
            main_type = next((node.get("ContentType") for node in children(types, "Override")
                              if node.get("PartName", "").lstrip("/") == self.workbook), "")
            if main_type != "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml":
                raise InvalidWorkbook("The package is not a supported .xlsx workbook")
        except BaseException:
            self.archive.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.archive.close()

    def xml(self, name):
        with self.archive.open(name) as stream:
            parser = etree.XMLParser(resolve_entities=False, load_dtd=False, no_network=True)
            root = etree.parse(stream, parser).getroot()
        if root.getroottree().docinfo.doctype:
            raise InvalidWorkbook("XLSX XML contains a document type declaration")
        return root

    def sheet_events(self, name):
        tags = ("row", "mergeCell", "hyperlink", "drawing", "picture", "tablePart", "pivotTablePart",
                "legacyDrawing", "legacyDrawingHF", "oleObject", "control")
        with self.archive.open(name) as stream:
            parser = etree.iterparse(stream, events=("end",), tag=tuple("{*}" + tag for tag in tags),
                                     resolve_entities=False, load_dtd=False, no_network=True)
            for _, node in parser:
                if node.getroottree().docinfo.doctype:
                    raise InvalidWorkbook("XLSX XML contains a document type declaration")
                yield local_name(node), node
                if local_name(node) == "row":
                    parent = node.getparent()
                    node.clear()
                    while node.getprevious() is not None:
                        del parent[0]

    def relationships(self, part):
        if part:
            path = PurePosixPath(part)
            rels_path = str(path.parent / "_rels" / (path.name + ".rels"))
        else:
            rels_path = "_rels/.rels"
        if rels_path not in self.parts:
            return {}
        result = {}
        for node in children(self.xml(rels_path), "Relationship"):
            target = node.get("Target", "")
            external = node.get("TargetMode", "").casefold() == "external"
            if not external:
                parsed = urlsplit(target)
                if parsed.scheme or parsed.netloc:
                    continue
                target = unquote(parsed.path)
                target = posixpath.normpath(posixpath.join(posixpath.dirname(part), target))
                target = target.lstrip("/")
                if not target or target == ".." or target.startswith("../"):
                    continue
            identifier = node.get("Id")
            if identifier:
                result[identifier] = Relationship(node.get("Type", "").rsplit("/", 1)[-1], target, external)
        return result

    def read(self, part, *, limit):
        if self.parts[part].file_size > limit:
            raise ValueError("XLSX object exceeds the input size limit")
        return self.archive.read(part)


@lru_cache(maxsize=512)
def currency_unit(value):
    text = value or ""
    parts = text.split("[$")
    if len(parts) > 1:
        remaining = [parts[0]]
        for part in parts[1:]:
            spec, _, tail = part.partition("]")
            symbol = spec.partition("-")[0]
            if symbol:
                remaining.append(symbol)
            remaining.append(tail)
        text = " ".join(remaining)
    words = []
    word = []
    for char in text:
        if char.isalpha():
            word.append(char)
        elif word:
            words.append("".join(word))
            word = []
    if word:
        words.append("".join(word))
    for word in words:
        if word in _CURRENCIES or word.upper() in _CURRENCIES:
            return _CURRENCIES.get(word) or _CURRENCIES[word.upper()]
    return next((_CURRENCIES[char] for char in text if char in _CURRENCY_SYMBOLS), "")


@lru_cache(maxsize=256)
def _number_pattern(number_format):
    pattern = []
    elapsed = False
    cursor = 0
    while cursor < len(number_format):
        char = number_format[cursor]
        if char == ";":
            break
        if char in "\\_*":
            cursor += 2
            continue
        if char in '\"[':
            closer = {"[": "]", '"': '"'}[char]
            end = number_format.find(closer, cursor + 1)
            if end < 0:
                break
            spec = number_format[cursor + 1:end].casefold()
            if char == "[" and spec and set(spec) <= set("hms"):
                elapsed = True
            cursor = end + 1
            continue
        pattern.append(char.casefold())
        cursor += 1
    return "".join(pattern), elapsed


def _decimal_text(value, places=None):
    if places is not None:
        return format(value, f".{places}f")
    return format(value, "f")


def _temporal(number, pattern, date1904, elapsed):
    if elapsed:
        return Cell(_decimal_text(number * 86400) + " s", "duration", str(number * 86400), "s")
    is_time = "h" in pattern or "s" in pattern
    is_date = "y" in pattern or "d" in pattern or (not is_time and "m" in pattern)
    if not is_date and not is_time:
        return None
    if not is_date and number < 0:
        return None
    if is_date and not date1904 and 60 <= number < 61:
        return Cell(_decimal_text(number), "error")
    epoch = datetime(1899, 12, 30)
    if date1904:
        epoch = datetime(1904, 1, 1)
    if not date1904 and 0 < number < 60:
        number += 1
    try:
        value = epoch + timedelta(microseconds=int(number * 86400000000))
    except (OverflowError, ValueError):
        return None
    if is_date and is_time:
        return Cell(value.isoformat(sep=" "), "datetime", value.isoformat())
    if is_date:
        return Cell(value.date().isoformat(), "date", value.date().isoformat())
    return Cell(value.time().isoformat(), "time", value.time().isoformat())


def parse_cell(node, shared, number_formats, date1904):
    style = int(node.get("s", "0"))
    if child(node, "f") is not None:
        return Cell("", "skipped", style=style)
    number_format = "General"
    if 0 <= style < len(number_formats):
        number_format = number_formats[style]
    value_node = child(node, "v")
    raw = None
    if value_node is not None:
        raw = value_node.text
    cell_type = node.get("t", "n")
    cell = None
    if cell_type == "inlineStr":
        raw = text_runs(child(node, "is"))
        cell = Cell(raw, "text", raw)
    elif raw is None:
        return None
    elif cell_type == "s":
        index = int(raw)
        if not 0 <= index < len(shared):
            raise InvalidWorkbook("XLSX contains an invalid shared string reference")
        cell = Cell(shared[index], "text", shared[index])
    elif cell_type == "b":
        flags = {"1": "TRUE", "0": "FALSE"}
        cell = Cell(flags[raw], "boolean", raw == "1")
    elif cell_type == "e":
        cell = Cell(raw, "error")
    elif cell_type == "d":
        kind = "date"
        if "T" in raw:
            kind = "datetime"
        cell = Cell(raw, kind, raw)
    elif cell_type != "n":
        cell = Cell(raw, "text", raw)
    else:
        try:
            number = Decimal(raw)
        except InvalidOperation:
            return Cell(raw, "error", style=style, number_format=number_format)
        if not number.is_finite() or abs(number.adjusted()) > 350:
            return Cell(raw, "error", style=style, number_format=number_format)
        cleaned, elapsed = _number_pattern(number_format)
        currency = currency_unit(number_format)
        if "%" in cleaned:
            places = 0
            for char in cleaned.partition(".")[2]:
                if char not in "0#":
                    break
                places += 1
            cell = Cell(_decimal_text(number * 100, places) + "%", "percent", raw, "%")
        elif currency:
            cell = Cell(_decimal_text(number) + " " + currency, "currency", raw, currency)
        elif (time_value := _temporal(number, cleaned, date1904, elapsed)) is not None:
            cell = time_value
        elif len(cleaned) >= 2 and set(cleaned) == {"0"} and number == number.to_integral_value():
            padded = str(abs(int(number))).zfill(len(cleaned))
            if number < 0:
                padded = "-" + padded
            cell = Cell(padded, "text", padded)
        else:
            cell = Cell(_decimal_text(number), "number", raw)
    return Cell(cell.text, cell.kind, cell.value, cell.unit, style, number_format)


def read_styles(root):
    formats = dict(_BUILTIN_FORMATS)
    definitions = child(root, "numFmts")
    if definitions is not None:
        formats.update((int(node.get("numFmtId")), node.get("formatCode", "General"))
                       for node in children(definitions, "numFmt"))
    xfs = child(root, "cellXfs")
    if xfs is None:
        return [], set()
    values = list(children(xfs, "xf"))
    fonts = child(root, "fonts")
    bold_fonts = set()
    if fonts is not None:
        bold_fonts = {idx for idx, node in enumerate(children(fonts, "font"))
                      if (flag := child(node, "b")) is not None and flag.get("val") not in {"0", "false"}}
    return [formats.get(int(node.get("numFmtId", "0")), "General") for node in values], {
        idx for idx, node in enumerate(values) if int(node.get("fontId", "0")) in bold_fonts}

