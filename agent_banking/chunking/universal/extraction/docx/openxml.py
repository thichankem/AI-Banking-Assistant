"""Read visible WordprocessingML markup and preserve native text and math."""
from copy import deepcopy
from html import escape

from docx.oxml.ns import nsmap, qn
from lxml import etree


MC = "{http://schemas.openxmlformats.org/markup-compatibility/2006}"
SUPPORTED_CHOICE_NAMESPACES = {nsmap[prefix] for prefix in ("w", "a", "c", "dgm", "m", "pic", "wp")}
SUPPORTED_CHOICE_NAMESPACES.add("urn:schemas-microsoft-com:vml")
NONCONTENT = {"del", "moveFrom", "sectPr", "sdtPr", "sdtEndPr", "pPr", "rPr", "tblPr", "tblGrid", "trPr", "tcPr"}
EMBEDDED_CONTENT = {"altChunk", "object", "contentPart"}
TABLE_WRAPPERS = {"sdt", "sdtContent", "customXml", "ins", "moveTo"}
DELETED_ROW = "./" + qn("w:trPr") + "/" + qn("w:del")
TEX_CHARACTERS = str.maketrans({
    "\\": r"\backslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "%": r"\%",
    "#": r"\#", "&": r"\&", "_": r"\_", "^": r"\hat{}", "~": r"\sim{}",
    "×": r"\times{}", "÷": r"\div{}", "≤": r"\le{}", "≥": r"\ge{}",
    "≠": r"\ne{}", "±": r"\pm{}", "−": "-", "∞": r"\infty{}",
})


def local_name(node):
    if not isinstance(node.tag, str):
        return ""
    return node.tag.rsplit("}", 1)[-1]


def is_noncontent(node):
    return (local_name(node) in NONCONTENT
            or (node.tag == qn("w:tr") and node.find(DELETED_ROW) is not None))


def content_children(node):
    """Yield selected markup branches without changing source XPaths."""
    for child in node:
        if child.tag != MC + "AlternateContent":
            yield child
            continue
        branch = None
        for choice in child.findall(MC + "Choice"):
            required = choice.get("Requires", "").split()
            if required and all(choice.nsmap.get(prefix) in SUPPORTED_CHOICE_NAMESPACES for prefix in required):
                branch = choice
                break
        if branch is None:
            branch = child.find(MC + "Fallback")
        if branch is None:
            yield child  # Keep the unresolved marker visible to source review.
        else:
            yield from content_children(branch)


def content_nodes(node, stop_at=(), hidden_run=None):
    name = local_name(node)
    if not name or is_noncontent(node) or (hidden_run is not None and hidden_run(node)):
        return
    yield node
    if name not in EMBEDDED_CONTENT and name not in stop_at and node.tag != MC + "AlternateContent":
        for child in content_children(node):
            yield from content_nodes(child, stop_at, hidden_run)


def table_items(node, tag):
    """Unwrap row or cell controls without collecting a nested table's grid."""
    for child in content_children(node):
        if is_noncontent(child):
            continue
        if child.tag == tag:
            yield child
        elif local_name(child) in TABLE_WRAPPERS:
            yield from table_items(child, tag)


def native_math(node):
    """Render common OMML expressions as Markdown math."""
    try:
        if local_name(node) == "oMathPara":
            expressions = [_math_latex(child) for child in content_children(node)
                           if local_name(child) == "oMath"]
            if not expressions:
                raise ValueError("Missing display math expression")
            return "\n\n".join("$$\n" + expression + "\n$$" for expression in expressions)
        latex = _math_latex(node)
    except ValueError:
        # Keep less common constructs reconstructable from the original OMML.
        preserved = deepcopy(node)
        if not any(item.tag == MC + "AlternateContent" or any(key.startswith(MC) for key in item.attrib)
                   for item in preserved.iter() if isinstance(item.tag, str)):
            etree.cleanup_namespaces(preserved)
        xml = etree.tostring(preserved, encoding="unicode", with_tail=False)
        return '<code data-format="omml">' + escape(xml) + "</code>"
    return "$" + latex + "$"


def _math_latex(node):
    name = local_name(node)
    if name == "t":
        return (node.text or "").translate(TEX_CHARACTERS)
    if name in ("rPr", "ctrlPr", "argPr", "oMathParaPr"):
        return ""
    if name in ("oMath", "oMathPara", "r", "e", "num", "den", "sup", "sub", "deg"):
        return "".join(_math_latex(child) for child in content_children(node))
    arguments = {local_name(child): child for child in content_children(node)}

    def arg(key):
        child = arguments.get(key)
        if child is None:
            raise ValueError("Missing OMML argument " + key)
        return _math_latex(child)

    if name == "f":
        fraction_kind = node.find(qn("m:fPr") + "/" + qn("m:type"))
        if fraction_kind is not None and fraction_kind.get(qn("m:val"), "bar") != "bar":
            raise ValueError("Unsupported OMML fraction layout")
        return r"\frac{" + arg("num") + "}{" + arg("den") + "}"
    if name in ("sSup", "sSub", "sSubSup"):
        latex = "{" + arg("e") + "}"
        if name in ("sSub", "sSubSup"):
            latex += "_{" + arg("sub") + "}"
        if name in ("sSup", "sSubSup"):
            latex += "^{" + arg("sup") + "}"
        return latex
    if name == "rad":
        degree = arg("deg")
        hidden = node.find(qn("m:radPr") + "/" + qn("m:degHide"))
        if hidden is not None and hidden.get(qn("m:val"), "1") not in ("0", "false", "off"):
            degree = ""
        index = "[" + degree + "]" if degree else ""
        return r"\sqrt" + index + "{" + arg("e") + "}"
    raise ValueError("Unsupported OMML construct " + name)


def _run_script(node, styles=None):
    if node.tag != qn("w:r"):
        return ""
    properties = node.find(qn("w:rPr"))
    if properties is not None:
        script = properties.find(qn("w:vertAlign"))
        if script is not None:
            return {"superscript": "sup", "subscript": "sub"}.get(script.get(qn("w:val")), "")
        style = properties.find(qn("w:rStyle"))
        if styles is not None and style is not None:
            return styles.get(style.get(qn("w:val")), "")
    return ""


def _script_text(text, script):
    if text and script:
        return "<" + script + ">" + escape(text, quote=False) + "</" + script + ">"
    return text


def native_text(node, styles=None):
    name = local_name(node)
    if is_noncontent(node) or name in EMBEDDED_CONTENT or node.tag == MC + "AlternateContent":
        return ""
    if name in ("drawing", "pict", "txbxContent"):
        return ""
    if name in ("t", "instrText"):
        if name == "instrText":
            return ""
        return node.text or ""
    if name in ("br", "cr"):
        return "\n"
    if name in ("tab", "ptab"):
        return "\t"
    if name == "noBreakHyphen":
        return "‑"
    if name == "softHyphen":
        return "\u00ad"
    if name == "sym":
        value = int(node.get(qn("w:char"), ""), 16)
        if value < 1 or value > 0xFFFF or 0xD800 <= value <= 0xDFFF:
            raise ValueError("Invalid Word symbol code point")
        return chr(value)
    if name in ("footnoteReference", "endnoteReference"):
        return ""
    if name in ("oMath", "oMathPara"):
        return native_math(node)
    return _script_text("".join(native_text(child, styles) for child in content_children(node)),
                        _run_script(node, styles))
