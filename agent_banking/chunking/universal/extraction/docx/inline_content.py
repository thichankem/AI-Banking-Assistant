"""Read visible inline text, Word link targets and authored textbox positions."""

import shlex

from docx.oxml.ns import qn

from .openxml import (
    EMBEDDED_CONTENT,
    MC,
    NONCONTENT,
    _run_script,
    _script_text,
    content_children,
    content_nodes,
    local_name,
    native_text,
)


TEXT_NODES = {"t", "tab", "ptab", "br", "cr", "noBreakHyphen", "softHyphen", "sym", "oMath", "oMathPara"}
VISUAL_NODES = {"drawing", "pict", "txbxContent"}


def paragraph_parts(paragraph, part, script_styles, hidden_run, *, include_images=False):
    """Yield text segments with their hyperlinks, interleaved with textbox nodes."""
    buffer = []
    fragments = {}
    fields = []
    for kind, value, link in _inline_events(paragraph, part, script_styles, hidden_run, include_images=include_images):
        if kind in ("visual", "image"):
            if buffer:
                yield "text", "".join(buffer), _links(fragments)
                buffer.clear()
                fragments.clear()
            yield ("textbox" if kind == "visual" else kind), value, []
        elif kind == "field":
            marker = value.get(qn("w:fldCharType"))
            if marker == "begin":
                fields.append({"node": value, "instruction": [], "target": None})
            elif marker == "separate" and fields:
                fields[-1]["target"] = _field_link("".join(fields[-1]["instruction"]))
            elif marker == "end" and fields:
                fields.pop()
        elif kind == "instruction":
            if fields:
                fields[-1]["instruction"].append(value)
        else:
            buffer.append(value)
            if link is None:
                link = next(((field["node"], field["target"]) for field in reversed(fields) if field["target"]), None)
            if link is not None:
                fragments.setdefault(link, []).append(value)
    if buffer:
        yield "text", "".join(buffer), _links(fragments)


def _links(fragments):
    return [{"text": "".join(values), "target": target}
            for (_, target), values in fragments.items() if target and any(values)]


def _target_with_anchor(target, anchor):
    if not anchor:
        return target
    return (target or "") + "#" + anchor


def _field_link(instruction):
    try:
        tokens = shlex.split(instruction, posix=False)
    except ValueError:
        return None
    if len(tokens) < 2 or tokens[0].casefold() != "hyperlink":
        return None
    target = None
    if not tokens[1].startswith("\\"):
        target = tokens[1].strip('"')
    for index, token in enumerate(tokens[1:-1], 1):
        if token.casefold() == "\\l":
            return _target_with_anchor(target, tokens[index + 1].strip('"'))
    return target


def _hyperlink_target(node, part):
    if node.tag == qn("w:fldSimple"):
        return _field_link(node.get(qn("w:instr"), ""))
    relation_id = node.get(qn("r:id"))
    target = None
    if relation_id and relation_id in part.rels:
        target = part.rels[relation_id].target_ref
    return _target_with_anchor(target, node.get(qn("w:anchor")))


def _inline_events(node, part, script_styles, hidden_run, link=None, *, include_images=False):
    for child in content_children(node):
        name = local_name(child)
        if name in NONCONTENT or name in EMBEDDED_CONTENT or child.tag == MC + "AlternateContent" or hidden_run(child):
            continue
        if name in VISUAL_NODES:
            if any(item.tag == qn("w:txbxContent") for item in content_nodes(child, stop_at=("txbxContent",))):
                yield "visual", child, None
            if include_images:
                for item in content_nodes(child, stop_at=("txbxContent",), hidden_run=hidden_run):
                    if item.tag == qn("a:blip") or local_name(item) == "imagedata":
                        yield "image", item, None
        elif name in TEXT_NODES:
            yield "text", native_text(child), link
        elif name == "fldChar":
            yield "field", child, None
        elif name == "instrText":
            yield "instruction", child.text or "", None
        elif name == "r":
            events = _inline_events(child, part, script_styles, hidden_run, link, include_images=include_images)
            yield from _script_events(events, _run_script(child, script_styles), link)
        else:
            target = _hyperlink_target(child, part) if name in ("hyperlink", "fldSimple") else None
            child_link = (child, target) if target else link
            yield from _inline_events(child, part, script_styles, hidden_run, child_link, include_images=include_images)


def _script_events(events, script, link):
    if not script:
        yield from events
        return
    text = []
    for kind, value, event_link in events:
        if kind == "text":
            text.append(value)
            continue
        if text:
            yield "text", _script_text("".join(text), script), link
            text.clear()
        yield kind, value, event_link
    if text:
        yield "text", _script_text("".join(text), script), link
