#!/usr/bin/env python3
"""Migrate TOSA op attribute syntax in test files after the LLVM bump.

Upstream TOSA moved inherent attributes out of `attr-dict` into keyword
syntax, e.g.

    tosa.transpose %a {perms = array<i32: 0, 2, 1>} : ...
 -> tosa.transpose %a perms([0, 2, 1]) : ...

This script rewrites the old spelling in place.  Attributes it does not know
about are left in the trailing `{...}` dictionary (they are discardable).
"""
import re
import sys

PAREN, ANGLE = "paren", "angle"
ARRAY, PLAIN, STRIP_TYPE, ENUM = "array", "plain", "strip_type", "enum"

# op mnemonic -> ordered list of (attr name, delimiter style, value style)
RULES = {
    "transpose": [("perms", PAREN, ARRAY)],
    "const_shape": [("values", PAREN, PLAIN)],
    "reduce_sum": [("axis", PAREN, STRIP_TYPE)],
    "reduce_max": [("axis", PAREN, STRIP_TYPE), ("nan_mode", ANGLE, ENUM)],
    "reduce_min": [("axis", PAREN, STRIP_TYPE), ("nan_mode", ANGLE, ENUM)],
    "reduce_all": [("axis", PAREN, STRIP_TYPE)],
    "reduce_any": [("axis", PAREN, STRIP_TYPE)],
    "reduce_prod": [("axis", PAREN, STRIP_TYPE)],
    "argmax": [("axis", PAREN, STRIP_TYPE), ("nan_mode", ANGLE, ENUM)],
    "concat": [("axis", PAREN, STRIP_TYPE)],
    "reverse": [("axis", PAREN, STRIP_TYPE)],
    "clamp": [("min_val", PAREN, PLAIN), ("max_val", PAREN, PLAIN),
              ("nan_mode", ANGLE, ENUM)],
    "maximum": [("nan_mode", ANGLE, ENUM)],
    "minimum": [("nan_mode", ANGLE, ENUM)],
    "custom": [("operator_name", PAREN, PLAIN), ("domain_name", PAREN, PLAIN),
               ("implementation_attrs", PAREN, PLAIN)],
    "conv2d": [("pad", PAREN, ARRAY), ("stride", PAREN, ARRAY),
               ("dilation", PAREN, ARRAY), ("acc_type", PAREN, PLAIN)],
    "conv3d": [("pad", PAREN, ARRAY), ("stride", PAREN, ARRAY),
               ("dilation", PAREN, ARRAY), ("acc_type", PAREN, PLAIN)],
    "depthwise_conv2d": [("pad", PAREN, ARRAY), ("stride", PAREN, ARRAY),
                         ("dilation", PAREN, ARRAY), ("acc_type", PAREN, PLAIN)],
    "transpose_conv2d": [("out_pad", PAREN, ARRAY), ("stride", PAREN, ARRAY),
                         ("acc_type", PAREN, PLAIN)],
    "max_pool2d": [("kernel", PAREN, ARRAY), ("stride", PAREN, ARRAY),
                   ("pad", PAREN, ARRAY), ("nan_mode", ANGLE, ENUM)],
    "avg_pool2d": [("kernel", PAREN, ARRAY), ("stride", PAREN, ARRAY),
                   ("pad", PAREN, ARRAY), ("acc_type", PAREN, PLAIN)],
    "matmul_t_block_scaled": [("block_size", ANGLE, ENUM)],
    "resize": [("mode", ANGLE, ENUM)],
    "rfft2d": [("local_bound", PAREN, PLAIN)],
}

OP_RE = re.compile(r"(\"?)tosa\.([a-z_0-9]+)\b\"?")


def split_top_level(text, sep=","):
    """Split on `sep` ignoring anything nested in brackets/quotes."""
    parts, depth, start, in_str = [], 0, 0, False
    i = 0
    while i < len(text):
        c = text[i]
        if in_str:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c in "([{<":
            depth += 1
        elif c in ")]}>":
            depth -= 1
        elif c == sep and depth == 0:
            parts.append(text[start:i])
            start = i + 1
        i += 1
    parts.append(text[start:])
    return [p.strip() for p in parts if p.strip()]


def find_attr_dict(text, pos):
    """Starting after a tosa op mnemonic, return (start, end) of its attr dict.

    Returns None when the op has no dictionary (we stop at the `:` that
    introduces the type list).  FileCheck `{{...}}` regexes are skipped.
    """
    depth = 0
    i = pos
    n = len(text)
    while i < n:
        c = text[i]
        if text.startswith("{{", i):  # FileCheck regex, not an attr dict
            j = text.find("}}", i)
            if j < 0:
                return None
            i = j + 2
            continue
        if text.startswith("->", i):
            i += 2
            continue
        if c == "\n":
            # Keep scanning onto the next line only while inside brackets.
            if depth == 0:
                nxt = text[i + 1:i + 200].lstrip()
                if not nxt.startswith(("{", ":")):
                    return None
        if c in "([<":
            depth += 1
        elif c in ")]>":
            depth -= 1
            if depth < 0:
                return None
        elif c == "{" and depth == 0:
            end = i + 1
            d = 1
            while end < n and d:
                if text[end] == "{":
                    d += 1
                elif text[end] == "}":
                    d -= 1
                end += 1
            return (i, end)
        elif c == ":" and depth == 0:
            return None
        i += 1
    return None


def render(name, delim, style, value):
    if style == ARRAY:
        m = re.fullmatch(r"array<\s*i\d+\s*:\s*(.*)>", value.strip(), re.S)
        if m:
            value = "[%s]" % m.group(1).strip()
        elif value.strip() == "array<i64>" or value.strip() == "array<i32>":
            value = "[]"
    elif style == STRIP_TYPE:
        value = re.sub(r"\s*:\s*i\d+$", "", value.strip())
    elif style == ENUM:
        m = re.fullmatch(r"#tosa\.[a-z_0-9]+<(.*)>", value.strip(), re.S)
        if m:
            value = m.group(1).strip()
    if delim == ANGLE:
        return "%s<%s>" % (name, value)
    return "%s(%s)" % (name, value)


def migrate(text):
    out = []
    pos = 0
    changed = 0
    for m in OP_RE.finditer(text):
        if m.start() < pos:
            continue
        generic = m.group(1) == '"'
        op = m.group(2)
        rules = RULES.get(op)
        if not rules:
            continue
        span = find_attr_dict(text, m.end())
        if not span:
            continue
        start, end = span
        body = text[start + 1:end - 1]
        entries = []
        for item in split_top_level(body):
            kv = item.split("=", 1)
            if len(kv) != 2:
                entries.append((None, item))
                continue
            entries.append((kv[0].strip(), kv[1].strip()))
        known = {k: v for k, v in entries if k}
        if generic:
            # Generic form: inherent attributes live in the `<{...}>`
            # properties dictionary, values keep their printed spelling.
            inherent = [name for name, _, _ in rules if name in known]
            if not inherent:
                continue
            moved = ["%s = %s" % (name, known.pop(name)) for name in inherent]
            new = "<{%s}>" % ", ".join(moved)
            rest = [item if k is None else "%s = %s" % (k, known[k])
                    for k, item in entries if not k or k in known]
            if rest:
                new += " {%s}" % ", ".join(rest)
            out.append(text[pos:start])
            out.append(new)
            pos = end
            changed += 1
            continue
        moved = [render(name, delim, style, known.pop(name))
                 for name, delim, style in rules if name in known]
        if not moved:
            continue
        leftover = [item for k, item in entries if not k or k in known]
        new = " ".join(moved)
        if leftover:
            new += " {%s}" % ", ".join(
                item if k is None else "%s = %s" % (k, known[k])
                for k, item in entries if not k or k in known)
        out.append(text[pos:start])
        out.append(new)
        pos = end
        # drop the space that preceded the dict if we emitted nothing
        changed += 1
    out.append(text[pos:])
    return "".join(out), changed


def main(paths):
    total = 0
    for path in paths:
        with open(path) as f:
            text = f.read()
        new, n = migrate(text)
        if n:
            with open(path, "w") as f:
                f.write(new)
            print("%s: %d op(s)" % (path, n))
            total += n
    print("total: %d" % total)


if __name__ == "__main__":
    main(sys.argv[1:])
