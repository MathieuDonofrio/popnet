"""Render self-contained Markdown previews for the code report."""

from html import escape
from xml.etree import ElementTree

from latex2mathml.converter import convert
from markdown_it import MarkdownIt
from mdit_py_plugins.dollarmath import dollarmath_plugin


MATHML = "http://www.w3.org/1998/Math/MathML"
MATH_ELEMENTS = {
    "math", "mrow", "mi", "mn", "mo", "mtext", "ms", "mspace", "mfrac",
    "msqrt", "mroot", "msub", "msup", "msubsup", "munder", "mover", "munderover",
    "mmultiscripts", "mprescripts", "none", "mtable", "mtr", "mtd", "mstyle",
    "mphantom", "mpadded", "menclose", "merror",
}
MATH_ATTRIBUTES = {
    "accent", "accentunder", "align", "bevelled", "columnalign", "columnlines",
    "columnspacing", "columnspan", "depth", "display", "displaystyle", "fence",
    "form", "height", "linethickness", "lspace", "mathsize", "mathvariant",
    "maxsize", "minsize", "movablelimits", "notation", "rowalign", "rowlines",
    "rowspacing", "rowspan", "rspace", "scriptlevel", "separator", "stretchy",
    "symmetric", "voffset", "width",
}


def render_math(latex: str, options: dict) -> str:
    display = options["display_mode"]
    delimiter = "$$" if display else "$"
    fallback = "<code>" + escape(delimiter + latex + delimiter) + "</code>"
    try:
        markup = convert(latex, display="block" if display else "inline")
    except Exception:
        # The converter uses command-specific exceptions for invalid TeX.
        return fallback
    try:
        math = ElementTree.fromstring(markup)
    except ElementTree.ParseError:
        return fallback

    # Only presentation MathML can be embedded. Drop arbitrary attributes
    # such as event handlers, links, or styles even if a TeX command emits them.
    for node in math.iter():
        namespace, _, name = node.tag.rpartition("}")
        if namespace != "{" + MATHML or name not in MATH_ELEMENTS:
            return fallback
        node.tag = name
        node.attrib = {key: value for key, value in node.attrib.items() if key in MATH_ATTRIBUTES}
    math.set("xmlns", MATHML)
    return ElementTree.tostring(math, encoding="unicode", short_empty_elements=False)


def render_display_in_paragraph(tokens, index, _options, _env):
    math = render_math(tokens[index].content.strip(), {"display_mode": True})
    return f'<span class="math display">{math}</span>'


# No raw HTML or automatic external image requests in the offline dashboard.
MARKDOWN = MarkdownIt("js-default").disable("image").use(
    dollarmath_plugin, allow_labels=False, double_inline=True, renderer=render_math
)
MARKDOWN.renderer.rules["math_inline_double"] = render_display_in_paragraph
