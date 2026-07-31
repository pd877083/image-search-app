"""Build a clean printable PDF from VIVA_PREP.md (Q&A cheat sheet for viva)."""

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

SRC = Path(__file__).resolve().parent / "VIVA_PREP.md"
DST = Path(__file__).resolve().parent / "VIVA_PREP.pdf"

NOTO_REGULAR = Path("C:/Windows/Fonts/NotoSans-Regular.ttf")
NOTO_BOLD = Path("C:/Windows/Fonts/NotoSans-Bold.ttf")
NOTO_ITALIC = Path("C:/Windows/Fonts/NotoSans-Italic.ttf")
SEGOE_REGULAR = Path("C:/Windows/Fonts/segoeui.ttf")
SEGOE_BOLD = Path("C:/Windows/Fonts/segoeuib.ttf")

if NOTO_REGULAR.exists() and NOTO_BOLD.exists():
    pdfmetrics.registerFont(TTFont("Body", str(NOTO_REGULAR)))
    pdfmetrics.registerFont(TTFont("BodyBold", str(NOTO_BOLD)))
    italic_path = str(NOTO_ITALIC) if NOTO_ITALIC.exists() else str(NOTO_REGULAR)
    pdfmetrics.registerFont(TTFont("BodyItalic", italic_path))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="BodyBold", italic="BodyItalic", boldItalic="BodyBold")
    BODY_FONT, BODY_BOLD, BODY_ITALIC = "Body", "BodyBold", "BodyItalic"
elif SEGOE_REGULAR.exists() and SEGOE_BOLD.exists():
    pdfmetrics.registerFont(TTFont("Body", str(SEGOE_REGULAR)))
    pdfmetrics.registerFont(TTFont("BodyBold", str(SEGOE_BOLD)))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="BodyBold", italic="Body", boldItalic="BodyBold")
    BODY_FONT, BODY_BOLD, BODY_ITALIC = "Body", "BodyBold", "Body"
else:
    BODY_FONT, BODY_BOLD, BODY_ITALIC = "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"


def _md_inline_to_html(s: str) -> str:
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<!\*)\*(?!\s)([^*\n]+?)\*(?!\*)", r"<i>\1</i>", s)
    s = re.sub(r"`([^`]+?)`", r'<font face="Courier">\1</font>', s)
    return s


def _md_to_paragraphs(md: str):
    in_code = False
    code_buf = []
    for raw in md.splitlines():
        line = raw.rstrip()
        if line.startswith("```"):
            if in_code:
                yield "code", "\n".join(code_buf)
                code_buf = []
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code_buf.append(line)
            continue
        if not line.strip():
            yield "blank", ""
            continue
        if line.startswith("# "):
            yield "h1", line[2:].strip()
        elif line.startswith("## "):
            yield "h2", line[3:].strip()
        elif line.startswith("### "):
            yield "h3", line[4:].strip()
        elif line.strip() == "---":
            yield "hr", ""
        elif line.startswith("> "):
            yield "quote", line[2:].strip()
        elif re.match(r"^\s*[-*]\s+", line):
            yield "bullet", re.sub(r"^\s*[-*]\s+", "", line)
        else:
            yield "p", line


def build():
    md = SRC.read_text(encoding="utf-8")
    doc = SimpleDocTemplate(
        str(DST), pagesize=LETTER,
        leftMargin=0.7*inch, rightMargin=0.7*inch,
        topMargin=0.7*inch, bottomMargin=0.7*inch,
        title="Viva Preparation - Q&A Cheat Sheet",
        author="Dhruv Singhal",
    )
    styles = {
        "h1": ParagraphStyle("h1", fontName=BODY_BOLD, fontSize=20, leading=24,
                             textColor=colors.HexColor("#780000"), spaceBefore=12, spaceAfter=6),
        "h2": ParagraphStyle("h2", fontName=BODY_BOLD, fontSize=14, leading=18,
                             textColor=colors.HexColor("#003049"), spaceBefore=10, spaceAfter=4),
        "h3": ParagraphStyle("h3", fontName=BODY_BOLD, fontSize=11, leading=14,
                             textColor=colors.HexColor("#c1121f"), spaceBefore=6, spaceAfter=3),
        "p": ParagraphStyle("p", fontName=BODY_FONT, fontSize=9.5, leading=12,
                            textColor=colors.HexColor("#1a1a1a"), spaceAfter=3),
        "quote": ParagraphStyle("quote", fontName=BODY_ITALIC, fontSize=9.5, leading=12,
                                textColor=colors.HexColor("#003049"), leftIndent=14,
                                borderPadding=4, spaceAfter=5, spaceBefore=2,
                                backColor=colors.HexColor("#fdf0d5")),
        "code": ParagraphStyle("code", fontName="Courier", fontSize=8, leading=10,
                                textColor=colors.HexColor("#1a1a1a"),
                                backColor=colors.HexColor("#f4f4f4"),
                                borderPadding=4, leftIndent=4, spaceAfter=6, spaceBefore=2),
        "bullet": ParagraphStyle("bullet", fontName=BODY_FONT, fontSize=9.5, leading=12,
                                  textColor=colors.HexColor("#1a1a1a"),
                                  leftIndent=14, bulletIndent=4, spaceAfter=2),
    }
    story = []
    for kind, text in _md_to_paragraphs(md):
        if kind == "blank":
            story.append(Spacer(1, 3))
        elif kind == "h1":
            story.append(Paragraph(_md_inline_to_html(text), styles["h1"]))
        elif kind == "h2":
            story.append(Paragraph(_md_inline_to_html(text), styles["h2"]))
        elif kind == "h3":
            story.append(Paragraph(_md_inline_to_html(text), styles["h3"]))
        elif kind == "p":
            story.append(Paragraph(_md_inline_to_html(text), styles["p"]))
        elif kind == "quote":
            story.append(Paragraph(_md_inline_to_html(text), styles["quote"]))
        elif kind == "bullet":
            story.append(Paragraph("•  " + _md_inline_to_html(text), styles["bullet"]))
        elif kind == "code":
            story.append(Paragraph(text.replace("\n", "<br/>"), styles["code"]))
        elif kind == "hr":
            story.append(Spacer(1, 4))
            story.append(Table([[""]], colWidths=[7.1*inch], rowHeights=[0.02*inch],
                               style=TableStyle([("LINEABOVE", (0, 0), (-1, -1), 0.5, colors.HexColor("#888888"))])))
            story.append(Spacer(1, 4))
    doc.build(story)
    print(f"PDF written: {DST}")
    print(f"Size: {DST.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    build()
