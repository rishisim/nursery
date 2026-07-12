from __future__ import annotations

import argparse
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "docs" / "frank_preaccess_experimental_plan.md"
DEFAULT_OUTPUT = ROOT / "output" / "pdf" / "frank_preaccess_experimental_plan.pdf"


def _ascii_punctuation(text: str) -> str:
    replacements = {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2192": "->",
        "\u0394": "Delta",
        "\u00b7": " | ",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _inline_markup(text: str) -> str:
    text = _ascii_punctuation(text.strip())
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
    return text


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    ink = colors.HexColor("#183143")
    muted = colors.HexColor("#536878")
    accent = colors.HexColor("#0B7285")
    return {
        "title": ParagraphStyle(
            "PlanTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=19,
            textColor=ink,
            alignment=TA_LEFT,
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "PlanSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8.2,
            leading=10,
            textColor=muted,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "PlanH2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=13.5,
            textColor=accent,
            spaceBefore=1,
            spaceAfter=5,
        ),
        "h3": ParagraphStyle(
            "PlanH3",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=9.1,
            leading=10.8,
            textColor=ink,
            spaceBefore=4,
            spaceAfter=2,
        ),
        "body": ParagraphStyle(
            "PlanBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.15,
            leading=10.15,
            textColor=colors.HexColor("#243642"),
            spaceAfter=4,
        ),
        "question": ParagraphStyle(
            "PlanQuestion",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10.1,
            leading=12.4,
            textColor=colors.white,
            backColor=accent,
            borderPadding=(7, 9, 7, 9),
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "PlanBullet",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.8,
            leading=9.6,
            textColor=colors.HexColor("#243642"),
        ),
        "code": ParagraphStyle(
            "PlanCode",
            parent=base["Code"],
            fontName="Courier",
            fontSize=6.5,
            leading=8,
            leftIndent=8,
            rightIndent=8,
            textColor=ink,
            backColor=colors.HexColor("#EEF4F5"),
            borderPadding=6,
            spaceBefore=2,
            spaceAfter=6,
        ),
        "footer": ParagraphStyle(
            "PlanFooter",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=6.8,
            textColor=muted,
            alignment=TA_CENTER,
        ),
    }


def _table(rows: list[list[str]], styles: dict[str, ParagraphStyle]) -> Table:
    parsed = [[Paragraph(_inline_markup(cell), styles["bullet"]) for cell in row] for row in rows]
    width = 7.25 * inch
    col_count = len(parsed[0])
    col_widths = [width / col_count] * col_count
    if col_count == 3:
        col_widths = [1.2 * inch, 3.2 * inch, 2.85 * inch]
    table = Table(parsed, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDECEF")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#183143")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#A8BBC4")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFA")]),
            ]
        )
    )
    return table


def _parse_markdown(text: str, styles: dict[str, ParagraphStyle]) -> list[object]:
    lines = text.splitlines()
    story: list[object] = []
    paragraph: list[str] = []
    bullets: list[str] = []
    numbered: list[str] = []
    in_code = False
    code_lines: list[str] = []
    table_rows: list[list[str]] = []
    in_math = False

    def flush_paragraph() -> None:
        if not paragraph:
            return
        content = " ".join(item.strip() for item in paragraph)
        style = styles["question"] if content.startswith("**Does synchronized") else styles["body"]
        story.append(Paragraph(_inline_markup(content), style))
        paragraph.clear()

    def flush_lists() -> None:
        if bullets:
            items = [ListItem(Paragraph(_inline_markup(item), styles["bullet"])) for item in bullets]
            story.append(ListFlowable(items, bulletType="bullet", leftIndent=15, bulletFontSize=5.5, spaceAfter=4))
            bullets.clear()
        if numbered:
            items = [ListItem(Paragraph(_inline_markup(item), styles["bullet"])) for item in numbered]
            story.append(ListFlowable(items, bulletType="1", start="1", leftIndent=17, bulletFontSize=7, spaceAfter=4))
            numbered.clear()

    def flush_table() -> None:
        if not table_rows:
            return
        usable = [row for row in table_rows if not all(re.fullmatch(r":?-+:?", cell.strip()) for cell in row)]
        story.append(_table(usable, styles))
        story.append(Spacer(1, 5))
        table_rows.clear()

    for raw in lines:
        line = raw.rstrip()
        if line.strip() == "\\[":
            flush_paragraph(); flush_lists(); flush_table(); in_math = True
            continue
        if in_math:
            if line.strip() == "\\]":
                story.append(Paragraph("<b>Delta = score(synchronized motor) - score(episode-shuffled motor)</b>", styles["body"]))
                in_math = False
            continue
        if line.startswith("```"):
            flush_paragraph(); flush_lists(); flush_table()
            if in_code:
                story.append(Preformatted(_ascii_punctuation("\n".join(code_lines)), styles["code"]))
                code_lines.clear()
            in_code = not in_code
            continue
        if in_code:
            code_lines.append(line)
            continue
        if "page-break-after" in line:
            flush_paragraph(); flush_lists(); flush_table(); story.append(PageBreak())
            continue
        if line.startswith("# "):
            flush_paragraph(); flush_lists(); flush_table()
            story.append(Paragraph(_inline_markup(line[2:]), styles["title"]))
            continue
        if line.startswith("## "):
            flush_paragraph(); flush_lists(); flush_table()
            story.append(Paragraph(_inline_markup(line[3:]), styles["h2"]))
            continue
        if line.startswith("### "):
            flush_paragraph(); flush_lists(); flush_table()
            story.append(Paragraph(_inline_markup(line[4:]), styles["h3"]))
            continue
        if re.match(r"^\|.*\|$", line.strip()):
            flush_paragraph(); flush_lists()
            table_rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
            continue
        if line.startswith("- "):
            flush_paragraph(); flush_table(); numbered and flush_lists()
            bullets.append(line[2:])
            continue
        numbered_match = re.match(r"^\d+\.\s+(.+)$", line)
        if numbered_match:
            flush_paragraph(); flush_table(); bullets and flush_lists()
            numbered.append(numbered_match.group(1))
            continue
        if not line.strip():
            flush_paragraph(); flush_lists(); flush_table()
            continue
        if line.strip() == "---":
            continue
        if line.startswith("**Pre-access"):
            flush_paragraph(); flush_lists(); flush_table()
            story.append(Paragraph(_inline_markup(line), styles["subtitle"]))
            continue
        paragraph.append(line)

    flush_paragraph(); flush_lists(); flush_table()
    return story


def _footer(canvas, document) -> None:  # type: ignore[no-untyped-def]
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#C8D5DA"))
    canvas.setLineWidth(0.4)
    canvas.line(0.62 * inch, 0.43 * inch, 7.88 * inch, 0.43 * inch)
    canvas.setFont("Helvetica", 6.8)
    canvas.setFillColor(colors.HexColor("#536878"))
    canvas.drawCentredString(4.25 * inch, 0.27 * inch, f"Rishi Simhadri | Pre-access experimental plan | {document.page}")
    canvas.restoreState()


def build_pdf(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    story = _parse_markdown(input_path.read_text(), styles)
    document = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=0.62 * inch,
        rightMargin=0.62 * inch,
        topMargin=0.48 * inch,
        bottomMargin=0.55 * inch,
        title="Toward a BabyView-calibrated sensorimotor cue study",
        author="Rishi Simhadri",
        subject="Pre-access experimental plan for discussion with Professor Michael Frank",
    )
    document.build(story, onFirstPage=_footer, onLaterPages=_footer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the two-page Frank pre-access plan PDF.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build_pdf(args.input, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
