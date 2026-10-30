"""Check the compiled paper and render pages for human layout review."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pymupdf

PAPER = Path(__file__).resolve().parents[1]
PDF = PAPER / "main.pdf"
PREVIEW = PAPER / ".build" / "preview"


def main():
    PREVIEW.mkdir(parents=True, exist_ok=True)
    log = (PAPER / ".build" / "main.log").read_text(encoding="utf-8", errors="replace")
    patterns = {
        "latex_error": r"^!|LaTeX Error|Fatal error",
        "missing_glyph": r"Missing character:",
        "unresolved_reference": r"(?:Reference|Citation) .+undefined|There were undefined|undefined references|undefined citations",
        "overfull": r"Overfull \\[hv]box",
    }
    findings = {name: re.findall(pattern, log, re.MULTILINE) for name, pattern in patterns.items()}
    document = pymupdf.open(PDF)
    texts, bounds_issues = [], []
    contact = pymupdf.open()
    cols, cell_w, cell_h, per_sheet = 4, 210, 310, 12
    sheets = []
    for index, page in enumerate(document):
        texts.append(page.get_text())
        for block in page.get_text("blocks"):
            rect = pymupdf.Rect(block[:4])
            if rect.x0 < -1 or rect.y0 < -1 or rect.x1 > page.rect.width + 1 or rect.y1 > page.rect.height + 1:
                bounds_issues.append({"page": index+1, "bbox": list(rect)})
        page.get_pixmap(dpi=110).save(PREVIEW / f"page-{index+1:02}.png")
        local = index % per_sheet
        if local == 0:
            sheet = contact.new_page(width=cols*cell_w, height=3*cell_h)
            sheets.append(sheet.number)
        # Reacquire each page after new_page to avoid orphaned wrappers.
        sheet = contact[sheets[-1]]
        x, y = (local % cols)*cell_w, (local // cols)*cell_h
        sheet.show_pdf_page(pymupdf.Rect(x+6, y+6, x+cell_w-6, y+cell_h-22), document, index)
        sheet.insert_text((x+8, y+cell_h-8), f"Page {index+1}", fontsize=10)
    for number in sheets:
        contact[number].get_pixmap(dpi=110).save(PREVIEW / f"contact-{number+1:02}.png")
    text = "\n".join(texts)
    (PAPER / ".build" / "paper-extracted.txt").write_text(text, encoding="utf-8")
    source_files = [PAPER / "main.tex", *(PAPER / "sections").glob("*.tex")]
    source = "\n".join(path.read_text(encoding="utf-8") for path in source_files)
    cite_keys = set(re.findall(r"\\cite\{([^}]+)\}", source))
    cite_keys = {key for group in cite_keys for key in group.split(",")}
    bib_keys = set(re.findall(r"@\w+\{([^,]+)", (PAPER / "references.bib").read_text(encoding="utf-8")))
    assert cite_keys <= bib_keys
    section_source = "\n".join(
        (PAPER / "sections" / name).read_text(encoding="utf-8")
        for name in ["01-multilayer.tex", "02-adaptive.tex", "03-backbone.tex", "04-alternatives.tex"])
    section_source = section_source.replace(r"\texorpdfstring{$\lambda$}{λ}", "λ")
    main_sections = re.findall(r"\\section\{([^}]+)\}", section_source)
    assert len(main_sections) == 4
    figure_names = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", source)
    figure_sources = json.loads((PAPER / "data" / "figure_sources.json").read_text(encoding="utf-8"))
    assert figure_names
    assert set(figure_names) <= {name + ".pdf" for name in figure_sources}
    assert all((PAPER / "figures" / name).is_file() for name in figure_names)
    failures: dict[str, object] = {key: values for key, values in findings.items() if values}
    if "\ufffd" in text or "[?]" in text or "??" in text:
        failures["extracted_text"] = "replacement character or unresolved reference marker"
    if bounds_issues:
        failures["page_bounds"] = bounds_issues
    result = {
        "passed": not failures, "pdf": "main.pdf", "pages": len(document),
        "pdf_bytes": PDF.stat().st_size, "main_sections": main_sections,
        "figures": len(figure_names), "available_figures": len(figure_sources),
        "bibliography_entries": len(bib_keys),
        "latex_findings": findings, "text_outside_page": bounds_issues,
        "page_rendering": f".build/preview/page-01.png through page-{len(document):02}.png",
        "visual_review": "Page rendering produced; visual inspection is recorded separately after review.",
        "training_performed": False,
    }
    (PAPER / "data" / "document_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit("Document validation failed: " + json.dumps(failures, ensure_ascii=False))


if __name__ == "__main__":
    main()
