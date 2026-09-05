import re
from pathlib import Path

import pymupdf


class DocumentError(ValueError):
    pass


def extract_document(path: Path, suffix: str) -> list[dict[str, object]]:
    if suffix == ".pdf":
        try:
            document = pymupdf.open(path)
        except Exception as exc:
            raise DocumentError(f"Unable to open PDF: {exc}") from exc
        if document.needs_pass:
            document.close()
            raise DocumentError("Encrypted PDFs are not supported")
        pages = [{"page": index + 1, "text": page.get_text("text")} for index, page in enumerate(document)]
        document.close()
    else:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise DocumentError("Text files must use UTF-8 encoding") from exc
        pages = [{"page": 1, "text": text}]
    if not any(str(page["text"]).strip() for page in pages):
        raise DocumentError("No extractable text was found; scanned PDFs require OCR")
    return pages


def clean_text(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def text_statistics(text: str) -> dict[str, int]:
    paragraphs = [part for part in re.split(r"\n\s*\n", text) if part.strip()]
    sentences = [part for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    return {"characters": len(text), "paragraphs": len(paragraphs), "sentences": len(sentences)}


def structured_units(text: str) -> list[dict[str, str]]:
    """Split text into meaningful units that can be packed into token-bounded chunks."""
    units: list[dict[str, str]] = []
    for paragraph in (part.strip() for part in re.split(r"\n\s*\n", text)):
        if not paragraph:
            continue
        lines = paragraph.splitlines()
        if len(lines) == 1 and (paragraph.startswith("#") or (len(paragraph) < 100 and paragraph.endswith(":"))):
            units.append({"kind": "heading", "text": paragraph})
            continue
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", paragraph) if part.strip()]
        if len(sentences) <= 1:
            units.append({"kind": "paragraph", "text": paragraph})
        else:
            units.extend({"kind": "sentence", "text": sentence} for sentence in sentences)
    return units
