from pathlib import Path

from pypdf import PdfReader


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages).strip()

    if not text:
        raise ValueError(f"No text extracted from {pdf_path.name}")

    return text


def parse_case(documents: list) -> dict[str, str]:
    return {doc.filename: extract_text(doc.path) for doc in documents}