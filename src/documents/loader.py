from dataclasses import dataclass
from pathlib import Path


@dataclass
class CaseDocument:
    case_id: str
    filename: str
    path: Path


def _pdf_files(directory: Path) -> list[Path]:
    return sorted(
        entry for entry in directory.iterdir()
        if entry.is_file() and entry.suffix.lower() == ".pdf"
    )


def find_cases(input_root: Path) -> list[str]:
    case_ids = []
    for entry in sorted(input_root.iterdir()):
        if entry.is_dir() and _pdf_files(entry):
            case_ids.append(entry.name)
    return case_ids


def load_case_documents(input_root: Path, case_id: str) -> list[CaseDocument]:
    case_dir = input_root / case_id
    if not case_dir.is_dir():
        raise ValueError(f"Unknown case: {case_id}")

    return [
        CaseDocument(case_id=case_id, filename=pdf.name, path=pdf)
        for pdf in _pdf_files(case_dir)
    ]