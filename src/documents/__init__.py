from .loader import CaseDocument, find_cases, load_case_documents
from .parser import extract_text, parse_case

CASE_TYPE_PREFIXES = {
    "dienstreiseantrag": "business_travel",
    "beschaffungsantrag": "procurement",
    "auslagenerstattung": "expense_reimbursement",
}


def determine_case_type(case_id: str) -> str:
    for prefix, case_type in CASE_TYPE_PREFIXES.items():
        if case_id.startswith(prefix):
            return case_type
    raise ValueError(f"Unknown case type for case_id: {case_id}")

__all__ = ["CaseDocument", "find_cases", "load_case_documents", "extract_text", "parse_case"]