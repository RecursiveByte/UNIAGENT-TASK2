import json

RESULT_MAP = {"angenommen": "angenommen", "abgelehnt": "abgelehnt"}


def write_prediction(output_file, case_id: str, decision: str, begruendung: str) -> None:
    result = RESULT_MAP.get(decision, "abgelehnt")
    output_file.write(
        json.dumps({"antrag": case_id, "result": result, "begruendung": begruendung}, ensure_ascii=False) + "\n"
    )
    output_file.flush()