import argparse
import os
from pathlib import Path

from agent import build_graph
from app_logging import EventLogger
from documents import determine_case_type, find_cases, load_case_documents, parse_case
from llm import build_client, load_settings
from output import write_prediction
from retriever import get_index
from tools import make_search_policy_tool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-directory", dest="input", type=Path,
                        default=os.environ.get("inputDataset"))
    parser.add_argument("--output-directory", dest="output", type=Path,
                        default=os.environ.get("outputDir"))

    args = parser.parse_args()
    if args.input is None or args.output is None:
        parser.error("input and output directory are required")
    args.input = Path(args.input)
    args.output = Path(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    settings = load_settings()
    logger = EventLogger(model=settings.openai_model)

    try:
        llm = build_client(settings)
        corpus_root = args.input / "retrieval-corpora"

        if corpus_root.is_dir():
            for corpus_dir in sorted(p for p in corpus_root.iterdir() if p.is_dir()):
                get_index(corpus_root, corpus_dir.name)

        search_policy = make_search_policy_tool(corpus_root, llm)
        graph = build_graph(llm, [search_policy])

        case_ids = find_cases(args.input)
        predictions_path = args.output / "predictions.jsonl"

        with open(predictions_path, "w", encoding="utf-8") as predictions_file:
            for case_id in case_ids:
                try:
                    case_type = determine_case_type(case_id)
                    documents = load_case_documents(args.input, case_id)
                    parsed = parse_case(documents)

                    result = graph.invoke({
                        "case_id": case_id,
                        "case_type": case_type,
                        "documents": parsed,
                        "messages": [],
                        "decision": None,
                        "begruendung": None,
                        "logger": logger,
                        "last_event_id": None,
                        "tool_call_count": 0,
                    })
                    decision = result["decision"]
                    begruendung = result["begruendung"]
                    write_prediction(predictions_file, case_id, decision, begruendung)

                except Exception as error:
                    logger.log_error(case_id, None, str(error), event_type="error")
                    write_prediction(
                        predictions_file,
                        case_id,
                        "abgelehnt",
                        f"Fehler bei der Verarbeitung: {error}",
                    )
    finally:
        logger.write(args.output / "run-trace.jsonl.log.gz")


if __name__ == "__main__":
    main()