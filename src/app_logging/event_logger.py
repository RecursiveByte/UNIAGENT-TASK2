import gzip
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class EventLogger:
    model: str
    events: list[dict] = field(default_factory=list)
    _counter: int = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"evt-{self._counter:04d}"

    def log(
        self,
        case_id: str,
        event_type: str,
        parent_event_id: str | None = None,
        tool: str | None = None,
        input: dict | None = None,
        output: dict | None = None,
        status: str = "ok",
        error: str | None = None,
    ) -> str:
        event_id = self._next_id()
        self.events.append({
            "case_id": case_id,
            "event_id": event_id,
            "parent_event_id": parent_event_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "model": self.model,
            "tool": tool,
            "input": input,
            "output": output,
            "status": status,
            "error": error,
        })
        return event_id

    def log_model_call(self, case_id: str, parent_event_id: str | None, prompt: dict, response: dict) -> str:
        return self.log(case_id, "model_call", parent_event_id, input=prompt, output=response)

    def log_tool_call(self, case_id: str, parent_event_id: str | None, tool: str, args: dict, result: dict) -> str:
        return self.log(case_id, "tool_call", parent_event_id, tool=tool, input=args, output=result)

    def log_decision(self, case_id: str, parent_event_id: str | None, decision: str, answer: str, evidence: list) -> str:
        return self.log(
            case_id, "decision", parent_event_id,
            output={"decision": decision, "answer": answer, "evidence": evidence},
        )

    def log_error(self, case_id: str, parent_event_id: str | None, error: str, event_type: str = "error") -> str:
        return self.log(case_id, event_type, parent_event_id, status="error", error=error)

    def write(self, path: Path) -> None:
        with gzip.open(path, "wt", encoding="utf-8") as f:
            for event in self.events:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")