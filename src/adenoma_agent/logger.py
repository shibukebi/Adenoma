from pathlib import Path

from adenoma_agent.utils import append_jsonl, ensure_dir, now_iso


class JsonlLogger(object):
    def __init__(self, case_id, log_path):
        self.case_id = case_id
        self.log_path = Path(log_path)
        ensure_dir(self.log_path.parent)
        self.log_path.write_text("", encoding="utf-8")

    def log(
        self,
        state,
        agent,
        input_ref=None,
        output_ref=None,
        latency_ms=None,
        status="ok",
        payload=None,
    ):
        event = {
            "timestamp": now_iso(),
            "case_id": self.case_id,
            "state": state,
            "agent": agent,
            "input_ref": input_ref,
            "output_ref": output_ref,
            "latency_ms": latency_ms,
            "status": status,
            "payload": payload or {},
        }
        append_jsonl(self.log_path, event)
        return event
