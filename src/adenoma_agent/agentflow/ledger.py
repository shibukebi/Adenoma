import json
from pathlib import Path
from typing import Iterable, Optional

from adenoma_agent.agentflow.contracts import EvidenceRecord, LedgerSnapshot


class StaleSnapshotError(RuntimeError):
    pass


class DuplicateEvidenceError(ValueError):
    pass


class EvidenceLedger(object):
    """Append-only evidence store with immutable, replayable snapshots."""

    def __init__(self, case_id, jsonl_path=None, replay=True):
        self.case_id = str(case_id)
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        self._records = []
        self._by_id = {}
        self._version = 0
        self._created_after_action_id = None
        if self.jsonl_path and replay and self.jsonl_path.exists():
            self._replay(self.jsonl_path)

    @property
    def current_snapshot_id(self):
        return "ledger_v{0:06d}".format(self._version)

    def snapshot(self):
        # Reconstruct records so callers cannot mutate the ledger through nested
        # metadata dictionaries held by a previously returned object.
        records = tuple(EvidenceRecord.from_dict(record.to_dict()) for record in self._records)
        return LedgerSnapshot(
            snapshot_id=self.current_snapshot_id,
            case_id=self.case_id,
            records=records,
            created_after_action_id=self._created_after_action_id,
        )

    def assert_current(self, snapshot_id):
        if str(snapshot_id) != self.current_snapshot_id:
            raise StaleSnapshotError(
                "Plan was built from {0}; current ledger snapshot is {1}".format(
                    snapshot_id,
                    self.current_snapshot_id,
                )
            )

    def append(self, record, created_after_action_id=None):
        return self.append_many([record], created_after_action_id=created_after_action_id)

    def append_many(self, records, created_after_action_id=None):
        records = [self._coerce_record(record) for record in records]
        if not records:
            return self.snapshot()
        new_records = []
        seen_in_batch = {}
        for record in records:
            if record.case_id != self.case_id:
                raise ValueError(
                    "Evidence case_id {0} does not match ledger case_id {1}".format(record.case_id, self.case_id)
                )
            existing = self._by_id.get(record.evidence_id) or seen_in_batch.get(record.evidence_id)
            if existing is not None:
                if existing.to_dict() != record.to_dict():
                    raise DuplicateEvidenceError(
                        "evidence_id {0} already exists with different content".format(record.evidence_id)
                    )
                continue
            seen_in_batch[record.evidence_id] = record
            new_records.append(record)
        if not new_records:
            return self.snapshot()
        next_version = self._version + 1
        if self.jsonl_path:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self.jsonl_path.open("a", encoding="utf-8") as handle:
                for record in new_records:
                    envelope = {
                        "ledger_schema_version": "evidence_ledger_jsonl_v1",
                        "case_id": self.case_id,
                        "snapshot_version": next_version,
                        "created_after_action_id": created_after_action_id,
                        "record": record.to_dict(),
                    }
                    handle.write(json.dumps(envelope, ensure_ascii=False, sort_keys=True) + "\n")
        for record in new_records:
            self._records.append(record)
            self._by_id[record.evidence_id] = record
        self._version = next_version
        self._created_after_action_id = created_after_action_id
        return self.snapshot()

    def _coerce_record(self, record):
        if isinstance(record, EvidenceRecord):
            return record
        if isinstance(record, dict):
            return EvidenceRecord.from_dict(record)
        raise TypeError("Expected EvidenceRecord or dict, got {0}".format(type(record).__name__))

    def _replay(self, path):
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                envelope = json.loads(line)
                if envelope.get("case_id") != self.case_id:
                    raise ValueError("Ledger case mismatch at line {0}".format(line_number))
                record = EvidenceRecord.from_dict(envelope["record"])
                existing = self._by_id.get(record.evidence_id)
                if existing is not None:
                    if existing.to_dict() != record.to_dict():
                        raise DuplicateEvidenceError(
                            "Conflicting duplicate evidence at line {0}".format(line_number)
                        )
                    continue
                self._records.append(record)
                self._by_id[record.evidence_id] = record
                self._version = max(self._version, int(envelope.get("snapshot_version", 0)))
                self._created_after_action_id = envelope.get("created_after_action_id")

    @classmethod
    def from_jsonl(cls, case_id, jsonl_path):
        return cls(case_id=case_id, jsonl_path=jsonl_path, replay=True)
