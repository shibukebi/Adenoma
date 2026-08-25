import csv
import json
import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from adenoma_agent.adapters.manifest import AdenomaManifestAdapter
from adenoma_agent.dashboard import export_dashboard_batch_from_harness_run
from adenoma_agent.orchestrator import AdenomaAgentOrchestrator
from adenoma_agent.schemas import CaseSpec
from adenoma_agent.utils import ensure_dir, read_json, read_jsonl, sha1_payload, write_json


def discover_grid_cases(wsi_dir, grid_dir, case_ids=None):
    wsi_dir = Path(wsi_dir)
    grid_dir = Path(grid_dir)
    requested = set(case_ids or [])
    cases = []
    for slide_path in sorted(wsi_dir.glob("*.svs")):
        case_id = slide_path.stem
        if requested and case_id not in requested:
            continue
        grid_thumbnail = grid_dir / "{0}_tissuegrid125_ds32_level2_grid.jpg".format(case_id)
        grid_metadata = grid_dir / "{0}_tissuegrid125_ds32_level2_grid.json".format(case_id)
        if not grid_thumbnail.exists() or not grid_metadata.exists():
            continue
        cases.append(
            {
                "case_id": case_id,
                "slide_path": str(slide_path),
                "grid_thumbnail_path": str(grid_thumbnail),
                "grid_metadata_path": str(grid_metadata),
            }
        )
    return cases


def build_grid_case_spec(bundle, item):
    case_id = item["case_id"]
    label = None
    serrated_target = None
    abnormal_crypt_target = None
    dysplasia_proxy_target = None
    metadata = {}
    question = (
        "Review this whole-slide image through a dual-branch colorectal polyp workflow. "
        "First identify reviewable mucosa, then route regions into the SSL pathway, conventional adenoma pathway, inflammatory polyp pathway, or low-value background. "
        "For SSL candidates, assess abnormal crypt architecture before SSL-branch dysplasia. For conventional adenoma candidates, "
        "assess conventional adenoma architecture and then conventional-branch dysplasia. Keep the two dysplasia sources separate."
    )
    try:
        adapter = AdenomaManifestAdapter(
            bundle["runtime"]["data"]["manifest_csv"],
            bundle["runtime"]["data"]["labels_csv"],
            serrated_labels=bundle["runtime"]["data"]["serrated_labels"],
            abnormal_crypt_positive_labels=bundle["runtime"]["data"]["abnormal_crypt_positive_labels"],
            dysplasia_positive_grades=bundle["runtime"]["data"]["dysplasia_positive_grades"],
        )
        base_case = adapter.get_case(case_id)
        label = base_case.label
        serrated_target = base_case.serrated_target
        abnormal_crypt_target = base_case.abnormal_crypt_target
        dysplasia_proxy_target = base_case.dysplasia_proxy_target
        question = base_case.question
        metadata = dict(base_case.metadata)
    except Exception:
        pass
    return CaseSpec(
        case_id=case_id,
        slide_path=item["slide_path"],
        task_type="ssl_others_dual_branch_cpathagent_grid",
        question=question,
        input_mode="grid_thumbnail",
        grid_thumbnail_path=item["grid_thumbnail_path"],
        grid_metadata_path=item["grid_metadata_path"],
        overview_thumbnail_path=None,
        label=label,
        serrated_target=serrated_target,
        abnormal_crypt_target=abnormal_crypt_target,
        dysplasia_proxy_target=dysplasia_proxy_target,
        metadata={
            **metadata,
            "grid_thumbnail_path": item["grid_thumbnail_path"],
            "grid_metadata_path": item["grid_metadata_path"],
        },
    )


class LocalServiceManager(object):
    def __init__(self, project_root, runtime, output_root):
        self.project_root = Path(project_root)
        self.runtime = runtime
        self.output_root = Path(output_root)
        self.processes = []
        self.records = []

    def _wait_ready(self, url, timeout_seconds):
        started = time.time()
        while time.time() - started < timeout_seconds:
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    if int(response.status) < 500:
                        return True
            except Exception:
                time.sleep(2)
        return False

    def maybe_start(self, enabled=False):
        if not enabled:
            self.records.append({"service": "service_manager", "status": "skipped", "reason": "auto_start_disabled"})
            return self.records
        self._start_qwen()
        self._start_chief()
        return self.records

    def _start_qwen(self):
        cfg = self.runtime.get("backends", {}).get("local_cpathagent_qwen", {})
        url = str(cfg.get("server_url", "http://127.0.0.1:18000/predict")).replace("/predict", "/docs")
        if self._wait_ready(url, 1):
            self.records.append(
                {
                    "service": "pathreasoner_qwen",
                    "status": "reused_existing_ready",
                    "pid": None,
                    "url": url,
                    "startup_ms": 0,
                    "model_path": cfg.get("model_name") or self.runtime.get("models", {}).get("cpathagent_qwen_model_id"),
                    "adapter_path": self.runtime.get("models", {}).get("cpathagent_qwen_adapter_path"),
                    "gpu": self.runtime.get("execution", {}).get("cpathagent_qwen_cuda_visible_devices", "unknown"),
                }
            )
            return
        env = os.environ.copy()
        env.update(
            {
                "QWEN_SERVER_PYTHON": self.runtime["paths"].get("cpathagent_qwen_python", env.get("PYTHON", "python3")),
                "QWEN_SERVER_HOST": "127.0.0.1",
                "QWEN_SERVER_PORT": str(url.split(":")[-1].split("/")[0]),
                "QWEN_SERVER_GPU": str(self.runtime.get("execution", {}).get("cpathagent_qwen_cuda_visible_devices", "0")),
                "QWEN_MODEL_ID": self.runtime["models"].get("cpathagent_qwen_model_id", str(self.project_root / "models/Qwen2.5-VL-7B-Instruct")),
                "QWEN_ADAPTER_PATH": self.runtime["models"].get("cpathagent_qwen_adapter_path", str(self.project_root / "models/PathReasoner-R1")),
            }
        )
        started = time.time()
        process = subprocess.Popen(["bash", str(self.project_root / "scripts/start_qwen_model_server.sh")], cwd=str(self.project_root), env=env)
        self.processes.append(process)
        ready = self._wait_ready(url, 240)
        self.records.append(
            {
                "service": "pathreasoner_qwen",
                "status": "ready" if ready else "not_ready",
                "pid": process.pid,
                "url": url,
                "startup_ms": int(round((time.time() - started) * 1000)),
                "model_path": env["QWEN_MODEL_ID"],
                "adapter_path": env["QWEN_ADAPTER_PATH"],
                "gpu": env["QWEN_SERVER_GPU"],
            }
        )

    def _start_chief(self):
        cfg = self.runtime.get("chief_llm", {})
        url = str(cfg.get("server_url", "http://127.0.0.1:18100/predict")).replace("/predict", "/docs")
        configured_name = str(cfg.get("model_name", ""))
        model_path = str(self.project_root / "models/DeepSeek-R1-Distill-Qwen-14B")
        if self._wait_ready(url, 1):
            self.records.append(
                {
                    "service": "deepseek_chief",
                    "status": "reused_existing_ready",
                    "pid": None,
                    "url": url,
                    "startup_ms": 0,
                    "model_path": model_path,
                    "configured_model_name": configured_name,
                    "model_downgraded": bool(configured_name and "32B" in configured_name and "14B" in model_path),
                    "gpu": cfg.get("cuda_visible_devices", "unknown"),
                }
            )
            return
        env = os.environ.copy()
        env.update(
            {
                "CHIEF_SERVER_PYTHON": self.runtime["paths"].get("chief_python", env.get("PYTHON", "python3")),
                "CHIEF_SERVER_HOST": "127.0.0.1",
                "CHIEF_SERVER_PORT": str(url.split(":")[-1].split("/")[0]),
                "CHIEF_SERVER_GPU": str(cfg.get("cuda_visible_devices", "1,2")),
                "CHIEF_MODEL_PATH": model_path,
                "CHIEF_FALLBACK_MODEL_PATH": model_path,
                "CHIEF_MAX_NEW_TOKENS": str(cfg.get("max_new_tokens", 2048)),
                "CHIEF_MAX_MODEL_LEN": str(cfg.get("max_model_len", 8192)),
                "CHIEF_LOG_PATH": str(self.output_root / "chief_model_server.log"),
            }
        )
        started = time.time()
        process = subprocess.Popen(["bash", str(self.project_root / "scripts/start_chief_model_server.sh")], cwd=str(self.project_root), env=env)
        self.processes.append(process)
        ready = self._wait_ready(url, 300)
        self.records.append(
            {
                "service": "deepseek_chief",
                "status": "ready" if ready else "not_ready",
                "pid": process.pid,
                "url": url,
                "startup_ms": int(round((time.time() - started) * 1000)),
                "model_path": model_path,
                "configured_model_name": configured_name,
                "model_downgraded": bool(configured_name and "32B" in configured_name and "14B" in model_path),
                "gpu": env["CHIEF_SERVER_GPU"],
                "log_path": env["CHIEF_LOG_PATH"],
            }
        )

    def stop(self):
        stopped = []
        for process in self.processes:
            if process.poll() is not None:
                stopped.append({"pid": process.pid, "status": "already_exited"})
                continue
            try:
                process.terminate()
                process.wait(timeout=20)
                stopped.append({"pid": process.pid, "status": "terminated"})
            except Exception:
                try:
                    os.kill(process.pid, signal.SIGKILL)
                except Exception:
                    pass
                stopped.append({"pid": process.pid, "status": "killed"})
        return stopped


class DashboardFrontendManager(object):
    def __init__(self, project_root, dashboard_dir, host="127.0.0.1", port=8000, log_dir=None):
        self.project_root = Path(project_root)
        self.dashboard_dir = Path(dashboard_dir)
        self.host = str(host or "127.0.0.1")
        self.port = int(port)
        self.log_dir = Path(log_dir) if log_dir else self.project_root / "artifacts/.dashboard_server_logs"
        self.records = []

    def _health_url(self):
        host = "127.0.0.1" if self.host in {"0.0.0.0", "::"} else self.host
        return "http://{0}:{1}/".format(host, self.port)

    def _pid_files_for_port(self):
        if not self.log_dir.exists():
            return []
        suffix = "_{0}.pid".format(self.port)
        return sorted(path for path in self.log_dir.glob("*.pid") if path.name.endswith(suffix))

    def stop_existing_on_port(self):
        stopped = []
        known_pids = set()
        for pid_file in self._pid_files_for_port():
            try:
                pid_text = pid_file.read_text(encoding="utf-8").strip()
                pid = int(pid_text)
            except Exception:
                pid = None
            if not pid:
                try:
                    pid_file.unlink()
                except Exception:
                    pass
                stopped.append({"pid_file": str(pid_file), "status": "removed_invalid_pid_file"})
                continue
            known_pids.add(pid)
            try:
                os.kill(pid, 0)
            except Exception:
                try:
                    pid_file.unlink()
                except Exception:
                    pass
                stopped.append({"pid": pid, "pid_file": str(pid_file), "status": "stale_pid_removed"})
                continue
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(1)
                try:
                    os.kill(pid, 0)
                    os.kill(pid, signal.SIGKILL)
                    status = "killed"
                except Exception:
                    status = "terminated"
                pid_file.unlink(missing_ok=True)
                stopped.append({"pid": pid, "pid_file": str(pid_file), "status": status})
            except Exception as exc:
                stopped.append({"pid": pid, "pid_file": str(pid_file), "status": "stop_failed", "error": str(exc)})
        for pid in self._listening_pids_on_port():
            if pid in known_pids:
                continue
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(1)
                try:
                    os.kill(pid, 0)
                    os.kill(pid, signal.SIGKILL)
                    status = "killed_untracked_port_owner"
                except Exception:
                    status = "terminated_untracked_port_owner"
                stopped.append({"pid": pid, "port": self.port, "status": status})
            except Exception as exc:
                stopped.append({"pid": pid, "port": self.port, "status": "stop_untracked_port_owner_failed", "error": str(exc)})
        return stopped

    def _listening_pids_on_port(self):
        pids = set()
        try:
            completed = subprocess.run(
                ["ss", "-ltnp"],
                text=True,
                capture_output=True,
                timeout=5,
            )
            pid_pattern = re.compile(r"pid=(\d+)")
            port_tokens = {
                ":{0} ".format(self.port),
                ":{0}\t".format(self.port),
                ":{0}\n".format(self.port),
            }
            for line in completed.stdout.splitlines():
                if not any(token in line for token in port_tokens):
                    continue
                for match in pid_pattern.finditer(line):
                    pids.add(int(match.group(1)))
        except Exception:
            pass
        return sorted(pids)

    def _wait_ready(self, timeout_seconds=10):
        started = time.time()
        url = self._health_url()
        last_error = ""
        while time.time() - started < timeout_seconds:
            try:
                request = urllib.request.Request(url, method="HEAD")
                with urllib.request.urlopen(request, timeout=2) as response:
                    return {
                        "ready": int(response.status) < 500,
                        "status_code": int(response.status),
                        "url": url,
                    }
            except urllib.error.HTTPError as exc:
                return {"ready": int(exc.code) < 500, "status_code": int(exc.code), "url": url}
            except Exception as exc:
                last_error = str(exc)
                time.sleep(1)
        return {"ready": False, "url": url, "error": last_error}

    def switch_to(self, stop_existing=True):
        ensure_dir(self.log_dir)
        record = {
            "service": "dashboard_frontend",
            "dashboard_dir": str(self.dashboard_dir),
            "host": self.host,
            "port": self.port,
            "url": self._health_url(),
            "stopped_existing": [],
        }
        if stop_existing:
            record["stopped_existing"] = self.stop_existing_on_port()
        env = os.environ.copy()
        env.update(
            {
                "DASHBOARD_DIR": str(self.dashboard_dir),
                "DASHBOARD_HOST": self.host,
                "DASHBOARD_PORT": str(self.port),
                "DASHBOARD_LOG_DIR": str(self.log_dir),
                "DASHBOARD_SERVER_MODE": env.get("DASHBOARD_SERVER_MODE", "tmux-python"),
            }
        )
        command = ["bash", str(self.project_root / "scripts/start_agent_dashboard_server.sh"), "start"]
        started = time.time()
        completed = subprocess.run(
            command,
            cwd=str(self.project_root),
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
        )
        record.update(
            {
                "status": "started" if completed.returncode == 0 else "start_failed",
                "returncode": completed.returncode,
                "startup_ms": int(round((time.time() - started) * 1000)),
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "health": self._wait_ready(10),
            }
        )
        self.records.append(record)
        return record


def publish_harness_dashboard(run_dir, output_dir, project_root=None, host="127.0.0.1", port=8000, switch_frontend=False, stop_existing=True):
    run_dir = Path(run_dir).resolve()
    output_dir = ensure_dir(Path(output_dir).resolve())
    project_root = Path(project_root or Path.cwd())
    export_dashboard_batch_from_harness_run(run_dir, output_dir)
    payload = {
        "dashboard_dir": str(output_dir),
        "index_html": str(output_dir / "index.html"),
        "harness_run_dir": str(run_dir),
        "frontend": None,
    }
    if switch_frontend:
        manager = DashboardFrontendManager(project_root, output_dir, host=host, port=port)
        payload["frontend"] = manager.switch_to(stop_existing=stop_existing)
    write_json(output_dir / "dashboard_status.json", payload)
    return payload


def summarize_case_artifacts(case_dir):
    case_dir = Path(case_dir)
    row = {
        "case_id": case_dir.name,
        "case_dir": str(case_dir),
        "final_status": "failed",
        "recovery_status": "failed",
        "quality_tier": "failed",
        "fallback_stages": [],
        "normalization_count": 0,
        "normalization_actions": [],
        "repair_count": 0,
        "repair_stages": [],
        "repair_actions": [],
        "failed_stage_history": [],
        "attempt_count": 0,
        "case_result_path": "",
        "total_runtime_ms": 0,
    }
    result_path = case_dir / "case_result.json"
    if result_path.exists():
        result = read_json(result_path)
        row["final_status"] = result.get("status", "ok")
        row["recovery_status"] = "ok" if result.get("status") in {"ok", "warn"} else "failed"
        row["case_result_path"] = str(result_path)
        row["total_runtime_ms"] = result.get("timing", {}).get("total_runtime_ms", 0)
    events_path = case_dir / "events.jsonl"
    if events_path.exists():
        for event in read_jsonl(events_path):
            if event.get("status") == "error":
                row["failed_stage_history"].append(
                    {
                        "state": event.get("state"),
                        "agent": event.get("agent"),
                        "payload": event.get("payload", {}),
                    }
                )
    for path in [
        case_dir / "navigation/navigation_steps.json",
        case_dir / "observe/pathological_report.json",
        case_dir / "trace/trace_backend_attempts.json",
    ]:
        if not path.exists():
            continue
        payload = read_json(path)
        attempts = payload.get("backend_attempts") or payload.get("attempts") or []
        row["attempt_count"] += len(attempts)
        for attempt in attempts:
            if attempt.get("backend") == "deepseek_output_repair":
                row["repair_count"] += 1
                stage_name = attempt.get("repair_stage") or ("observe_report" if path.name == "pathological_report.json" else path.parent.name)
                if stage_name not in row["repair_stages"]:
                    row["repair_stages"].append(stage_name)
                for action in attempt.get("repair_actions", []) or []:
                    if action not in row["repair_actions"]:
                        row["repair_actions"].append(action)
            if attempt.get("backend") == "heuristic" and attempt.get("status") == "ok":
                stage = "observe_report" if path.name == "pathological_report.json" else path.parent.name
                if stage not in row["fallback_stages"]:
                    row["fallback_stages"].append(stage)
    chief_dir = case_dir / "observe/chief_reviews"
    if chief_dir.exists():
        for response_path in sorted(chief_dir.glob("*_chief_response.json")):
            payload = read_json(response_path)
            row["attempt_count"] += 1
            actions = []
            if payload.get("parse_error"):
                actions.append("chief_parse_error")
            raw = str(payload.get("raw_generated_text", ""))
            candidate = str(payload.get("answer_candidate_text", ""))
            if raw and candidate and raw != candidate:
                actions.append("chief_extracted_answer_candidate")
            if payload.get("review_source") != "chief_model":
                actions.append("chief_non_model_source")
            if payload.get("chief_confidence") == 0.5 or str(payload.get("review_id", "")).startswith("global_review_"):
                actions.append("chief_normalized_or_defaulted_fields")
            for action in actions:
                if action not in row["normalization_actions"]:
                    row["normalization_actions"].append(action)
            row["normalization_count"] += len(actions)
    if row["recovery_status"] != "ok":
        row["quality_tier"] = "failed"
    elif row["fallback_stages"]:
        row["quality_tier"] = "partial_fallback"
    elif row["repair_count"] > 0:
        row["quality_tier"] = "repaired_real_model"
    elif row["normalization_count"] > 0:
        row["quality_tier"] = "normalized_real_model"
    else:
        row["quality_tier"] = "clean_real_model"
    if result_path.exists():
        result = read_json(result_path)
        metadata = dict(result.get("metadata", {}))
        metadata["harness_quality"] = {
            "quality_tier": row["quality_tier"],
            "recovery_status": row["recovery_status"],
            "fallback_stages": row["fallback_stages"],
            "normalization_count": row["normalization_count"],
            "normalization_actions": row["normalization_actions"],
            "repair_count": row["repair_count"],
            "repair_stages": row["repair_stages"],
            "repair_actions": row["repair_actions"],
            "failed_stage_count": len(row["failed_stage_history"]),
            "attempt_count": row["attempt_count"],
        }
        result["metadata"] = metadata
        write_json(result_path, result)
    return row


def write_experiment_summaries(output_root, rows, service_records, runtime_payload, budget_payload):
    output_root = Path(output_root)
    quality_counts = {}
    for row in rows:
        quality_counts[row["quality_tier"]] = quality_counts.get(row["quality_tier"], 0) + 1
    summary = {
        "output_root": str(output_root),
        "case_count": len(rows),
        "completed_count": len([row for row in rows if row["recovery_status"] == "ok"]),
        "failed_count": len([row for row in rows if row["recovery_status"] != "ok"]),
        "quality_counts": quality_counts,
        "runtime_hash": sha1_payload(runtime_payload),
        "budget_hash": sha1_payload(budget_payload),
        "cases": rows,
    }
    write_json(output_root / "experiment_summary.json", summary)
    write_json(output_root / "quality_summary.json", {"quality_counts": quality_counts, "cases": rows})
    write_json(output_root / "service_status.json", {"services": service_records})
    tsv_path = output_root / "experiment_summary.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "final_status",
                "recovery_status",
                "quality_tier",
                "fallback_stages",
                "normalization_count",
                "repair_count",
                "repair_stages",
                "attempt_count",
                "total_runtime_ms",
                "case_result_path",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "case_id": row["case_id"],
                    "final_status": row["final_status"],
                    "recovery_status": row["recovery_status"],
                    "quality_tier": row["quality_tier"],
                    "fallback_stages": ",".join(row["fallback_stages"]),
                    "normalization_count": row["normalization_count"],
                    "repair_count": row["repair_count"],
                    "repair_stages": ",".join(row["repair_stages"]),
                    "attempt_count": row["attempt_count"],
                    "total_runtime_ms": row["total_runtime_ms"],
                    "case_result_path": row["case_result_path"],
                }
            )
    return summary


def run_grid_batch_experiment(
    bundle,
    wsi_dir,
    grid_dir,
    output_root,
    case_ids=None,
    auto_start_services=False,
    dashboard_output_dir=None,
    switch_dashboard=False,
    dashboard_host="127.0.0.1",
    dashboard_port=8000,
):
    output_root = ensure_dir(output_root)
    runtime_payload = bundle.get("runtime", {})
    budget_payload = bundle.get("budget", {})
    project_root = runtime_payload.get("project", {}).get("project_root") or Path.cwd()
    manager = LocalServiceManager(project_root, runtime_payload, output_root)
    service_records = []
    rows = []
    try:
        service_records.extend(manager.maybe_start(enabled=auto_start_services))
        cases = discover_grid_cases(wsi_dir, grid_dir, case_ids=case_ids)
        orchestrator = AdenomaAgentOrchestrator(bundle)
        for item in cases:
            case_spec = build_grid_case_spec(bundle, item)
            case_dir = Path(output_root) / case_spec.case_id
            try:
                orchestrator.run_case(case_spec, output_root)
            except Exception as exc:
                ensure_dir(case_dir)
                write_json(case_dir / "case_failure.json", {"case_id": case_spec.case_id, "error": str(exc)})
            rows.append(summarize_case_artifacts(case_dir))
    finally:
        if auto_start_services:
            service_records.append({"service": "service_manager_stop", "stopped": manager.stop()})
    summary = write_experiment_summaries(output_root, rows, service_records, runtime_payload, budget_payload)
    if dashboard_output_dir:
        dashboard_status = publish_harness_dashboard(
            run_dir=output_root,
            output_dir=dashboard_output_dir,
            project_root=project_root,
            host=dashboard_host,
            port=dashboard_port,
            switch_frontend=switch_dashboard,
        )
        summary["dashboard"] = dashboard_status
        write_json(Path(output_root) / "experiment_summary.json", summary)
    return summary
