"""Run candidate subprocesses as an unprivileged verifier user."""

from __future__ import annotations

import ast
import os
import re
import subprocess
from pathlib import Path


RUNNER_UID = 10002
RUNNER_GID = 10002
RUNNER_ENV = {
    "HOME": "/tmp",
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUNBUFFERED": "1",
}
_REAL_RUN = subprocess.run
_REAL_POPEN = subprocess.Popen
_CASE_OUTCOMES: dict[str, str] = {}
_CASE_IDS_BY_TEST: dict[str, list[str]] | None = None


def _case_ids_by_test() -> dict[str, list[str]]:
    global _CASE_IDS_BY_TEST
    if _CASE_IDS_BY_TEST is not None:
        return _CASE_IDS_BY_TEST
    source_path = Path(__file__).with_name("test_outputs.py")
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    lines = source.splitlines()
    mapping: dict[str, list[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue
        body = "\n".join(lines[node.lineno - 1 : node.end_lineno])
        values: list[str] = []
        for match in re.finditer(r"#\s*case_id:\s*(.+)", body):
            values.extend(
                item.strip()
                for item in re.split(r"[,，\s]+", match.group(1))
                if item.strip()
            )
        mapping[node.name] = list(dict.fromkeys(values))
    _CASE_IDS_BY_TEST = mapping
    return mapping


def pytest_runtest_logreport(report):
    if report.when not in {"call", "setup"}:
        return
    test_name = report.nodeid.rsplit("::", 1)[-1].split("[", 1)[0]
    if report.failed:
        _CASE_OUTCOMES[test_name] = "FAILED"
    elif report.when == "call" and report.passed:
        _CASE_OUTCOMES[test_name] = "PASSED"


def pytest_sessionfinish(session, exitstatus):
    mapping = _case_ids_by_test()
    for test_name in sorted(_CASE_OUTCOMES):
        outcome = _CASE_OUTCOMES[test_name]
        for case_id in mapping.get(test_name, []):
            print(f"{case_id} {outcome}")


def _argv_list(args):
    if isinstance(args, (str, bytes)):
        return None
    try:
        return list(args)
    except TypeError:
        return None


def _candidate_call(args) -> bool:
    argv = _argv_list(args)
    if argv is None:
        return False
    return any(str(item).startswith("/app/") for item in argv)


def _make_tmp_accessible(args, cwd) -> None:
    paths: list[Path] = []
    argv = _argv_list(args) or []
    for item in argv:
        value = str(item)
        if value.startswith("/tmp/"):
            paths.append(Path(value))
    if cwd is not None:
        paths.append(Path(cwd))
    for candidate in paths:
        if candidate.exists() and candidate.is_file():
            try:
                os.chmod(candidate, 0o666)
            except OSError:
                pass
        current = candidate if candidate.exists() and candidate.is_dir() else candidate.parent
        while str(current).startswith("/tmp") and current.exists():
            try:
                os.chmod(current, 0o777)
            except OSError:
                break
            if current == Path("/tmp"):
                break
            current = current.parent


def _lower_candidate(kwargs) -> None:
    # A root verifier can switch directly without the setgroups call that
    # restricted container runtimes can reject.  A non-root package probe
    # cannot switch uid, so it keeps the effective verifier uid and relies on
    # the read-only tests mount plus process-group cleanup.
    kwargs.pop("extra_groups", None)
    if os.getuid() == 0:
        kwargs["user"] = RUNNER_UID
        kwargs["group"] = RUNNER_GID
    else:
        kwargs.pop("user", None)
        kwargs.pop("group", None)
    kwargs["env"] = RUNNER_ENV.copy()


def _run(args, *p_args, **kwargs):
    if _candidate_call(args):
        _make_tmp_accessible(args, kwargs.get("cwd"))
        _lower_candidate(kwargs)
    return _REAL_RUN(args, *p_args, **kwargs)


def _popen(args, *p_args, **kwargs):
    if _candidate_call(args):
        _make_tmp_accessible(args, kwargs.get("cwd"))
        _lower_candidate(kwargs)
    return _REAL_POPEN(args, *p_args, **kwargs)


subprocess.run = _run
subprocess.Popen = _popen
