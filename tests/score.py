#!/usr/bin/env python3
"""Compute a normalized weighted reward from a pytest JUnit report.

Copy this file into a task package as ``tests/score.py``. The package keeps a
``tests/scoring.json`` file that maps pytest cases to non-negative weights and
declares one or more natural completion gates. Gate failures produce reward 0;
completed runs with passing gates receive the weighted pass fraction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


CONFIG_SCHEMA_VERSION = 1
WEIGHT_TOLERANCE = 1e-9
WEIGHT_POLICY_VERSION = 1
PRIORITIES = {"core", "supporting", "boundary", "recovery", "schema"}
CORE_MIN_SHARE = 0.5
GENERIC_RATIONALES = {
    "该用例验证题面明确要求且可由参考实现独立满足的可观察行为。",
    "验证题面要求。",
}


class ScoringError(ValueError):
    """The scoring configuration or JUnit report is not trustworthy."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ScoringError(f"scoring config contains duplicate key: {key}")
        payload[key] = value
    return payload


def load_scoring_config(path: Path) -> dict[str, Any]:
    """Load and validate the package scoring contract."""
    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except OSError as exc:
        raise ScoringError(f"cannot read scoring config: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ScoringError(f"invalid scoring JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise ScoringError("scoring config must be a JSON object")
    if raw.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ScoringError(
            f"scoring config schema_version must be {CONFIG_SCHEMA_VERSION}"
        )
    policy = raw.get("weight_policy")
    if not isinstance(policy, dict):
        raise ScoringError(
            "weight_policy is required and must use business_priority weighting"
        )
    if policy.get("version") != WEIGHT_POLICY_VERSION:
        raise ScoringError(
            f"weight_policy.version must be {WEIGHT_POLICY_VERSION}"
        )
    if policy.get("basis") != "business_priority":
        raise ScoringError("weight_policy.basis must be business_priority")
    if policy.get("core_min_share") != CORE_MIN_SHARE:
        raise ScoringError("weight_policy.core_min_share must be 0.5")

    gate_tests = raw.get("gate_tests")
    tests = raw.get("tests")
    if not isinstance(gate_tests, list) or not gate_tests:
        raise ScoringError("gate_tests must be a non-empty list")
    if not all(isinstance(item, str) and item.strip() for item in gate_tests):
        raise ScoringError("gate_tests entries must be non-empty strings")
    if len(set(gate_tests)) != len(gate_tests):
        raise ScoringError("gate_tests must not contain duplicates")
    if not isinstance(tests, dict) or not tests:
        raise ScoringError("tests must be a non-empty object")
    if set(gate_tests) & set(tests):
        raise ScoringError("a test cannot be both a gate and a scored test")

    normalized: dict[str, dict[str, Any]] = {}
    for case_id, item in tests.items():
        if not isinstance(case_id, str) or "::" not in case_id:
            raise ScoringError(
                "scored test IDs must use '<file>.py::<test>' or include a class"
            )
        if not isinstance(item, dict):
            raise ScoringError(f"scored test entry must be an object: {case_id}")
        weight = item.get("weight")
        rationale = item.get("rationale")
        priority = item.get("priority")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ScoringError(f"weight must be numeric: {case_id}")
        weight = float(weight)
        if not math.isfinite(weight) or weight <= 0.0 or weight > 1.0:
            raise ScoringError(f"weight must be in (0, 1]: {case_id}")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ScoringError(f"weight rationale is required: {case_id}")
        rationale = rationale.strip()
        if len(rationale) < 20:
            raise ScoringError(
                f"weight rationale must describe the business impact in at least 20 characters: {case_id}"
            )
        if rationale in GENERIC_RATIONALES:
            raise ScoringError(
                f"weight rationale is too generic to justify business weighting: {case_id}"
            )
        if priority not in PRIORITIES:
            raise ScoringError(
                f"priority must be one of {', '.join(sorted(PRIORITIES))}: {case_id}"
            )
        normalized[case_id] = {
            "weight": weight,
            "priority": priority,
            "rationale": rationale,
        }

    total = math.fsum(item["weight"] for item in normalized.values())
    if abs(total - 1.0) > WEIGHT_TOLERANCE:
        raise ScoringError(
            f"scored-test weights must sum to 1, got {total:.12f}"
        )
    core = [item["weight"] for item in normalized.values() if item["priority"] == "core"]
    non_core = [item["weight"] for item in normalized.values() if item["priority"] != "core"]
    if not core:
        raise ScoringError("at least one scored test must have priority=core")
    if not non_core:
        raise ScoringError("at least one non-core test is required to prove differentiated weighting")
    core_share = math.fsum(core)
    if core_share + WEIGHT_TOLERANCE < CORE_MIN_SHARE:
        raise ScoringError(
            f"core business behaviors must receive at least {CORE_MIN_SHARE:.2f} weight, got {core_share:.6f}"
        )
    if min(core) <= max(non_core) + WEIGHT_TOLERANCE:
        raise ScoringError(
            "every core test weight must be strictly greater than every non-core test weight"
        )
    if any(abs(round(item["weight"], 3) - item["weight"]) > WEIGHT_TOLERANCE for item in normalized.values()):
        raise ScoringError("test weights must use at most three decimal places")

    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "weight_policy": {
            "version": WEIGHT_POLICY_VERSION,
            "basis": "business_priority",
            "core_min_share": CORE_MIN_SHARE,
        },
        "gate_tests": list(gate_tests),
        "tests": normalized,
    }


def _canonical_case_id(testcase: ET.Element) -> str:
    """Build a stable ID from pytest's JUnit attributes.

    pytest normally emits ``classname='test_outputs'`` and
    ``name='test_example'``, or ``classname='test_outputs.TestCase'`` for a
    method. If a ``file`` attribute is present, prefer it and recover nested
    classes from ``classname``.
    """
    name = testcase.get("name", "").strip()
    classname = testcase.get("classname", "").strip()
    file_attr = testcase.get("file", "").strip()
    if not name:
        raise ScoringError("JUnit testcase is missing its name")

    class_parts: list[str] = []
    if classname:
        parts = [part for part in classname.split(".") if part]
        class_index = next(
            (index for index, part in enumerate(parts) if part[:1].isupper()),
            None,
        )
        if class_index is None:
            module_parts = parts
        else:
            module_parts = parts[:class_index]
            class_parts = parts[class_index:]
    else:
        module_parts = []

    if file_attr:
        file_path = Path(file_attr).name
        if not file_path.endswith(".py"):
            file_path += ".py"
    elif module_parts:
        file_path = module_parts[-1] + ".py"
    else:
        raise ScoringError(
            f"cannot determine source file for JUnit testcase {name!r}"
        )

    return "::".join([file_path, *class_parts, name])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _declared_count(suites: list[ET.Element], attribute: str) -> int:
    total = 0
    for suite in suites:
        raw = suite.get(attribute)
        if raw is None:
            raise ScoringError(f"JUnit testsuite is missing {attribute}")
        try:
            value = int(raw)
        except ValueError as exc:
            raise ScoringError(f"JUnit testsuite {attribute} is not an integer") from exc
        if value < 0:
            raise ScoringError(f"JUnit testsuite {attribute} cannot be negative")
        total += value
    return total


def validate_provenance(
    provenance_path: Path,
    junit_path: Path,
    config_path: Path,
    tests_path: Path,
) -> dict[str, int | str]:
    """Validate that the accepted report came from the protected verifier run."""
    try:
        payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ScoringError(f"cannot read verifier provenance: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ScoringError("verifier provenance schema is invalid")
    report_stat = junit_path.stat()
    verifier_uid = os.getuid()
    if report_stat.st_uid != verifier_uid:
        raise ScoringError("JUnit report must be owned by the active verifier user")
    if stat.S_IMODE(report_stat.st_mode) != 0o600:
        raise ScoringError("JUnit report must use mode 0600")
    checks = {
        "report_sha256": _sha256(junit_path),
        "report_size": report_stat.st_size,
        "config_sha256": _sha256(config_path),
        "tests_sha256": _sha256(tests_path),
    }
    for key, expected in checks.items():
        if payload.get(key) != expected:
            raise ScoringError(f"verifier provenance mismatch: {key}")
    if payload.get("verifier_uid") != verifier_uid:
        raise ScoringError("verifier uid does not match the active scorer")
    if payload.get("pytest_uid") != verifier_uid:
        raise ScoringError("pytest uid does not match the active verifier")
    candidate_uid = payload.get("candidate_uid")
    if not isinstance(candidate_uid, int) or candidate_uid == 0:
        raise ScoringError("candidate must not execute with uid 0")
    if verifier_uid == 0 and candidate_uid == verifier_uid:
        raise ScoringError("root verifier must isolate the candidate uid")
    if payload.get("pytest_returncode") not in (0, 1):
        raise ScoringError("pytest did not complete with a scoreable status")
    return checks


def parse_junit(path: Path) -> dict[str, str]:
    """Return canonical test IDs and pass/fail outcomes after count validation."""
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ScoringError(f"cannot parse JUnit report: {exc}") from exc

    if root.tag not in {"testsuite", "testsuites"}:
        raise ScoringError("JUnit root must be testsuite or testsuites")
    suites = list(root.iter("testsuite"))
    if not suites:
        raise ScoringError("JUnit report contains no testsuite")
    testcases = list(root.iter("testcase"))
    actual_failures = sum(1 for case in testcases if case.find("failure") is not None)
    actual_errors = sum(1 for case in testcases if case.find("error") is not None)
    actual_skipped = sum(1 for case in testcases if case.find("skipped") is not None)
    declared_tests = _declared_count(suites, "tests")
    declared_failures = _declared_count(suites, "failures")
    declared_errors = _declared_count(suites, "errors")
    declared_skipped = _declared_count(suites, "skipped")
    if declared_tests != len(testcases):
        raise ScoringError(
            "JUnit declared test count does not equal executed testcase count"
        )
    if declared_failures != actual_failures:
        raise ScoringError("JUnit declared failure count does not match testcases")
    if declared_errors != actual_errors:
        raise ScoringError("JUnit declared error count does not match testcases")
    if declared_skipped != actual_skipped:
        raise ScoringError("JUnit declared skipped count does not match testcases")

    outcomes: dict[str, str] = {}
    for testcase in testcases:
        case_id = _canonical_case_id(testcase)
        if case_id in outcomes:
            raise ScoringError(f"JUnit report contains duplicate testcase: {case_id}")
        if testcase.find("error") is not None:
            raise ScoringError(f"testcase raised an execution error: {case_id}")
        if testcase.find("skipped") is not None:
            raise ScoringError(f"testcase was skipped: {case_id}")
        if testcase.find("failure") is not None:
            outcomes[case_id] = "fail"
        else:
            outcomes[case_id] = "pass"

    if not outcomes:
        raise ScoringError("JUnit report contains no executed testcases")
    return outcomes


def compute_reward(config: dict[str, Any], outcomes: dict[str, str]) -> dict[str, Any]:
    """Compute the reward and a per-case report from validated inputs."""
    expected = set(config["gate_tests"]) | set(config["tests"])
    observed = set(outcomes)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing or unexpected:
        detail: list[str] = []
        if missing:
            detail.append("missing=" + ", ".join(missing))
        if unexpected:
            detail.append("unexpected=" + ", ".join(unexpected))
        raise ScoringError(
            "JUnit testcase set does not match scoring config: " + "; ".join(detail)
        )

    gate_results = [
        {"id": case_id, "status": outcomes[case_id]}
        for case_id in config["gate_tests"]
    ]
    gate_passed = all(item["status"] == "pass" for item in gate_results)
    scored_results = []
    for case_id, item in config["tests"].items():
        status = outcomes[case_id]
        scored_results.append(
            {
                "id": case_id,
                "status": status,
                "weight": item["weight"],
                "earned": item["weight"] if status == "pass" else 0.0,
                "rationale": item["rationale"],
            }
        )

    if not gate_passed:
        reward = 0.0
    elif all(item["status"] == "pass" for item in scored_results):
        reward = 1.0
    else:
        reward = math.fsum(item["earned"] for item in scored_results)
        reward = min(1.0, max(0.0, round(reward, 3)))

    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "status": "COMPLETED",
        "reward": reward,
        "gate_passed": gate_passed,
        "gate_tests": gate_results,
        "tests": scored_results,
    }


def evaluate(
    config_path: Path,
    junit_path: Path,
    provenance_path: Path | None = None,
    tests_path: Path = Path("/tests/test_outputs.py"),
) -> dict[str, Any]:
    """Load both inputs, validate their exact case set, and score them."""
    config = load_scoring_config(config_path)
    provenance = (
        validate_provenance(provenance_path, junit_path, config_path, tests_path)
        if provenance_path is not None
        else {}
    )
    outcomes = parse_junit(junit_path)
    report = compute_reward(config, outcomes)
    report["executed_case_count"] = len(outcomes)
    report["failed_case_count"] = sum(1 for status in outcomes.values() if status == "fail")
    report["provenance"] = provenance
    return report


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def write_outputs(
    report_out: Path,
    reward_out: Path,
    report: dict[str, Any],
) -> None:
    """Persist a structured report and a three-decimal raw reward."""
    _atomic_write(
        report_out,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(reward_out, f"{float(report['reward']):.3f}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--junit", required=True, type=Path)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--report-out", required=True, type=Path)
    parser.add_argument("--reward-out", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        report = evaluate(
            args.config,
            args.junit,
            args.provenance,
            Path("/tests/test_outputs.py"),
        )
    except ScoringError as exc:
        report = {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "status": "INCOMPLETE",
            "reward": 0.0,
            "error": str(exc),
            "gate_passed": False,
            "gate_tests": [],
            "tests": [],
        }
        write_outputs(args.report_out, args.reward_out, report)
        print(f"INCOMPLETE: {exc}", file=sys.stderr)
        return 2

    write_outputs(args.report_out, args.reward_out, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
