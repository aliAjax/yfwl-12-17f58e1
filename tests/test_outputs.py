"""Behavior checks for robust calibration with unlabeled channel transforms."""

from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

import pytest


CANDIDATE = os.environ.get("REGRESSION_ENTRY", "/app/regression_recover.py")
INTERCEPT = 0.65
SLOPE = 1.35
TRANSFORMS = [(0.78 + 0.12 * index, -0.9 + 0.2 * index) for index in range(10)]
PERMUTATION = [4, 0, 9, 2, 7, 1, 8, 3, 6, 5]


def canonical(value: dict) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def base_payload(case_id: str = "regression-case") -> dict:
    batches = []
    for batch_index, transform_index in enumerate(PERMUTATION):
        scale, shift = TRANSFORMS[transform_index]
        points = []
        for point_index in range(14):
            z_value = -1.25 + 2.5 * point_index / 13
            raw_x = (z_value - shift) / scale
            noise = 0.025 * math.sin((batch_index + 1) * (point_index + 2) * 1.7)
            y_value = INTERCEPT + SLOPE * z_value + noise
            if point_index in (3, 11):
                y_value += 0.8 if (batch_index + point_index) % 2 else -0.65
            points.append({"x": round(raw_x, 6), "y": round(y_value, 6)})
        batches.append({"channel_id": f"C{batch_index + 1:02d}", "points": points})
    return {
        "case_id": case_id,
        "max_batches": 10,
        "max_points": 160,
        "transform_catalog": [
            {"label": f"t{index + 1:02d}", "scale": TRANSFORMS[index][0], "shift": TRANSFORMS[index][1]}
            for index in [4, 0, 9, 2, 7, 1, 8, 3, 6, 5]
        ],
        "batches": batches,
    }


def execute(
    tmp_path: Path,
    payload: dict | bytes,
    name: str,
    *,
    old_report: bytes | None = None,
    old_ledger: bytes | None = None,
    report_is_directory: bool = False,
) -> dict:
    folder = tmp_path / name
    folder.mkdir()
    source = folder / "input.json"
    report = folder / "report.json"
    ledger = folder / "ledger.jsonl"
    source.write_bytes(payload if isinstance(payload, bytes) else canonical(payload))
    if old_report is not None:
        report.write_bytes(old_report)
    if old_ledger is not None:
        ledger.write_bytes(old_ledger)
    if report_is_directory:
        report.mkdir()
    process = subprocess.run(
        [sys.executable, CANDIDATE, str(source), str(report), str(ledger)],
        capture_output=True,
        text=True,
        timeout=90,
    )
    return {"folder": folder, "source": source, "report": report, "ledger": ledger, "process": process}


def report_of(case: dict) -> dict:
    return json.loads(case["report"].read_text(encoding="utf-8"))


def ledger_rows(case: dict) -> list[dict]:
    return [
        json.loads(line)
        for line in case["ledger"].read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_robust_intercept_is_recovered(tmp_path):
    """题面：「线性模型截距必须由全部已匹配通道的稳健共识恢复」"""
    report = report_of(execute(tmp_path, base_payload("intercept"), "case_01"))
    assert abs(report["model"]["intercept"] - INTERCEPT) < 0.12


def test_robust_slope_is_recovered(tmp_path):
    """题面：「线性模型斜率必须使用变换后的横坐标恢复，不能使用原始通道横坐标。」"""
    report = report_of(execute(tmp_path, base_payload("slope"), "case_02"))
    assert abs(report["model"]["slope"] - SLOPE) < 0.12


def test_channel_transform_correspondence_is_recovered(tmp_path):
    """题面：「通道与变换目录的对应关系没有直接给出，必须由斜率指纹和截距共识联合恢复，每个目录标签恰好出现一次。」"""
    report = report_of(execute(tmp_path, base_payload("transform-match"), "case_03"))
    mapping = {item["channel_id"]: item["transform"] for item in report["batch_assignments"]}
    assert mapping["C01"] == "t05"
    assert mapping["C04"] == "t03"
    assert mapping["C10"] == "t06"


def test_outlier_consensus_keeps_known_corruption_out(tmp_path):
    """题面：「每个通道最多有两个大偏差点，其余点围绕同一条线性模型波动。」"""
    report = report_of(execute(tmp_path, base_payload("outliers"), "case_04"))
    assert report["quality"]["outlier_count"] == 20
    assert report["quality"]["inlier_count"] == 120


def test_outlier_count_follows_published_mad_formula(tmp_path):
    """题面锁定中心化残差、1.4826 MAD 和 3.5 倍阈值。"""
    payload = base_payload("outlier-formula")
    report = report_of(execute(tmp_path, payload, "case_formula"))
    mapping = {
        item["channel_id"]: item["transform"]
        for item in report["batch_assignments"]
    }
    catalog = {item["label"]: item for item in payload["transform_catalog"]}
    residuals = []
    for batch in payload["batches"]:
        transform = catalog[mapping[batch["channel_id"]]]
        for point in batch["points"]:
            z = transform["scale"] * point["x"] + transform["shift"]
            residuals.append(
                point["y"] - report["model"]["intercept"] - report["model"]["slope"] * z
            )
    center = statistics.median(residuals)
    raw_mad = statistics.median(abs(value - center) for value in residuals)
    scale = max(1e-6, 1.4826 * raw_mad)
    inliers = sum(abs(value - center) <= 3.5 * scale for value in residuals)
    assert report["quality"]["residual_mad"] == pytest.approx(round(scale, 12), abs=1e-9)
    assert report["quality"]["inlier_count"] == inliers
    assert report["quality"]["outlier_count"] == len(residuals) - inliers


def test_assignment_cost_is_finite_and_bounded(tmp_path):
    """题面：「`inlier_count` 与 `outlier_count` 使用残差中位数和 MAD 判定，`assignment_cost` 是最终对应在裁剪稳健残差下的总代价，必须小于二，`residual_mad` 是原始残差的稳健尺度。」"""
    report = report_of(execute(tmp_path, base_payload("assignment-cost"), "case_05"))
    assert 0.0 <= report["quality"]["assignment_cost"] < 2.0


def test_assignments_are_sorted_by_channel_id(tmp_path):
    """题面：「`batch_assignments` 按通道编号字典序排列十项，每项精确键为 `channel_id` 和 `transform`。」"""
    report = report_of(execute(tmp_path, base_payload("assignment-order"), "case_06"))
    channels = [item["channel_id"] for item in report["batch_assignments"]]
    assert channels == [f"C{index:02d}" for index in range(1, 11)]


def test_report_is_canonical_json(tmp_path):
    """题面：「成功报告使用 UTF-8、键排序、逗号后无空白，并以一个换行结束。」"""
    case = execute(tmp_path, base_payload("canonical"), "case_07")
    raw = case["report"].read_bytes()
    assert raw.endswith(b"\n")
    assert b": " not in raw
    assert raw == canonical(json.loads(raw))


def test_success_uses_public_token(tmp_path):
    """题面：「成功时标准输出只写一行 `REGRESSION_RECOVERED`，标准错误为空。」"""
    case = execute(tmp_path, base_payload("success-channel"), "case_08")
    assert case["process"].returncode == 0
    assert case["process"].stdout == "REGRESSION_RECOVERED\n"
    assert case["process"].stderr == ""


def test_ledger_appends_after_publication(tmp_path):
    """题面：「报告原子发布成功后才允许追加账本记录。」"""
    case = execute(tmp_path, base_payload("ledger-append"), "case_09")
    rows = ledger_rows(case)
    assert len(rows) == 1
    assert set(rows[0]) == {"case_id", "input_sha256", "output_sha256", "sequence"}


def test_identical_rerun_is_idempotent(tmp_path):
    """题面：「相同输入摘要和报告摘要的重跑必须幂等，不得追加重复记录；新的报告摘要必须按连续序号追加。」"""
    case = execute(tmp_path, base_payload("ledger-idempotent"), "case_10")
    before = case["ledger"].read_bytes()
    again = subprocess.run(
        [sys.executable, CANDIDATE, str(case["source"]), str(case["report"]), str(case["ledger"])],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert again.returncode == 0
    assert case["ledger"].read_bytes() == before


def test_changed_input_appends_next_sequence(tmp_path):
    """题面：「新的报告摘要必须按连续序号追加。」"""
    case = execute(tmp_path, base_payload("ledger-sequence"), "case_11")
    second = case["folder"] / "second.json"
    data = base_payload("ledger-sequence")
    data["batches"][0]["points"][0]["y"] = round(data["batches"][0]["points"][0]["y"] + 0.0001, 6)
    second.write_bytes(canonical(data))
    process = subprocess.run(
        [sys.executable, CANDIDATE, str(second), str(case["report"]), str(case["ledger"])],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert process.returncode == 0
    assert [row["sequence"] for row in ledger_rows(case)] == [1, 2]


def test_concurrent_append_uses_exclusive_lock(tmp_path):
    """题面：「账本追加需要独占锁，进程中断后下一次运行从最后一个完整换行形成的检查点恢复。」"""
    folder = tmp_path / "case_12"
    folder.mkdir()
    source = folder / "input.json"
    report = folder / "report.json"
    ledger = folder / "ledger.jsonl"
    source.write_bytes(canonical(base_payload("ledger-lock")))
    processes = [
        subprocess.Popen(
            [sys.executable, CANDIDATE, str(source), str(report), str(ledger)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(3)
    ]
    completed = [process.communicate(timeout=90) for process in processes]
    assert all(process.returncode == 0 for process in processes)
    assert all(stdout == "REGRESSION_RECOVERED\n" for stdout, _ in completed)
    assert len(ledger_rows({"ledger": ledger})) == 1


def test_incomplete_ledger_tail_recovers(tmp_path):
    """题面：「进程中断后下一次运行从最后一个完整换行形成的检查点恢复。」"""
    first = canonical({"case_id":"prior","input_sha256":"a"*64,"output_sha256":"b"*64,"sequence":1})
    case = execute(
        tmp_path,
        base_payload("ledger-checkpoint"),
        "case_13",
        old_ledger=first + b'{"partial":',
    )
    assert [row["sequence"] for row in ledger_rows(case)] == [1, 2]
    assert case["ledger"].read_bytes().endswith(b"\n")


def test_refusal_preserves_previous_artifacts(tmp_path):
    """题面：「验收失败后，上一版报告与账本必须原样保留，目录中不应出现半套文件。」"""
    old_report = b'{"prior":"regression"}\n'
    old_ledger = canonical({"case_id":"old","input_sha256":"0"*64,"output_sha256":"1"*64,"sequence":1})
    payload = base_payload("preserve")
    payload["batches"][0]["points"][0]["x"] = payload["batches"][0]["points"][1]["x"]
    case = execute(tmp_path, payload, "case_14", old_report=old_report, old_ledger=old_ledger)
    assert case["process"].returncode == 2
    assert case["report"].read_bytes() == old_report
    assert case["ledger"].read_bytes() == old_ledger


def test_top_level_missing_key_is_refused(tmp_path):
    """题面：「设备请求的对象由五个成员构成：请求号、通道上限、点上限、变换目录与通道批次。」"""
    payload = base_payload("top-missing")
    payload.pop("transform_catalog")
    case = execute(tmp_path, payload, "case_15")
    assert not case["report"].exists()
    assert not case["ledger"].exists()
    assert case["process"].returncode == 2
    assert "missing keys" in case["process"].stderr


def test_top_level_extra_key_is_refused(tmp_path):
    """题面：「设备请求的对象由五个成员构成：请求号、通道上限、点上限、变换目录与通道批次。」"""
    payload = base_payload("top-extra")
    payload["notes"] = "unexpected"
    case = execute(tmp_path, payload, "case_16")
    assert not case["report"].exists()
    assert not case["ledger"].exists()
    assert case["process"].returncode == 2
    assert "extra keys" in case["process"].stderr


def test_duplicate_json_key_is_refused(tmp_path):
    """题面：「设备请求的对象由五个成员构成：请求号、通道上限、点上限、变换目录与通道批次。」"""
    raw = canonical(base_payload("duplicate-top"))[:-2] + b',"case_id":"again"}\n'
    case = execute(tmp_path, raw, "case_17")
    assert not case["report"].exists()
    assert not case["ledger"].exists()
    assert case["process"].returncode == 2
    assert "duplicate key" in case["process"].stderr


def test_wrong_batch_limit_is_refused(tmp_path):
    """题面：「`max_batches` 必须等于公开上限 10，`max_points` 必须等于公开上限 160。」"""
    payload = base_payload("wrong-batches")
    payload["max_batches"] = 9
    case = execute(tmp_path, payload, "case_18")
    assert not case["report"].exists()
    assert not case["ledger"].exists()
    assert case["process"].returncode == 2
    assert "max_batches must equal 10" in case["process"].stderr


def test_wrong_point_limit_is_refused(tmp_path):
    """题面：「`max_points` 必须等于公开上限 160。」"""
    payload = base_payload("wrong-points")
    payload["max_points"] = 159
    case = execute(tmp_path, payload, "case_19")
    assert not case["report"].exists()
    assert not case["ledger"].exists()
    assert case["process"].returncode == 2
    assert "max_points must equal 160" in case["process"].stderr


def test_catalog_must_have_ten_entries(tmp_path):
    """题面：「`transform_catalog` 恰好包含十项，每项精确键为 `label`、`scale`、`shift`。」"""
    payload = base_payload("catalog-count")
    payload["transform_catalog"].pop()
    case = execute(tmp_path, payload, "case_20")
    assert not case["report"].exists()
    assert not case["ledger"].exists()
    assert case["process"].returncode == 2
    assert "ten entries" in case["process"].stderr


def test_catalog_extra_field_is_refused(tmp_path):
    """题面：「`transform_catalog` 恰好包含十项，每项精确键为 `label`、`scale`、`shift`。」"""
    payload = base_payload("catalog-keys")
    payload["transform_catalog"][0]["extra"] = 1
    case = execute(tmp_path, payload, "case_21")
    assert not case["report"].exists()
    assert not case["ledger"].exists()
    assert case["process"].returncode == 2
    assert "extra keys" in case["process"].stderr


def test_duplicate_catalog_label_is_refused(tmp_path):
    """题面：「目录标签唯一且长度为 1 到 12。」"""
    payload = base_payload("catalog-label")
    payload["transform_catalog"][1]["label"] = payload["transform_catalog"][0]["label"]
    case = execute(tmp_path, payload, "case_22")
    assert not case["report"].exists()
    assert not case["ledger"].exists()
    assert case["process"].returncode == 2
    assert "labels must be unique" in case["process"].stderr


def test_duplicate_transform_is_refused(tmp_path):
    """题面：「两个目录项不得给出重复的尺度和位移组合，任意两项也不能同时过于接近。」"""
    payload = base_payload("catalog-duplicate")
    payload["transform_catalog"][1]["scale"] = payload["transform_catalog"][0]["scale"]
    payload["transform_catalog"][1]["shift"] = payload["transform_catalog"][0]["shift"]
    case = execute(tmp_path, payload, "case_23")
    assert not case["report"].exists()
    assert not case["ledger"].exists()
    assert case["process"].returncode == 2
    assert "duplicate transforms" in case["process"].stderr


def test_scale_out_of_range_is_refused(tmp_path):
    """题面：「尺度位于 `[0.45, 2.2]`，位移位于 `[-1.5, 1.5]`。」"""
    payload = base_payload("scale-range")
    payload["transform_catalog"][0]["scale"] = 2.5
    case = execute(tmp_path, payload, "case_24")
    assert not case["report"].exists()
    assert not case["ledger"].exists()
    assert case["process"].returncode == 2
    assert "transform scale" in case["process"].stderr


def test_batches_must_have_ten_channels(tmp_path):
    """题面：「`batches` 恰好包含十个通道，每项精确键为 `channel_id` 和 `points`。」"""
    payload = base_payload("batch-count")
    payload["batches"].pop()
    case = execute(tmp_path, payload, "case_25")
    assert not case["report"].exists()
    assert not case["ledger"].exists()
    assert case["process"].returncode == 2
    assert "ten channels" in case["process"].stderr


def test_channel_point_count_boundaries(tmp_path):
    """题面：「每个通道包含十二到二十个点，全部通道的点数之和不得超过 160。」"""
    payload = base_payload("point-count")
    payload["batches"][0]["points"] = payload["batches"][0]["points"][:11]
    case = execute(tmp_path, payload, "case_26")
    assert not case["report"].exists()
    assert not case["ledger"].exists()
    assert case["process"].returncode == 2
    assert "point count" in case["process"].stderr


def test_duplicate_x_value_is_refused(tmp_path):
    """题面：「同一通道的横坐标必须互不相同。」"""
    payload = base_payload("duplicate-x")
    payload["batches"][0]["points"][0]["x"] = payload["batches"][0]["points"][1]["x"]
    case = execute(tmp_path, payload, "case_27")
    assert not case["report"].exists()
    assert not case["ledger"].exists()
    assert case["process"].returncode == 2
    assert "duplicate x" in case["process"].stderr


def test_non_numeric_point_is_refused(tmp_path):
    """题面：「每个点精确键为 `x` 和 `y`，值必须是 `[-5, 5]` 内的有限数值。」"""
    payload = base_payload("point-type")
    payload["batches"][0]["points"][0]["y"] = "0.5"
    case = execute(tmp_path, payload, "case_28")
    assert not case["report"].exists()
    assert not case["ledger"].exists()
    assert case["process"].returncode == 2
    assert "must be numeric" in case["process"].stderr


def test_publish_failure_leaves_no_partial_file(tmp_path):
    """题面：「验收失败后，上一版报告与账本必须原样保留，目录中不应出现半套文件。」"""
    old_ledger = canonical({"case_id":"old","input_sha256":"c"*64,"output_sha256":"d"*64,"sequence":1})
    case = execute(
        tmp_path,
        base_payload("publish-failure"),
        "case_29",
        old_ledger=old_ledger,
        report_is_directory=True,
    )
    leftovers = [path.name for path in case["folder"].iterdir() if path.name.startswith(".report.json.")]
    assert case["process"].returncode == 2
    assert case["report"].is_dir()
    assert case["ledger"].read_bytes() == old_ledger
    assert leftovers == []


def test_candidate_entry_is_explicit_and_runnable(tmp_path):
    """题面：「调用方式是 `python3 /app/regression_recover.py <input.json> <report.json> <ledger.jsonl>`。」"""
    case = execute(tmp_path, base_payload("entry-runnable"), "case_30")
    assert Path("/app/regression_recover.py").as_posix() == "/app/regression_recover.py"
    assert os.environ.get("REGRESSION_ENTRY", "/app/regression_recover.py") == CANDIDATE
    assert case["process"].returncode == 0
