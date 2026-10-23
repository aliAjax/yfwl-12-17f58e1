#!/usr/bin/env python3
"""Recover calibration transforms and a robust linear model from unlabeled channels."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
from pathlib import Path


MAX_BATCHES = 10
MAX_POINTS = 160
TOP_KEYS = {"case_id", "max_batches", "max_points", "transform_catalog", "batches"}
CATALOG_KEYS = {"label", "scale", "shift"}
BATCH_KEYS = {"channel_id", "points"}
POINT_KEYS = {"x", "y"}
LEDGER_KEYS = {"case_id", "input_sha256", "output_sha256", "sequence"}


class RecoveryError(ValueError):
    """Raised when a channel set cannot identify the robust calibration."""


def reject_constant(token: str) -> None:
    raise RecoveryError(f"non-finite JSON number {token}")


def pairs_object(pairs: list[tuple[str, object]]) -> dict:
    value: dict = {}
    for key, item in pairs:
        if key in value:
            raise RecoveryError(f"duplicate key {key}")
        value[key] = item
    return value


def mapping(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise RecoveryError(f"{label} must be an object")
    return value


def exact_keys(value: dict, expected: set[str], label: str) -> None:
    missing = sorted(expected.difference(value))
    extra = sorted(set(value).difference(expected))
    if missing:
        raise RecoveryError(f"{label} missing keys: {','.join(missing)}")
    if extra:
        raise RecoveryError(f"{label} extra keys: {','.join(extra)}")


def number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecoveryError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RecoveryError(f"{label} must be finite")
    return result


def integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecoveryError(f"{label} must be an integer")
    return value


def read_json(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RecoveryError("input cannot be read") from exc
    try:
        payload = json.loads(
            text,
            object_pairs_hook=pairs_object,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RecoveryError("input is not valid JSON") from exc
    return mapping(payload, "payload")


def parse_catalog(raw_catalog: object) -> list[dict]:
    if not isinstance(raw_catalog, list) or len(raw_catalog) != MAX_BATCHES:
        raise RecoveryError("transform_catalog must contain ten entries")
    catalog: list[dict] = []
    labels: set[str] = set()
    signatures: set[tuple[float, float]] = set()
    for index, raw_item in enumerate(raw_catalog):
        item = mapping(raw_item, f"transform_catalog[{index}]")
        exact_keys(item, CATALOG_KEYS, f"transform_catalog[{index}]")
        label = item["label"]
        if not isinstance(label, str) or not 1 <= len(label) <= 12:
            raise RecoveryError(f"transform_catalog[{index}].label is invalid")
        if label in labels:
            raise RecoveryError("transform catalog labels must be unique")
        labels.add(label)
        scale = number(item["scale"], f"transform_catalog[{index}].scale")
        shift = number(item["shift"], f"transform_catalog[{index}].shift")
        if not 0.45 <= scale <= 2.2:
            raise RecoveryError("transform scale outside supported range")
        if not -1.5 <= shift <= 1.5:
            raise RecoveryError("transform shift outside supported range")
        signature = (round(scale, 9), round(shift, 9))
        if signature in signatures:
            raise RecoveryError("transform catalog contains duplicate transforms")
        signatures.add(signature)
        catalog.append({"label": label, "scale": scale, "shift": shift})
    ordered = sorted(catalog, key=lambda item: (item["scale"], item["shift"]))
    for left in range(len(ordered)):
        for right in range(left + 1, len(ordered)):
            scale_delta = abs(ordered[left]["scale"] - ordered[right]["scale"])
            shift_delta = abs(ordered[left]["shift"] - ordered[right]["shift"])
            if scale_delta < 0.035 and shift_delta < 0.035:
                raise RecoveryError("transform catalog is not separable")
    return catalog


def parse_batches(raw_batches: object) -> list[dict]:
    if not isinstance(raw_batches, list) or len(raw_batches) != MAX_BATCHES:
        raise RecoveryError("batches must contain ten channels")
    batches: list[dict] = []
    channel_ids: set[str] = set()
    total_points = 0
    for index, raw_batch in enumerate(raw_batches):
        batch = mapping(raw_batch, f"batches[{index}]")
        exact_keys(batch, BATCH_KEYS, f"batches[{index}]")
        channel_id = batch["channel_id"]
        if not isinstance(channel_id, str) or not 1 <= len(channel_id) <= 12:
            raise RecoveryError(f"batches[{index}].channel_id is invalid")
        if channel_id in channel_ids:
            raise RecoveryError("channel_id values must be unique")
        channel_ids.add(channel_id)
        raw_points = batch["points"]
        if not isinstance(raw_points, list) or not 12 <= len(raw_points) <= 20:
            raise RecoveryError(f"batches[{index}] point count is invalid")
        points: list[dict] = []
        x_values: set[float] = set()
        for point_index, raw_point in enumerate(raw_points):
            point = mapping(raw_point, f"batches[{index}].points[{point_index}]")
            exact_keys(point, POINT_KEYS, f"batches[{index}].points[{point_index}]")
            x_value = number(point["x"], f"batches[{index}].points[{point_index}].x")
            y_value = number(point["y"], f"batches[{index}].points[{point_index}].y")
            if not -5.0 <= x_value <= 5.0 or not -5.0 <= y_value <= 5.0:
                raise RecoveryError("point coordinate outside supported range")
            if x_value in x_values:
                raise RecoveryError("a channel contains duplicate x values")
            x_values.add(x_value)
            points.append({"x": x_value, "y": y_value})
        total_points += len(points)
        batches.append({"channel_id": channel_id, "points": points})
    if total_points > MAX_POINTS:
        raise RecoveryError("total point budget exceeded")
    return batches


def parse_payload(payload: dict) -> dict:
    exact_keys(payload, TOP_KEYS, "payload")
    case_id = payload["case_id"]
    if not isinstance(case_id, str) or not 1 <= len(case_id) <= 48:
        raise RecoveryError("case_id must be a non-empty string")
    if any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in case_id):
        raise RecoveryError("case_id has unsupported characters")
    max_batches = integer(payload["max_batches"], "max_batches")
    max_points = integer(payload["max_points"], "max_points")
    if max_batches != MAX_BATCHES:
        raise RecoveryError(f"max_batches must equal {MAX_BATCHES}")
    if max_points != MAX_POINTS:
        raise RecoveryError(f"max_points must equal {MAX_POINTS}")
    return {
        "case_id": case_id,
        "max_batches": max_batches,
        "max_points": max_points,
        "transform_catalog": parse_catalog(payload["transform_catalog"]),
        "batches": parse_batches(payload["batches"]),
    }


def median(values: list[float]) -> float:
    if not values:
        raise RecoveryError("median of an empty collection")
    return float(statistics.median(values))


def robust_line(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Theil-Sen slope plus median intercept, with outlier resistance."""
    slopes: list[float] = []
    for left in range(len(points)):
        for right in range(left + 1, len(points)):
            delta_x = points[right][0] - points[left][0]
            if abs(delta_x) > 1e-10:
                slopes.append((points[right][1] - points[left][1]) / delta_x)
    if len(slopes) < 12:
        raise RecoveryError("not enough distinct x values for robust slope")
    slope = median(slopes)
    intercept = median(
        y_value - slope * x_value for x_value, y_value in points
    )
    return intercept, slope


def hungarian(cost: list[list[float]]) -> list[int]:
    size = len(cost)
    row_potential = [0.0] * (size + 1)
    column_potential = [0.0] * (size + 1)
    assignment = [0] * (size + 1)
    previous = [0] * (size + 1)
    for row in range(1, size + 1):
        assignment[0] = row
        minimum = [math.inf] * (size + 1)
        used = [False] * (size + 1)
        column0 = 0
        while True:
            used[column0] = True
            row0 = assignment[column0]
            delta = math.inf
            column1 = 0
            for column in range(1, size + 1):
                if used[column]:
                    continue
                reduced = (
                    cost[row0 - 1][column - 1]
                    - row_potential[row0]
                    - column_potential[column]
                )
                if reduced < minimum[column]:
                    minimum[column] = reduced
                    previous[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    column1 = column
            for column in range(size + 1):
                if used[column]:
                    row_potential[assignment[column]] += delta
                    column_potential[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if assignment[column0] == 0:
                break
        while True:
            column1 = previous[column0]
            assignment[column0] = assignment[column1]
            column0 = column1
            if column0 == 0:
                break
    result = [-1] * size
    for column in range(1, size + 1):
        result[assignment[column] - 1] = column - 1
    return result


def initial_assignment(
    lines: list[tuple[float, float]],
    catalog: list[dict],
) -> tuple[float, float, list[int], float]:
    """Generate algebraic candidates from observed slopes, then assign."""
    best: tuple[float, float, list[int], float] | None = None
    for _, local_slope in lines:
        for transform in catalog:
            trial_slope = local_slope / transform["scale"]
            if abs(trial_slope) < 0.20:
                continue
            costs = [
                [
                    (line[1] - trial_slope * item["scale"]) ** 2
                    for item in catalog
                ]
                for line in lines
            ]
            assignment = hungarian(costs)
            intercept_guesses = [
                lines[index][0] - trial_slope * catalog[assignment[index]]["shift"]
                for index in range(MAX_BATCHES)
            ]
            trial_intercept = median(intercept_guesses)
            score = sum(
                costs[index][assignment[index]]
                + (
                    lines[index][0]
                    - trial_intercept
                    - trial_slope * catalog[assignment[index]]["shift"]
                )
                ** 2
                for index in range(MAX_BATCHES)
            )
            if best is None or score < best[3]:
                best = (trial_intercept, trial_slope, assignment, score)
    if best is None:
        raise RecoveryError("no algebraic transform candidate is admissible")
    if best[3] > 0.3:
        raise RecoveryError("channel fingerprints do not match transform catalog")
    return best


def transformed_points(
    batches: list[dict],
    catalog: list[dict],
    assignment: list[int],
) -> list[tuple[float, float, int]]:
    result: list[tuple[float, float, int]] = []
    for batch_index, batch in enumerate(batches):
        transform = catalog[assignment[batch_index]]
        for point in batch["points"]:
            transformed_x = transform["scale"] * point["x"] + transform["shift"]
            result.append((transformed_x, point["y"], batch_index))
    return result


def fit_robust(payload: dict, assignment: list[int]) -> tuple[float, float]:
    points = transformed_points(
        payload["batches"], payload["transform_catalog"], assignment
    )
    return robust_line([(x_value, y_value) for x_value, y_value, _ in points])


def residual_statistics(
    payload: dict,
    assignment: list[int],
    intercept: float,
    slope: float,
) -> tuple[int, int, float]:
    residuals = []
    for x_value, y_value, _ in transformed_points(
        payload["batches"], payload["transform_catalog"], assignment
    ):
        residuals.append(y_value - intercept - slope * x_value)
    center = median(residuals)
    absolute = sorted(abs(value - center) for value in residuals)
    mad = median(absolute)
    scale = max(1e-6, 1.4826 * mad)
    inliers = sum(1 for value in residuals if abs(value - center) <= 3.5 * scale)
    outliers = len(residuals) - inliers
    return inliers, outliers, round(scale, 12)


def refine_assignment(
    payload: dict,
    assignment: list[int],
    intercept: float,
    slope: float,
) -> tuple[list[int], float]:
    catalog = payload["transform_catalog"]
    batches = payload["batches"]
    costs: list[list[float]] = []
    for batch in batches:
        row_cost = []
        for transform in catalog:
            residuals = []
            for point in batch["points"]:
                z_value = transform["scale"] * point["x"] + transform["shift"]
                residuals.append(point["y"] - intercept - slope * z_value)
            center = median(residuals)
            row_cost.append(
                sum(
                    min(abs(value - center), 0.25) ** 2
                    for value in residuals
                )
            )
        costs.append(row_cost)
    updated = hungarian(costs)
    cost = sum(costs[index][updated[index]] for index in range(MAX_BATCHES))
    return updated, round(cost, 12)


def recover(payload: dict) -> dict:
    lines = [
        robust_line([(point["x"], point["y"]) for point in batch["points"]])
        for batch in payload["batches"]
    ]
    intercept, slope, assignment, _ = initial_assignment(
        lines, payload["transform_catalog"]
    )
    for _ in range(4):
        intercept, slope = fit_robust(payload, assignment)
        updated, assignment_cost = refine_assignment(
            payload, assignment, intercept, slope
        )
        if updated == assignment:
            break
        assignment = updated
    intercept, slope = fit_robust(payload, assignment)
    inliers, outliers, residual_mad = residual_statistics(
        payload, assignment, intercept, slope
    )
    if outliers > 26:
        raise RecoveryError("outlier count is inconsistent with robust consensus")
    assignments = sorted(
        [
            {
                "channel_id": payload["batches"][index]["channel_id"],
                "transform": payload["transform_catalog"][assignment[index]]["label"],
            }
            for index in range(MAX_BATCHES)
        ],
        key=lambda item: item["channel_id"],
    )
    return {
        "status": "recovered",
        "model": {
            "intercept": round(intercept, 9),
            "slope": round(slope, 9),
        },
        "batch_assignments": assignments,
        "quality": {
            "inlier_count": inliers,
            "outlier_count": outliers,
            "assignment_cost": assignment_cost,
            "residual_mad": residual_mad,
        },
    }


def canonical(payload: dict) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    parent = path.parent
    if not parent.is_dir():
        raise RecoveryError("output parent does not exist")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        complete = raw[: raw.rfind(b"\n") + 1]
        temporary = path.with_name(path.name + ".recover")
        atomic_write(temporary, complete)
        os.replace(temporary, path)
        raw = complete
    records: list[dict] = []
    for line in raw.splitlines():
        if not line:
            raise RecoveryError("ledger contains an empty record")
        try:
            record = json.loads(line.decode("utf-8"), object_pairs_hook=pairs_object)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RecoveryError("ledger is not valid JSONL") from exc
        record = mapping(record, "ledger record")
        exact_keys(record, LEDGER_KEYS, "ledger record")
        records.append(record)
    return records


def append_ledger(path: Path, case_id: str, input_hash: str, output_hash: str) -> bool:
    if not path.parent.is_dir():
        raise RecoveryError("ledger parent does not exist")
    lock_path = path.with_name(path.name + ".lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(descriptor, "r+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        records = load_ledger(path)
        if any(
            record["case_id"] == case_id
            and record["input_sha256"] == input_hash
            and record["output_sha256"] == output_hash
            for record in records
        ):
            return False
        record = {
            "case_id": case_id,
            "input_sha256": input_hash,
            "output_sha256": output_hash,
            "sequence": max((int(item["sequence"]) for item in records), default=0) + 1,
        }
        with path.open("ab", buffering=0) as handle:
            handle.write(canonical(record))
            os.fsync(handle.fileno())
    return True


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print("REGRESSION_REFUSED: expected input output ledger", file=sys.stderr)
        return 2
    input_path = Path(argv[1])
    output_path = Path(argv[2])
    ledger_path = Path(argv[3])
    before_output = output_path.read_bytes() if output_path.is_file() else None
    before_ledger = ledger_path.read_bytes() if ledger_path.is_file() else None
    try:
        payload = parse_payload(read_json(input_path))
        report = recover(payload)
        output = canonical(report)
        atomic_write(output_path, output)
        append_ledger(
            ledger_path,
            payload["case_id"],
            sha256(input_path.read_bytes()),
            sha256(output),
        )
        print("REGRESSION_RECOVERED")
        return 0
    except Exception as exc:
        if before_output is None:
            try:
                output_path.unlink()
            except OSError:
                pass
        else:
            try:
                atomic_write(output_path, before_output)
            except Exception:
                pass
        if before_ledger is None:
            try:
                ledger_path.unlink()
            except OSError:
                pass
        else:
            try:
                atomic_write(ledger_path, before_ledger)
            except Exception:
                pass
        print(f"REGRESSION_REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
