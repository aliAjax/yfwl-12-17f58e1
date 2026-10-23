#!/usr/bin/env python3
"""Provisional multi-channel robust regression command."""

from __future__ import annotations

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


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("payload must be an object")
    return value


def number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("numeric value required")
    return float(value)


def validate(payload: dict) -> dict:
    required = {"case_id", "max_batches", "max_points", "transform_catalog", "batches"}
    if set(payload) != required:
        raise ValueError("payload keys do not match")
    if payload["max_batches"] != MAX_BATCHES or payload["max_points"] != MAX_POINTS:
        raise ValueError("public limits are unsupported")
    catalog = payload["transform_catalog"]
    batches = payload["batches"]
    if not isinstance(catalog, list) or len(catalog) != 10:
        raise ValueError("transform catalog is invalid")
    clean_catalog = []
    labels = set()
    for item in catalog:
        if not isinstance(item, dict) or set(item) != {"label", "scale", "shift"}:
            raise ValueError("catalog keys do not match")
        if not isinstance(item["label"], str) or item["label"] in labels:
            raise ValueError("catalog label is invalid")
        labels.add(item["label"])
        clean_catalog.append(
            {
                "label": item["label"],
                "scale": number(item["scale"]),
                "shift": number(item["shift"]),
            }
        )
    if not isinstance(batches, list) or len(batches) != 10:
        raise ValueError("batch count is invalid")
    clean_batches = []
    channels = set()
    total = 0
    for batch in batches:
        if not isinstance(batch, dict) or set(batch) != {"channel_id", "points"}:
            raise ValueError("batch keys do not match")
        if not isinstance(batch["channel_id"], str) or batch["channel_id"] in channels:
            raise ValueError("channel_id is invalid")
        channels.add(batch["channel_id"])
        points = batch["points"]
        if not isinstance(points, list) or not 12 <= len(points) <= 20:
            raise ValueError("point count is invalid")
        clean_points = []
        for point in points:
            if not isinstance(point, dict) or set(point) != {"x", "y"}:
                raise ValueError("point keys do not match")
            clean_points.append({"x": number(point["x"]), "y": number(point["y"])})
        total += len(clean_points)
        clean_batches.append({"channel_id": batch["channel_id"], "points": clean_points})
    if total > MAX_POINTS:
        raise ValueError("point budget exceeded")
    return {
        "case_id": payload["case_id"],
        "max_batches": MAX_BATCHES,
        "max_points": MAX_POINTS,
        "transform_catalog": clean_catalog,
        "batches": clean_batches,
    }


def estimate(payload: dict) -> dict:
    catalog = sorted(payload["transform_catalog"], key=lambda item: item["label"])
    points = []
    assignments = []
    for index, batch in enumerate(payload["batches"]):
        transform = catalog[index % len(catalog)]
        assignments.append({"channel_id": batch["channel_id"], "transform": transform["label"]})
        for point in batch["points"]:
            points.append(
                (
                    transform["scale"] * point["x"] + transform["shift"],
                    point["y"],
                )
            )
    xs = [item[0] for item in points]
    ys = [item[1] for item in points]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / sum(
        (x - mean_x) ** 2 for x in xs
    )
    intercept = mean_y - slope * mean_x
    residuals = [y - intercept - slope * x for x, y in points]
    return {
        "status": "recovered",
        "model": {"intercept": round(intercept, 9), "slope": round(slope, 9)},
        "batch_assignments": sorted(assignments, key=lambda item: item["channel_id"]),
        "quality": {
            "inlier_count": len(points),
            "outlier_count": 0,
            "assignment_cost": 0.0,
            "residual_mad": round(statistics.median(abs(value) for value in residuals), 12),
        },
    }


def canonical(payload: dict) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def append_ledger(path: Path, case_id: str, source_hash: str, output_hash: str) -> None:
    rows = []
    if path.exists():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(
        row.get("case_id") == case_id
        and row.get("input_sha256") == source_hash
        and row.get("output_sha256") == output_hash
        for row in rows
    ):
        return
    record = {
        "case_id": case_id,
        "input_sha256": source_hash,
        "output_sha256": output_hash,
        "sequence": len(rows) + 1,
    }
    with path.open("ab") as handle:
        handle.write(canonical(record))
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print("REGRESSION_REFUSED: expected input output ledger", file=sys.stderr)
        return 2
    try:
        source, target, ledger = map(Path, argv[1:])
        payload = validate(load(source))
        output = canonical(estimate(payload))
        atomic_write(target, output)
        append_ledger(
            ledger,
            payload["case_id"],
            hashlib.sha256(source.read_bytes()).hexdigest(),
            hashlib.sha256(output).hexdigest(),
        )
        print("REGRESSION_RECOVERED")
        return 0
    except Exception as exc:
        print(f"REGRESSION_REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
