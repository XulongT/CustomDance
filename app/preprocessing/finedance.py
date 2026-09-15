"""Reproduce the standing-catalog quality gates, then materialize raw motions."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np

from app.utils.retriever_text import normalize_genre
from app.preprocessing.motion_library import MotionLibraryBuilder, _floor_offset
from app.preprocessing.motion_quality import (
    DEFAULT_THRESHOLDS,
    classify_quality,
    decode_motion_to_joints,
    detect_boundary_trim,
    sequence_metrics,
)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_policy(path: Path) -> dict:
    policy = json.loads(path.read_text(encoding="utf-8"))
    if policy.get("schema_version") != "customdance-finedance-preparation-v1":
        raise ValueError("unsupported preparation policy")
    if (policy["fps"], policy["window_frames"], policy["hop_frames"]) != (30, 120, 120):
        raise ValueError("this backend requires 30 FPS, 120-frame non-overlapping windows")
    splits = policy["source_splits"]
    if not splits or not set(splits.values()) <= {"train", "val", "test"}:
        raise ValueError("invalid source split contract")
    thresholds = policy["thresholds"]
    if set(thresholds) != set(DEFAULT_THRESHOLDS):
        raise ValueError("threshold keys must match the supported quality policy")
    if any(not np.isfinite(float(value)) or float(value) < 0 for value in thresholds.values()):
        raise ValueError("thresholds must be finite and nonnegative")
    return policy


def validate_output(output: Path, *inputs: Path) -> None:
    if output.exists():
        raise FileExistsError(f"output already exists; choose a new directory: {output}")
    for path in inputs:
        if output == path or output in path.parents or path in output.parents:
            raise ValueError("output must not overlap a raw-data or model-index input")


def scan_sources(raw_root: Path, splits: dict[str, str]) -> list[dict]:
    categories = {"label_json": ".json", "motion": ".npy", "music_npy": ".npy"}
    sets = {}
    for category, suffix in categories.items():
        directory = raw_root / category
        if not directory.is_dir():
            raise FileNotFoundError(f"missing FineDance directory: {category}")
        sets[category] = {path.stem for path in directory.glob("*" + suffix) if path.is_file()}
    complete = set.intersection(*sets.values())
    if complete != set(splits):
        raise ValueError(
            "raw dataset does not match the release source catalog: "
            f"{len(set(splits) - complete)} missing, {len(complete - set(splits))} extra complete sources"
        )
    sources = []
    for source_id in sorted(complete):
        paths = {name: raw_root / name / (source_id + suffix)
                 for name, suffix in categories.items()}
        for path in paths.values():
            if raw_root not in path.resolve(strict=True).parents:
                raise ValueError("raw source file escapes the dataset root")
        label = json.loads(paths["label_json"].read_text(encoding="utf-8"))
        if not isinstance(label, dict):
            raise ValueError(f"{source_id}: label must be an object")
        sources.append({
            "source_id": source_id,
            "source_split": splits[source_id],
            # Normalize dataset label fields only; this never interprets user intent.
            "genre": normalize_genre(
                label["style2"] if str(label.get("style2", "")).strip() else label.get("style1", "")
            ),
            "source_song": str(label.get("name", source_id)),
            "paths": paths,
        })
    return sources


def filter_source(source: dict, rest: np.ndarray, policy: dict) -> tuple[list[dict], dict]:
    """Compute gates from raw arrays, independent of the supplied retrieval index."""
    paths = source["paths"]
    raw = np.load(paths["motion"], mmap_mode="r", allow_pickle=False)
    music = np.load(paths["music_npy"], mmap_mode="r", allow_pickle=False)
    source_id = source["source_id"]
    if raw.ndim != 2 or raw.shape[1] != 315:
        raise ValueError(f"{source_id}: motion must have shape (T,315)")
    if music.ndim != 2 or music.shape[1] != 35:
        raise ValueError(f"{source_id}: music features must have shape (T,35)")
    if not np.isfinite(raw).all() or not np.isfinite(music).all():
        raise ValueError(f"{source_id}: non-finite raw motion/music values")
    report = {
        "source_id": source_id, "source_split": source["source_split"],
        "genre": source["genre"], "raw_shapes": [list(raw.shape), list(music.shape)],
    }
    frames = min(len(raw), len(music))
    if frames < 120:
        return [], {**report, "excluded": "too_short"}
    thresholds = policy["thresholds"]
    positions = decode_motion_to_joints(raw[:frames], rest)
    if not np.isfinite(positions).all():
        raise ValueError(f"{source_id}: non-finite decoded joints")
    source_metrics = sequence_metrics(positions, fps=30, thresholds=thresholds)
    report["source_quality"] = source_metrics
    if "sloped_floor_candidate" in classify_quality(source_metrics, thresholds)["flags"]:
        return [], {**report, "excluded": "sloped_floor"}
    boundary = detect_boundary_trim(
        positions, fps=30,
        speed_threshold_mps=float(thresholds["leading_idle_speed_mps"]),
        min_idle_seconds=float(thresholds["boundary_idle_min_seconds"]),
        scan_seconds=float(thresholds["boundary_scan_seconds"]),
        context_frames=int(thresholds["boundary_context_frames"]),
        min_active_seconds=float(thresholds["leading_active_min_seconds"]),
    )
    report["boundary"] = boundary
    start, end = int(boundary["trim_start_frame"]), int(boundary["trim_end_frame"])
    if boundary["all_inactive"] or end - start < 120:
        return [], {**report, "excluded": "inactive_or_too_short_after_trim"}
    trimmed = positions[start:end]
    floor_y = float(sequence_metrics(trimmed, fps=30, thresholds=thresholds)["foot_floor_height_y"])
    report["floor_offset_y"] = floor_y
    if source["source_split"] != "train":
        return [], {**report, "excluded": "not_deployment_split"}
    rows, dropped = [], []
    for offset in range(0, len(trimmed) - 119, 120):
        metrics = sequence_metrics(trimmed[offset:offset + 120], fps=30, thresholds=thresholds)
        flags = classify_quality(metrics, thresholds)["flags"]
        reasons = [flag for flag in flags if flag == "sustained_low_pelvis_candidate"]
        if source["genre"] in policy["floorwork_genres"]:
            reasons.extend(flag for flag in flags if flag in {"floorwork_candidate", "extreme_tilt_candidate"})
        raw_start, raw_end = start + offset, start + offset + 119
        phrase_id = f"FD_{source_id}_{raw_start:06d}_{raw_end:06d}"
        if reasons:
            dropped.append({"phrase_id": phrase_id, "reasons": reasons})
            continue
        rows.append({
            "phrase_id": phrase_id, "source_id": source_id, "source_split": "train",
            "genre": source["genre"], "raw_start_frame": raw_start, "raw_end_frame": raw_end,
            # Deployment metadata stores each window's floor, not the whole-source floor.
            "floor_offset_y": float(metrics["floor_offset_y"]),
        })
    report.update({
        "candidate_windows": len(rows) + len(dropped),
        "kept_windows": len(rows), "dropped_windows": dropped,
    })
    return rows, report


def validate_pairing(generated: list[dict], indexed: list[dict]) -> None:
    actual = {row["phrase_id"]: row for row in generated}
    expected = {row["phrase_id"]: row for row in indexed}
    if len(actual) != len(generated) or len(expected) != len(indexed):
        raise ValueError("duplicate phrase IDs in preparation or retrieval index")
    if actual.keys() != expected.keys():
        raise ValueError(
            "preprocessing/index mismatch: "
            f"{len(expected.keys() - actual.keys())} missing, "
            f"{len(actual.keys() - expected.keys())} unexpected phrases; "
            "use the matching raw dataset, skeleton reference, policy and model bundle"
        )
    for phrase_id, row in actual.items():
        other = expected[phrase_id]
        for field in ("source_id", "source_split", "genre", "raw_start_frame", "raw_end_frame"):
            if row[field] != other[field]:
                raise ValueError(f"preprocessing/index {field} mismatch: {phrase_id}")
        if not np.isclose(row["floor_offset_y"], _floor_offset(other), atol=1e-6, rtol=0):
            raise ValueError(f"preprocessing/index floor offset mismatch: {phrase_id}")


def prepare(
    *, raw_root: Path, rest_joints: Path, index_root: Path, output_root: Path,
    policy_path: Path, progress: Callable[[dict], None] | None = None,
) -> dict:
    raw_root = raw_root.expanduser().resolve(strict=True)
    rest_joints = rest_joints.expanduser().resolve(strict=True)
    index_root = index_root.expanduser().resolve(strict=True)
    policy_path = policy_path.expanduser().resolve(strict=True)
    output_root = output_root.expanduser().resolve()
    validate_output(output_root, raw_root, index_root)
    policy = load_policy(policy_path)
    rest = np.asarray(np.load(rest_joints, allow_pickle=False), dtype=np.float32)
    if rest.shape != (55, 3) or not np.isfinite(rest).all():
        raise ValueError("skeleton reference must be finite (55,3)")
    sources = scan_sources(raw_root, policy["source_splits"])
    # Validate the supplied index before the expensive source pass.
    checker = MotionLibraryBuilder(raw_root=raw_root, deployment_index_root=index_root,
                                  output_root=output_root)
    _, indexed = checker._load_index()
    checker._validate_rows(indexed, expected_rows=policy["expected_rows"],
                           expected_sources=policy["expected_sources"])
    selected, reports = [], []
    raw_file_count = sum(len(source["paths"]) for source in sources)
    for number, source in enumerate(sources, 1):
        rows, report = filter_source(source, rest, policy)
        selected.extend(rows)
        reports.append(report)
        if progress and (number % 10 == 0 or number == len(sources)):
            progress({"stage": "FILTERING", "completed": number, "total": len(sources),
                      "kept_windows": len(selected)})
    validate_pairing(selected, indexed)
    preparation = {
        "index_pairing": "EXACT_PHRASE_SET_AND_FLOOR_MATCH",
        "raw_files": raw_file_count, "training_required": False,
    }
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.preparing-",
                                    dir=output_root.parent)).resolve()
    try:
        builder = MotionLibraryBuilder(
            raw_root=raw_root, deployment_index_root=index_root,
            output_root=staging / "library", preparation=preparation,
        )
        manifest = builder.build(
            expected_rows=policy["expected_rows"], expected_sources=policy["expected_sources"],
            progress=(lambda done, total: progress(
                {"stage": "MATERIALIZING", "completed": done, "total": total}
            )) if progress else None,
        )
        library = staging / "library"
        audit = library / "preprocessing"
        audit.mkdir()
        summary = {
            "status": "READY", "input_sources": len(sources),
            "source_exclusions": dict(Counter(
                row["excluded"] for row in reports if row.get("excluded")
            )),
            "candidate_train_windows": sum(row.get("candidate_windows", 0) for row in reports),
            "dropped_train_windows": sum(len(row.get("dropped_windows", [])) for row in reports),
            "clip_count": manifest["clip_count"], "index_pairing": "PASS",
            "raw_files": raw_file_count,
        }
        write_json(audit / "report.json", {"summary": summary, "policy": policy,
                                          "sources": reports})
        write_json(audit / "selected_windows.json", selected)
        # Never replace a user's existing output, even if created during processing.
        if output_root.exists():
            raise FileExistsError(f"output appeared during preparation: {output_root}")
        os.rename(library, output_root)
        return summary
    finally:
        if staging.parent == output_root.parent and staging.name.startswith(
            f".{output_root.name}.preparing-"
        ):
            shutil.rmtree(staging)
