#!/usr/bin/env python3
"""Evaluate weak real-data top-level composition.

This helper is for the real-data setting where top-level ground truth is a small
set of broad incident families (for example, ``top_level.json`` links a top-level
family to constituent low-level incident IDs) rather than a complete graph of
high-level relations.

It supports two complementary measurements:

1. low_real child support: using the already-computed low_real incident-level
   summary, ask whether each method recovered enough child incidents to support a
   top-level event. This does not require raw result folders.
2. linker-baseline composition support: run the existing offline composition
   baselines from detection/baselines/composition_baselines.py over the merged
   real low-level predictions, then ask whether any emitted composite groups
   together enough matched child incidents from a top-level event.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


def progress_iter(iterable, *, total=None, desc="progress", enabled=True, unit="it", leave=True):
    if enabled and tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, unit=unit, leave=leave, dynamic_ncols=True, mininterval=0.5)
    return iterable


def progress_write(message: str, *, enabled: bool = True) -> None:
    if enabled and tqdm is not None:
        tqdm.write(message, file=sys.stderr)
    else:
        print(message, file=sys.stderr, flush=True)


def parse_dt(value: Any, *, assume_pacific: bool = False) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        if assume_pacific and ZoneInfo is not None:
            dt = dt.replace(tzinfo=ZoneInfo("America/Los_Angeles"))
        else:
            dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def dt_to_iso(dt: Optional[datetime]) -> str:
    if dt is None:
        return ""
    if dt.tzinfo is None:
        return dt.isoformat()
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or value == "":
            return default
        f = float(value)
        if not math.isfinite(f):
            return default
        return f
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def ratio(num: float, den: float) -> Optional[float]:
    return float(num) / float(den) if den else None


def bool_from_cell(value: Any) -> bool:
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def normalize_top_gt(top_gt: Any) -> Dict[str, Dict[str, Any]]:
    if isinstance(top_gt, dict):
        return {str(k): v for k, v in top_gt.items() if isinstance(v, dict)}
    if isinstance(top_gt, list):
        return {str(x.get("top_level_incident_id") or i): x for i, x in enumerate(top_gt) if isinstance(x, dict)}
    return {}


def normalize_low_gt(low_gt: Any) -> Dict[str, Dict[str, Any]]:
    if isinstance(low_gt, dict):
        return {str(k): v for k, v in low_gt.items() if isinstance(v, dict)}
    if isinstance(low_gt, list):
        return {str(x.get("incident_id") or i): x for i, x in enumerate(low_gt) if isinstance(x, dict)}
    return {}


def group_low_real_rows(rows: Sequence[Mapping[str, str]]) -> Dict[Tuple[str, str], Mapping[str, str]]:
    out: Dict[Tuple[str, str], Mapping[str, str]] = {}
    for row in rows:
        method = str(row.get("method") or "").strip()
        gt_id = str(row.get("gt_id") or "").strip()
        if method and gt_id:
            out[(method, gt_id)] = row
    return out


def earliest_match_time_by_method_gt(match_rows: Sequence[Mapping[str, str]]) -> Dict[Tuple[str, str], datetime]:
    out: Dict[Tuple[str, str], datetime] = {}
    for row in match_rows:
        if str(row.get("type_correct") or "").strip().lower() not in {"true", "1", "yes"}:
            continue
        method = str(row.get("method") or "").strip()
        gt_id = str(row.get("gt_id") or "").strip()
        dt = parse_dt(row.get("system_report_time"))
        if not method or not gt_id or dt is None:
            continue
        key = (method, gt_id)
        prev = out.get(key)
        if prev is None or dt < prev:
            out[key] = dt
    return out


def pred_to_gt_map_from_match_rows(match_rows: Sequence[Mapping[str, str]]) -> Dict[Tuple[str, str], str]:
    """Return {(method, pred_id): gt_low_id} from type-correct low_real matches."""
    out: Dict[Tuple[str, str], str] = {}
    for row in match_rows:
        if str(row.get("type_correct") or "").strip().lower() not in {"true", "1", "yes"}:
            continue
        method = str(row.get("method") or "").strip()
        gt_id = str(row.get("gt_id") or "").strip()
        pred_id = str(row.get("pred_id") or "").strip()
        if method and pred_id and gt_id:
            out[(method, pred_id)] = gt_id.split("::", 1)[0]
    return out


def compute_child_support_top_metrics(
    *,
    method: str,
    top_id: str,
    top: Mapping[str, Any],
    low_gt: Mapping[str, Mapping[str, Any]],
    low_rows_by_method_gt: Mapping[Tuple[str, str], Mapping[str, str]],
    match_time_by_method_gt: Mapping[Tuple[str, str], datetime],
    min_child_hits: int,
    min_child_recall: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    child_ids_all = [str(x) for x in top.get("low_level_incident_ids", [])]
    child_ids = [cid for cid in child_ids_all if cid in low_gt]
    missing_child_ids = [cid for cid in child_ids_all if cid not in low_gt]

    child_rows: List[Dict[str, Any]] = []
    hit_ids: List[str] = []
    hit_names: List[str] = []
    pre_report_hits = 0
    coverage_weight_num = coverage_weight_den = 0.0
    inverse_weight_num = inverse_weight_den = 0.0
    earliest_detection: Optional[datetime] = None
    first_article: Optional[datetime] = None

    for cid in child_ids:
        gt = low_gt.get(cid, {})
        row = low_rows_by_method_gt.get((method, cid), {})
        matched = bool_from_cell(row.get("matched", row.get("recall", "")))
        if "matched" not in row and "recall" in row:
            matched = safe_float(row.get("recall"), 0.0) == 1.0
        coverage = safe_float(row.get("coverage_score"), None)
        if coverage is None:
            coverage = safe_float(row.get("gt_coverage_score"), None)
        if coverage is None:
            coverage = 1.0
        coverage = max(1e-6, float(coverage))
        coverage_weight_den += coverage
        inverse_weight_den += 1.0 / coverage
        if matched:
            hit_ids.append(cid)
            hit_names.append(str(gt.get("final_name") or cid))
            coverage_weight_num += coverage
            inverse_weight_num += 1.0 / coverage
        article_dt = parse_dt(row.get("earliest_article_time"), assume_pacific=False) or parse_dt(gt.get("earliest_article_datetime_pacific"), assume_pacific=True)
        if article_dt is not None and (first_article is None or article_dt < first_article):
            first_article = article_dt
        det_dt = match_time_by_method_gt.get((method, cid)) or parse_dt(row.get("matched_system_report_time"))
        if matched and det_dt is not None:
            if earliest_detection is None or det_dt < earliest_detection:
                earliest_detection = det_dt
            if article_dt is not None and det_dt <= article_dt:
                pre_report_hits += 1
        child_rows.append({
            "evaluation_variant": "child_support",
            "method": method,
            "source_method": method,
            "composition_method": "low_real_child_support",
            "top_level_incident_id": top_id,
            "top_level_incident_name": top.get("top_level_incident_name", top_id),
            "child_incident_id": cid,
            "child_incident_name": gt.get("final_name", cid),
            "child_incident_type": gt.get("incident_type", ""),
            "child_start_datetime_pacific": gt.get("start_datetime_pacific", ""),
            "child_end_datetime_pacific": gt.get("end_datetime_pacific", ""),
            "child_earliest_article_datetime_pacific": gt.get("earliest_article_datetime_pacific", ""),
            "matched": int(bool(matched)),
            "coverage_score": coverage,
            "matched_system_report_time": dt_to_iso(det_dt),
        })

    child_count = len(child_ids)
    hit_count = len(hit_ids)
    child_recall = ratio(hit_count, child_count) or 0.0
    cov_recall = ratio(coverage_weight_num, coverage_weight_den) or 0.0
    inv_cov_recall = ratio(inverse_weight_num, inverse_weight_den) or 0.0

    min_k = hit_count >= int(min_child_hits)
    frac_ok = child_recall >= float(min_child_recall)
    support_detected = bool(min_k or (frac_ok and hit_count >= 2))
    top_start = parse_dt(top.get("start_datetime_pacific"), assume_pacific=True)
    delay_hours = ""
    if earliest_detection is not None and top_start is not None:
        delay_hours = (earliest_detection - top_start).total_seconds() / 3600.0
    pre_report_any = bool(earliest_detection is not None and first_article is not None and earliest_detection <= first_article)

    row = {
        "evaluation_variant": "child_support",
        "method": method,
        "source_method": method,
        "composition_method": "low_real_child_support",
        "top_level_incident_id": top_id,
        "top_level_incident_name": top.get("top_level_incident_name", top_id),
        "incident_types": ";".join(str(x) for x in top.get("incident_types", [])),
        "top_start_datetime_pacific": top.get("start_datetime_pacific", ""),
        "top_end_datetime_pacific": top.get("end_datetime_pacific", ""),
        "top_earliest_child_article_time": dt_to_iso(first_article),
        "total_child_ids_in_top_gt": len(child_ids_all),
        "evaluable_child_ids": child_count,
        "missing_child_ids": ";".join(missing_child_ids),
        "child_hit_count": hit_count,
        "child_recall": child_recall,
        "coverage_weighted_child_recall": cov_recall,
        "inverse_coverage_weighted_child_recall": inv_cov_recall,
        "any_child_detected": int(hit_count >= 1),
        "min_child_hits_detected": int(min_k),
        "fraction_threshold_detected": int(frac_ok),
        "composition_support_detected": int(support_detected),
        "pre_report_child_hit_count": pre_report_hits,
        "pre_report_child_recall": ratio(pre_report_hits, child_count) or 0.0,
        "pre_report_any_child_detected": int(pre_report_any),
        "earliest_detection_time": dt_to_iso(earliest_detection),
        "earliest_detection_delay_hours_from_top_start": delay_hours,
        "child_hit_ids": ";".join(hit_ids),
        "child_hit_names": "; ".join(hit_names),
        "best_composite_id": "",
        "num_composites_emitted": "",
        "metric_note": "Weak real top-level support from detected low-level child incidents.",
    }
    return row, child_rows


def summarize_by_method(top_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    keys = sorted({(str(r.get("method")), str(r.get("source_method")), str(r.get("composition_method")), str(r.get("evaluation_variant"))) for r in top_rows})
    out: List[Dict[str, Any]] = []
    for method, source_method, composition_method, evaluation_variant in keys:
        rows = [r for r in top_rows if r.get("method") == method and r.get("source_method") == source_method and r.get("composition_method") == composition_method and r.get("evaluation_variant") == evaluation_variant]
        if not rows:
            continue
        num_top = len(rows)
        total_children = sum(safe_int(r.get("evaluable_child_ids"), 0) for r in rows)
        total_hits = sum(safe_int(r.get("child_hit_count"), 0) for r in rows)
        any_top = sum(safe_int(r.get("any_child_detected"), 0) for r in rows)
        support_top = sum(safe_int(r.get("composition_support_detected"), 0) for r in rows)
        min_k_top = sum(safe_int(r.get("min_child_hits_detected"), 0) for r in rows)
        frac_top = sum(safe_int(r.get("fraction_threshold_detected"), 0) for r in rows)
        pre_report_any = sum(safe_int(r.get("pre_report_any_child_detected"), 0) for r in rows)
        child_recalls = [safe_float(r.get("child_recall"), 0.0) or 0.0 for r in rows]
        cov_recalls = [safe_float(r.get("coverage_weighted_child_recall"), 0.0) or 0.0 for r in rows]
        inv_cov_recalls = [safe_float(r.get("inverse_coverage_weighted_child_recall"), 0.0) or 0.0 for r in rows]
        delays = [safe_float(r.get("earliest_detection_delay_hours_from_top_start"), None) for r in rows]
        delays = [d for d in delays if d is not None]
        out.append({
            "evaluation_variant": evaluation_variant,
            "method": method,
            "source_method": source_method,
            "composition_method": composition_method,
            "num_top_level_incidents": num_top,
            "top_level_recall_any_child": ratio(any_top, num_top),
            "top_level_recall_min_child_hits": ratio(min_k_top, num_top),
            "top_level_recall_fraction_threshold": ratio(frac_top, num_top),
            "top_level_recall_composition_support": ratio(support_top, num_top),
            "top_level_pre_report_any_child_recall": ratio(pre_report_any, num_top),
            "micro_child_recall": ratio(total_hits, total_children),
            "macro_child_recall": statistics.mean(child_recalls) if child_recalls else None,
            "macro_coverage_weighted_child_recall": statistics.mean(cov_recalls) if cov_recalls else None,
            "macro_inverse_coverage_weighted_child_recall": statistics.mean(inv_cov_recalls) if inv_cov_recalls else None,
            "child_hit_count": total_hits,
            "evaluable_child_count": total_children,
            "mean_earliest_detection_delay_hours_from_top_start": statistics.mean(delays) if delays else None,
            "median_earliest_detection_delay_hours_from_top_start": statistics.median(delays) if delays else None,
        })
    return out


def pred_to_baseline_row(pred: Any, ev: Any) -> Dict[str, Any]:
    geom = ev.prediction_primary_geometry(pred)
    lat = lon = None
    if geom is not None and not getattr(geom, "is_empty", True):
        c = geom.centroid
        lat = float(c.y)
        lon = float(c.x)
    row: Dict[str, Any] = {
        "prediction_id": str(pred.prediction_id),
        "incident_type": str(pred.incident_type),
        "confidence": pred.confidence,
        "time_first_incident_predicted": dt_to_iso(pred.detection_time),
        "active_interval": {
            "start": dt_to_iso(pred.start_time),
            "end": dt_to_iso(pred.end_time),
        },
    }
    if lat is not None and lon is not None:
        row["affected_region"] = {
            "center": {"latitude": lat, "longitude": lon},
            "support_points": [{"latitude": lat, "longitude": lon, "coverage": 1.0}],
            "radius_km": 1.0,
        }
    return row


def load_merged_real_prediction_rows(args: argparse.Namespace) -> Dict[str, List[Dict[str, Any]]]:
    """Load, dedupe, and merge real low-level predictions using evaluate_results helpers."""
    try:
        try:
            import evaluation.evaluate_results as ev  # type: ignore
        except Exception:
            import evaluate_results as ev  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"Could not import evaluate_results helpers. Run from repo root or set PYTHONPATH=. ({exc})") from exc

    allowed_types = ev.load_allowed_types(args.incident_types)
    result_dirs = ev.discover_real_low_result_dirs(
        Path(args.results_root),
        real_experiment_name=args.real_experiment_name,
        experiment_glob=args.experiment_glob,
        methods=args.methods or (),
    )
    if not result_dirs:
        raise FileNotFoundError(f"No real low-level result dirs found under {args.results_root} for {args.real_experiment_name}")

    first_owner_index = ev.build_prediction_first_owner_index(result_dirs) if args.leakage_filter else {}
    grouped: Dict[str, List[Any]] = defaultdict(list)
    progress_enabled = bool(getattr(args, "progress_bar", True))
    for method, experiment, date_name, result_dir in progress_iter(
        result_dirs,
        total=len(result_dirs),
        desc="load real low-level predictions",
        enabled=progress_enabled,
        unit="dir",
        leave=False,
    ):
        if not (result_dir / ev.LOW_LEVEL_FILENAME).exists():
            continue
        pred_items, _summary = ev.load_predictions_for_result_dir(
            result_dir,
            run_name=date_name,
            allowed_types=allowed_types,
            min_coverage_normalized=args.min_coverage_normalized,
            method=method,
            experiment=experiment,
            first_owner_index=first_owner_index,
            leakage_filter_enabled=args.leakage_filter,
            leakage_filter_methods=args.leakage_filter_methods or (),
            write_denoised_low_level=False,
            denoised_methods=(),
            prefer_denoised_low_level=args.prefer_denoised_low_level,
            include_rationale_type_guesses=args.real_rationale_type_guesses,
        )
        for pred in pred_items:
            if pred.detection_time is None:
                pred.detection_time = pred.start_time or pred.end_time
        grouped[method].extend(pred_items)

    out: Dict[str, List[Dict[str, Any]]] = {}
    method_items = sorted(grouped.items())
    for method, preds in progress_iter(
        method_items,
        total=len(method_items),
        desc="merge real predictions",
        enabled=progress_enabled,
        unit="method",
        leave=False,
    ):
        progress_write(f"Preparing linker input for {method}: raw={len(preds)}", enabled=progress_enabled)
        deduped, _dedupe = ev.dedupe_real_predictions(preds)
        progress_write(f"  {method}: after exact dedupe={len(deduped)}", enabled=progress_enabled)
        merged, _merge = ev.merge_real_prediction_updates(
            deduped,
            enabled=args.real_merge_prediction_updates,
            max_temporal_gap_hours=args.real_merge_temporal_gap_hours,
            max_spatial_distance_km=args.real_merge_spatial_distance_km,
            min_spatial_iou=args.real_merge_min_spatial_iou,
            progress_enabled=progress_enabled,
            progress_desc=f"merge updates {method}",
        )
        progress_write(f"  {method}: after update merge={len(merged)}", enabled=progress_enabled)
        out[method] = [pred_to_baseline_row(pred, ev) for pred in merged if not getattr(pred, "rationale_type_guess", False)]
    return out


def run_linker_baselines(
    *,
    args: argparse.Namespace,
    top_gt: Mapping[str, Mapping[str, Any]],
    low_gt: Mapping[str, Mapping[str, Any]],
    match_rows: Sequence[Mapping[str, str]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    try:
        from detection.baselines import composition_baselines as cb  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"Could not import detection.baselines.composition_baselines. Run from repo root or set PYTHONPATH=. ({exc})") from exc

    pred_to_gt = pred_to_gt_map_from_match_rows(match_rows)
    rows_by_method = load_merged_real_prediction_rows(args)
    top_rows: List[Dict[str, Any]] = []
    child_rows: List[Dict[str, Any]] = []
    composite_rows: List[Dict[str, Any]] = []

    progress_enabled = bool(getattr(args, "progress_bar", True))
    method_items = sorted(rows_by_method.items())
    for source_method, low_rows in progress_iter(
        method_items,
        total=len(method_items),
        desc="run composition linkers",
        enabled=progress_enabled,
        unit="method",
    ):
        progress_write(f"Running composition linkers for {source_method}: low_rows={len(low_rows)}", enabled=progress_enabled)
        payloads: Dict[str, Dict[str, Any]] = {}
        wanted = set(args.composition_baselines or [])
        if "simple_proximity_only_linker" in wanted:
            payloads["simple_proximity_only_linker"] = cb.simple_proximity_only_linker(
                "<merged_real_low_level_predictions>",
                max_distance_km=args.composition_proximity_max_distance_km,
                min_children=args.composition_min_children,
                low_level_rows=low_rows,
                source_label=f"merged:{source_method}:{args.real_experiment_name}",
            )
        if "same_type_temporal_overlap_linker" in wanted:
            payloads["same_type_temporal_overlap_linker"] = cb.same_type_temporal_overlap_linker(
                "<merged_real_low_level_predictions>",
                max_temporal_gap_hours=args.composition_same_type_temporal_gap_hours,
                max_distance_km=args.composition_same_type_max_distance_km,
                min_children=args.composition_min_children,
                low_level_rows=low_rows,
                source_label=f"merged:{source_method}:{args.real_experiment_name}",
            )

        for linker_name, payload in payloads.items():
            composites = payload.get("high_level_incidents", []) if isinstance(payload, dict) else []
            method_name = f"{source_method}+{linker_name}"
            comp_infos: List[Dict[str, Any]] = []
            for comp in composites:
                if not isinstance(comp, dict):
                    continue
                child_pred_ids = [str(x) for x in comp.get("child_prediction_ids", [])]
                matched_gt_ids = sorted({pred_to_gt.get((source_method, pid)) for pid in child_pred_ids if pred_to_gt.get((source_method, pid))})
                comp_id = str(comp.get("composite_id") or f"{linker_name}_{len(comp_infos):04d}")
                info = {
                    "composite_id": comp_id,
                    "source_method": source_method,
                    "composition_method": linker_name,
                    "method": method_name,
                    "incident_type": comp.get("incident_type", ""),
                    "child_prediction_count": len(child_pred_ids),
                    "matched_low_gt_count": len(matched_gt_ids),
                    "matched_low_gt_ids": matched_gt_ids,
                    "child_prediction_ids": child_pred_ids,
                }
                comp_infos.append(info)
                composite_rows.append({
                    **{k: (";".join(v) if isinstance(v, list) else v) for k, v in info.items()},
                    "confidence": comp.get("confidence", ""),
                    "relationship": ((comp.get("composition_reasoning") or {}).get("relationship") if isinstance(comp.get("composition_reasoning"), dict) else ""),
                })

            for top_id, top in top_gt.items():
                if not isinstance(top, dict):
                    continue
                top_child_ids = [str(x) for x in top.get("low_level_incident_ids", []) if str(x) in low_gt]
                top_child_set = set(top_child_ids)
                best_hits: List[str] = []
                best_comp_id = ""
                union_hits: set[str] = set()
                for info in comp_infos:
                    hits = sorted(top_child_set & set(info["matched_low_gt_ids"]))
                    union_hits.update(hits)
                    if len(hits) > len(best_hits):
                        best_hits = hits
                        best_comp_id = info["composite_id"]
                # Use the best single composite for top-level composition support.
                hit_count = len(best_hits)
                child_count = len(top_child_ids)
                child_recall = ratio(hit_count, child_count) or 0.0
                min_k = hit_count >= int(args.min_child_hits)
                frac_ok = child_recall >= float(args.min_child_recall)
                support_detected = bool(min_k or (frac_ok and hit_count >= 2))
                hit_names = [str(low_gt.get(cid, {}).get("final_name") or cid) for cid in best_hits]
                row = {
                    "evaluation_variant": "linker_baseline",
                    "method": method_name,
                    "source_method": source_method,
                    "composition_method": linker_name,
                    "top_level_incident_id": top_id,
                    "top_level_incident_name": top.get("top_level_incident_name", top_id),
                    "incident_types": ";".join(str(x) for x in top.get("incident_types", [])),
                    "top_start_datetime_pacific": top.get("start_datetime_pacific", ""),
                    "top_end_datetime_pacific": top.get("end_datetime_pacific", ""),
                    "top_earliest_child_article_time": "",
                    "total_child_ids_in_top_gt": len(top.get("low_level_incident_ids", [])),
                    "evaluable_child_ids": child_count,
                    "missing_child_ids": ";".join(str(x) for x in top.get("low_level_incident_ids", []) if str(x) not in low_gt),
                    "child_hit_count": hit_count,
                    "child_recall": child_recall,
                    "coverage_weighted_child_recall": "",
                    "inverse_coverage_weighted_child_recall": "",
                    "any_child_detected": int(hit_count >= 1),
                    "min_child_hits_detected": int(min_k),
                    "fraction_threshold_detected": int(frac_ok),
                    "composition_support_detected": int(support_detected),
                    "pre_report_child_hit_count": "",
                    "pre_report_child_recall": "",
                    "pre_report_any_child_detected": "",
                    "earliest_detection_time": "",
                    "earliest_detection_delay_hours_from_top_start": "",
                    "child_hit_ids": ";".join(best_hits),
                    "child_hit_names": "; ".join(hit_names),
                    "best_composite_id": best_comp_id,
                    "num_composites_emitted": len(comp_infos),
                    "metric_note": "Real composition linker metric: a top-level event is supported only if one emitted composite groups enough children that were matched to listed low-level GT incidents.",
                }
                top_rows.append(row)
                for cid in top_child_ids:
                    child_rows.append({
                        "evaluation_variant": "linker_baseline",
                        "method": method_name,
                        "source_method": source_method,
                        "composition_method": linker_name,
                        "top_level_incident_id": top_id,
                        "top_level_incident_name": top.get("top_level_incident_name", top_id),
                        "child_incident_id": cid,
                        "child_incident_name": low_gt.get(cid, {}).get("final_name", cid),
                        "child_incident_type": low_gt.get(cid, {}).get("incident_type", ""),
                        "child_start_datetime_pacific": low_gt.get(cid, {}).get("start_datetime_pacific", ""),
                        "child_end_datetime_pacific": low_gt.get(cid, {}).get("end_datetime_pacific", ""),
                        "child_earliest_article_datetime_pacific": low_gt.get(cid, {}).get("earliest_article_datetime_pacific", ""),
                        "matched": int(cid in set(best_hits)),
                        "coverage_score": "",
                        "matched_system_report_time": "",
                    })
    return top_rows, child_rows, composite_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--top-level-gt",
        default="evaluation/ground_truth/real/top_level.json",
        help="Path to the real-data top-level composition labels.",
    )
    parser.add_argument(
        "--low-level-gt",
        default="evaluation/ground_truth/real/low_level_gt_corrected.json",
        help="Path to the corrected real-data low-level incident labels.",
    )
    parser.add_argument("--low-real-summary-dir", required=True, help="Path to evaluation/evaluation_summary/low_real")
    parser.add_argument("--out-dir", default="evaluation/evaluation_summary/real_composition")
    parser.add_argument("--methods", nargs="*", default=None, help="Optional low-level source method subset. Defaults to all methods in low-real summary for child support; for linker baselines, pass e.g. incidentlens to limit runtime.")
    parser.add_argument("--min-child-hits", type=int, default=2)
    parser.add_argument("--min-child-recall", type=float, default=0.20)

    # Optional real linker-baseline mode. Uses existing composition_baselines.py.
    parser.add_argument("--include-linker-baselines", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--results-root", default="evaluation/results")
    parser.add_argument("--real-experiment-name", default="real_all_incidents")
    parser.add_argument("--experiment-glob", default=None)
    parser.add_argument("--incident-types", default="evaluation/filtered_incidents.txt")
    parser.add_argument("--composition-baselines", nargs="*", default=["simple_proximity_only_linker", "same_type_temporal_overlap_linker"])
    parser.add_argument("--composition-proximity-max-distance-km", type=float, default=20.0)
    parser.add_argument("--composition-same-type-temporal-gap-hours", type=float, default=24.0)
    parser.add_argument("--composition-same-type-max-distance-km", type=float, default=None)
    parser.add_argument("--composition-min-children", type=int, default=2)
    parser.add_argument("--prefer-denoised-low-level", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--real-rationale-type-guesses", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-coverage-normalized", type=float, default=0.0)
    parser.add_argument("--leakage-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--leakage-filter-methods", nargs="*", default=[])
    parser.add_argument("--real-merge-prediction-updates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--real-merge-temporal-gap-hours", type=float, default=24.0)
    parser.add_argument("--real-merge-spatial-distance-km", type=float, default=10.0)
    parser.add_argument("--real-merge-min-spatial-iou", type=float, default=0.01)
    parser.add_argument("--progress-bar", action=argparse.BooleanOptionalAction, default=True, help="Show progress bars for child-support scoring, loading/merging predictions, and linker baselines.")
    args = parser.parse_args()

    top_gt = normalize_top_gt(load_json(Path(args.top_level_gt)))
    low_gt = normalize_low_gt(load_json(Path(args.low_level_gt)))
    if not top_gt:
        raise ValueError(f"No top-level GT loaded from {args.top_level_gt}")
    if not low_gt:
        raise ValueError(f"No low-level GT loaded from {args.low_level_gt}")

    summary_dir = Path(args.low_real_summary_dir)
    by_name_path = summary_dir / "aggregate_metrics_by_gt_incident_name.csv"
    if not by_name_path.exists():
        raise FileNotFoundError(f"Missing {by_name_path}. Run evaluate_results --mode low_real first.")
    low_rows = read_csv_rows(by_name_path)
    match_rows = read_csv_rows(summary_dir / "match_records.csv")
    low_by_method_gt = group_low_real_rows(low_rows)
    match_time = earliest_match_time_by_method_gt(match_rows)
    methods = sorted({m for m, _ in low_by_method_gt.keys()})
    if args.methods:
        wanted = set(args.methods)
        methods = [m for m in methods if m in wanted]
    if not methods:
        raise ValueError("No methods found. Check aggregate_metrics_by_gt_incident_name.csv or --methods.")

    progress_enabled = bool(args.progress_bar)
    if progress_enabled and tqdm is None:
        progress_write("tqdm is not installed; using plain progress logs. Install with `pip install tqdm` for progress bars.", enabled=False)
        progress_enabled = False

    top_rows: List[Dict[str, Any]] = []
    child_rows: List[Dict[str, Any]] = []
    child_tasks = [(method, top_id, top) for method in methods for top_id, top in top_gt.items()]
    for method, top_id, top in progress_iter(
        child_tasks,
        total=len(child_tasks),
        desc="score top-level child support",
        enabled=progress_enabled,
        unit="top",
    ):
            row, children = compute_child_support_top_metrics(
                method=method,
                top_id=str(top_id),
                top=top,
                low_gt=low_gt,
                low_rows_by_method_gt=low_by_method_gt,
                match_time_by_method_gt=match_time,
                min_child_hits=args.min_child_hits,
                min_child_recall=args.min_child_recall,
            )
            top_rows.append(row)
            child_rows.extend(children)

    composite_rows: List[Dict[str, Any]] = []
    baseline_error = None
    if args.include_linker_baselines:
        try:
            linker_top, linker_child, composite_rows = run_linker_baselines(
                args=args,
                top_gt=top_gt,
                low_gt=low_gt,
                match_rows=match_rows,
            )
            top_rows.extend(linker_top)
            child_rows.extend(linker_child)
        except Exception as exc:
            baseline_error = repr(exc)
            print(f"WARNING: failed to run linker baselines: {baseline_error}", file=sys.stderr)

    summary_rows = summarize_by_method(top_rows)
    out_dir = Path(args.out_dir)
    top_fields = [
        "evaluation_variant", "method", "source_method", "composition_method",
        "top_level_incident_id", "top_level_incident_name", "incident_types",
        "top_start_datetime_pacific", "top_end_datetime_pacific", "top_earliest_child_article_time",
        "total_child_ids_in_top_gt", "evaluable_child_ids", "missing_child_ids",
        "child_hit_count", "child_recall", "coverage_weighted_child_recall", "inverse_coverage_weighted_child_recall",
        "any_child_detected", "min_child_hits_detected", "fraction_threshold_detected", "composition_support_detected",
        "pre_report_child_hit_count", "pre_report_child_recall", "pre_report_any_child_detected",
        "earliest_detection_time", "earliest_detection_delay_hours_from_top_start",
        "child_hit_ids", "child_hit_names", "best_composite_id", "num_composites_emitted", "metric_note",
    ]
    child_fields = [
        "evaluation_variant", "method", "source_method", "composition_method",
        "top_level_incident_id", "top_level_incident_name",
        "child_incident_id", "child_incident_name", "child_incident_type",
        "child_start_datetime_pacific", "child_end_datetime_pacific", "child_earliest_article_datetime_pacific",
        "matched", "coverage_score", "matched_system_report_time",
    ]
    summary_fields = [
        "evaluation_variant", "method", "source_method", "composition_method", "num_top_level_incidents",
        "top_level_recall_any_child", "top_level_recall_min_child_hits", "top_level_recall_fraction_threshold",
        "top_level_recall_composition_support", "top_level_pre_report_any_child_recall",
        "micro_child_recall", "macro_child_recall", "macro_coverage_weighted_child_recall",
        "macro_inverse_coverage_weighted_child_recall", "child_hit_count", "evaluable_child_count",
        "mean_earliest_detection_delay_hours_from_top_start", "median_earliest_detection_delay_hours_from_top_start",
    ]
    composite_fields = [
        "method", "source_method", "composition_method", "composite_id", "incident_type",
        "child_prediction_count", "matched_low_gt_count", "matched_low_gt_ids", "child_prediction_ids",
        "confidence", "relationship",
    ]
    write_csv(out_dir / "real_composition_by_top_level.csv", top_rows, top_fields)
    write_csv(out_dir / "real_composition_child_details.csv", child_rows, child_fields)
    write_csv(out_dir / "real_composition_summary_by_method.csv", summary_rows, summary_fields)
    write_csv(out_dir / "real_composition_linker_composites.csv", composite_rows, composite_fields)
    write_json(out_dir / "real_composition_by_top_level.json", top_rows)
    write_json(out_dir / "real_composition_child_details.json", child_rows)
    write_json(out_dir / "real_composition_summary_by_method.json", summary_rows)
    write_json(out_dir / "real_composition_linker_composites.json", composite_rows)
    write_json(out_dir / "real_composition_metadata.json", {
        "top_level_gt": str(args.top_level_gt),
        "low_level_gt": str(args.low_level_gt),
        "low_real_summary_dir": str(args.low_real_summary_dir),
        "methods": methods,
        "min_child_hits": args.min_child_hits,
        "min_child_recall": args.min_child_recall,
        "include_linker_baselines": args.include_linker_baselines,
        "composition_baselines": args.composition_baselines,
        "baseline_import_or_runtime_error": baseline_error,
        "metric_definition": (
            "Weak real top-level composition evaluation. The child_support variant asks whether enough listed "
            "low-level GT children were detected. The linker_baseline variant runs offline composition linkers over "
            "merged real low-level predictions and asks whether one emitted composite groups enough matched children "
            "from a top-level event. This is not full graph-level high-level accuracy."
        ),
    })
    print(f"Wrote real composition summary to {out_dir}")


if __name__ == "__main__":
    main()
