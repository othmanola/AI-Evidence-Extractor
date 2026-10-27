# AI Evidence Extractor — canvas-drawn heatmap + reliable upload progress.
# -*- coding: utf-8 -*-
"""AI Evidence Extractor — video evidence triage with visual, audio, and contextual analysis."""
import os, json, time, tempfile, shutil, subprocess, logging, re, pickle, math, hashlib
import queue, threading, csv, io, base64, html as _html
import uuid
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd, cv2, soundfile as sf, gradio as gr
import torch
from ultralytics import YOLO
import account_client as _account_client
import update_client as _update_client

logging.basicConfig(level="INFO", format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("aee")
APP_DATA = Path(os.environ.get("AEE_DATA_DIR", str(Path.home() / "AppData" / "Local" / "AI Evidence Extractor")))
MODEL_DIR = APP_DATA / "models"
OUTPUT_DIR = APP_DATA / "outputs"
HISTORY_FILE = APP_DATA / "analysis_history.json"
HISTORY_DIR = OUTPUT_DIR / "history"
HISTORY_COLUMNS = ["Analyzed at", "Video", "Duration (s)", "Detections", "Mode", "Case"]
_HISTORY_LOCK = threading.Lock()
MANIFEST = MODEL_DIR / "ensemble_manifest.json"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_model_path(path):
    model_path = Path(path).expanduser()
    return model_path if model_path.is_absolute() else MODEL_DIR / model_path


def _persist_download(source, filename):
    """Publish a generated artifact at a stable path for Gradio downloads."""
    if not source:
        return None
    source_path = Path(source)
    if not source_path.is_file() or source_path.stat().st_size <= 0:
        log.warning("Download artifact missing or empty: %s", source_path)
        return None
    target = OUTPUT_DIR / filename
    partial = target.with_name(target.stem + ".partial" + target.suffix)
    try:
        shutil.copy2(source_path, partial)
        os.replace(partial, target)
        return str(target)
    except Exception:
        log.exception("Could not publish download artifact: %s", source_path)
        try:
            partial.unlink(missing_ok=True)
        except OSError:
            pass
        return None


def _latest_published(pattern):
    """Return the newest previously generated report for a fresh desktop session."""
    files = [p for p in OUTPUT_DIR.glob(pattern)
             if p.is_file() and p.stat().st_size > 0]
    if not files:
        return None
    return str(max(files, key=lambda p: p.stat().st_mtime).resolve())


def _read_analysis_history():
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except FileNotFoundError:
        return []
    except Exception:
        log.exception("Could not read analysis history")
        return []


def _history_dataframe():
    with _HISTORY_LOCK:
        entries = _read_analysis_history()
    rows = []
    for entry in reversed(entries):
        if not isinstance(entry, dict):
            continue
        try:
            analyzed = datetime.fromisoformat(entry.get("analyzed_at", ""))
            analyzed_label = analyzed.astimezone().strftime("%Y-%m-%d %H:%M:%S")
        except (TypeError, ValueError):
            analyzed_label = str(entry.get("analyzed_at", ""))
        rows.append({
            "Analyzed at": analyzed_label,
            "Video": str(entry.get("video", "")),
            "Duration (s)": round(float(entry.get("duration_seconds", 0) or 0), 2),
            "Detections": int(entry.get("detections", 0) or 0),
            "Mode": "Enhanced" if entry.get("high_accuracy") else "Standard",
            "Case": str(entry.get("case", "")),
        })
    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


def _history_entry_id(entry, index):
    entry_id = str(entry.get("id") or "")
    if entry_id:
        return entry_id
    legacy = f"{entry.get('analyzed_at', '')}\0{entry.get('video', '')}\0{index}"
    return "legacy-" + hashlib.sha256(legacy.encode("utf-8")).hexdigest()[:20]


def _history_choices(selected_id=None):
    with _HISTORY_LOCK:
        entries = _read_analysis_history()
    choices = []
    for index, entry in reversed(list(enumerate(entries))):
        if not isinstance(entry, dict):
            continue
        entry_id = _history_entry_id(entry, index)
        title = str(entry.get("video") or "Untitled video")
        try:
            date = datetime.fromisoformat(str(entry.get("analyzed_at") or ""))
            label = f"{date.astimezone().strftime('%b %d, %Y · %H:%M')}  ·  {title}"
        except (TypeError, ValueError):
            label = title
        choices.append((label, entry_id))
    value = selected_id if any(item[1] == selected_id for item in choices) else (
        choices[0][1] if choices else None)
    return choices, value


def _history_list_items():
    with _HISTORY_LOCK:
        entries = _read_analysis_history()
    items = []
    for index, entry in reversed(list(enumerate(entries))):
        if not isinstance(entry, dict):
            continue
        try:
            analyzed = datetime.fromisoformat(
                str(entry.get("analyzed_at") or "")
            ).astimezone().strftime("%b %d, %Y · %H:%M")
        except (TypeError, ValueError):
            analyzed = "Date unavailable"
        items.append({
            "id": _history_entry_id(entry, index),
            "video": str(entry.get("video") or "Untitled video"),
            "analyzed": analyzed,
            "case": str(entry.get("case") or ""),
            "detections": int(entry.get("detections", 0) or 0),
        })
    return items


def _remove_history_artifacts(entry, remaining_entries):
    details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
    video_dirs = set()
    warnings = []
    for key in ("captures", "heatmaps", "highlights", "object_summaries"):
        paths = details.get(key, [])
        if not isinstance(paths, list):
            paths = [paths]
        for source in paths:
            if not isinstance(source, str) or not source:
                continue
            path = Path(source).resolve()
            if not path.is_relative_to(HISTORY_DIR.resolve()):
                log.warning("Skipping history artifact outside history storage: %s", path)
                continue
            if path.parent.name in {"captures", "heatmaps", "highlights", "summaries"}:
                video_dir = path.parent.parent
                if video_dir.parent.parent == HISTORY_DIR.resolve():
                    video_dirs.add(video_dir)

    removed_runs = set()
    for video_dir in video_dirs:
        run_dir = video_dir.parent
        if video_dir.is_dir():
            try:
                shutil.rmtree(video_dir)
            except OSError as exc:
                log.exception("Could not remove generated history artifacts: %s", video_dir)
                warnings.append(f"Could not remove generated files in {video_dir.name}: {exc}")
        removed_runs.add(run_dir)

    for run_dir in removed_runs:
        has_sibling = False
        for index, sibling in enumerate(remaining_entries):
            sibling_details = sibling.get("details") if isinstance(sibling, dict) else None
            if not isinstance(sibling_details, dict):
                continue
            for key in ("captures", "heatmaps", "highlights", "object_summaries"):
                paths = sibling_details.get(key, [])
                if not isinstance(paths, list):
                    paths = [paths]
                if any(
                    isinstance(source, str)
                    and source
                    and Path(source).resolve().is_relative_to(run_dir.resolve())
                    for source in paths
                ):
                    has_sibling = True
                    break
            if has_sibling:
                break
        if not has_sibling and run_dir.is_dir():
            if run_dir.parent.resolve() == HISTORY_DIR.resolve():
                try:
                    shutil.rmtree(run_dir)
                except OSError as exc:
                    log.exception("Could not remove empty history run: %s", run_dir)
                    warnings.append(f"Could not remove shared run files: {exc}")
    return warnings


def _remove_history_entry(entry_id):
    with _HISTORY_LOCK:
        entries = _read_analysis_history()
        selected = next(
            ((index, entry) for index, entry in enumerate(entries)
             if isinstance(entry, dict)
             and _history_entry_id(entry, index) == entry_id),
            None,
        )
        if selected is None:
            raise ValueError("The selected analysis is no longer in History.")
        selected_index, removed_entry = selected
        remaining = entries[:selected_index] + entries[selected_index + 1:]
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        partial_path = HISTORY_FILE.with_name(HISTORY_FILE.name + ".partial")
        try:
            partial_path.write_text(
                json.dumps(remaining, ensure_ascii=False, indent=2,
                           default=_history_json_default),
                encoding="utf-8",
            )
            os.replace(partial_path, HISTORY_FILE)
        except Exception:
            log.exception("Could not update analysis history after removal")
            partial_path.unlink(missing_ok=True)
            raise

    warnings = _remove_history_artifacts(removed_entry, remaining)
    export_path = OUTPUT_DIR / "history_exports" / f"{safe_slug(entry_id)}.zip"
    if export_path.parent.resolve() == (OUTPUT_DIR / "history_exports").resolve():
        try:
            export_path.unlink(missing_ok=True)
        except OSError as exc:
            log.exception("Could not remove cached history ZIP: %s", export_path)
            warnings.append(f"Could not remove its cached ZIP: {exc}")
    return len(remaining), warnings


def _clear_analysis_history():
    with _HISTORY_LOCK:
        entries = _read_analysis_history()
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        partial_path = HISTORY_FILE.with_name(HISTORY_FILE.name + ".partial")
        try:
            partial_path.write_text("[]", encoding="utf-8")
            os.replace(partial_path, HISTORY_FILE)
        except Exception:
            log.exception("Could not clear analysis history")
            partial_path.unlink(missing_ok=True)
            raise

    warnings = []
    for index, entry in enumerate(entries):
        if isinstance(entry, dict):
            warnings.extend(_remove_history_artifacts(entry, []))
            entry_id = _history_entry_id(entry, index)
            export_path = OUTPUT_DIR / "history_exports" / f"{safe_slug(entry_id)}.zip"
            if export_path.parent.resolve() == (OUTPUT_DIR / "history_exports").resolve():
                try:
                    export_path.unlink(missing_ok=True)
                except OSError as exc:
                    log.exception("Could not remove cached history ZIP: %s", export_path)
                    warnings.append(f"Could not remove a cached ZIP: {exc}")
    return len(entries), warnings


def _history_detail_ui(entry_id):
    if not entry_id:
        empty = ('<section class="aee-history-empty"><strong>Select an analysis</strong>'
                 '<span>Choose a saved video above to browse its captures, summaries, and evidence details.</span></section>')
        return empty, [], [], pd.DataFrame(), pd.DataFrame()
    with _HISTORY_LOCK:
        entries = _read_analysis_history()
    entry = next(
        (record for index, record in enumerate(entries)
         if isinstance(record, dict) and _history_entry_id(record, index) == entry_id),
        None,
    )
    if entry is None:
        empty = ('<section class="aee-history-empty"><strong>Analysis not found</strong>'
                 '<span>Refresh History and choose an available saved analysis.</span></section>')
        return empty, [], [], pd.DataFrame(), pd.DataFrame()
    details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
    captures = [path for path in details.get("captures", [])
                if isinstance(path, str) and Path(path).is_file()]
    heatmaps = [path for path in details.get("heatmaps", [])
                if isinstance(path, str) and Path(path).is_file()]
    summary_files = [path for path in details.get("object_summaries", [])
                     if isinstance(path, str) and Path(path).is_file()]
    highlight_files = [path for path in details.get("highlights", [])
                       if isinstance(path, str) and Path(path).is_file()]
    other_files = [
        details.get(key) or entry.get(key, "")
        for key in ("master_video", "report_pdf", "analysis_json")
    ]
    files = [path for path in (*summary_files, *highlight_files, *other_files)
             if isinstance(path, str) and path and Path(path).is_file()]
    warnings = details.get("storage_warnings", [])
    warning_html = ""
    if warnings:
        warning_html = ('<div class="aee-history-warning"><strong>Some files were not saved</strong><ul>' +
                         "".join(f"<li>{_html.escape(str(item))}</li>" for item in warnings) +
                         "</ul></div>")
    source_path = str(entry.get("source_path") or "")
    try:
        analyzed = datetime.fromisoformat(str(entry.get("analyzed_at") or "")).astimezone().strftime(
            "%B %d, %Y at %H:%M")
    except (TypeError, ValueError):
        analyzed = str(entry.get("analyzed_at") or "Date unavailable")
    case_details = details.get("case_details") if isinstance(details.get("case_details"), dict) else {}
    rows = [
        ("Analyzed", analyzed),
        ("Duration", f"{float(entry.get('duration_seconds', 0) or 0):.2f} seconds"),
        ("Detections", str(entry.get("detections", 0))),
        ("Mode", "Enhanced accuracy" if entry.get("high_accuracy") else "Standard"),
        ("Case", str(entry.get("case") or "Not specified")),
        ("Original video", source_path or "Path unavailable (original video was not copied)"),
        ("Resolution", f"{details.get('width', '—')} × {details.get('height', '—')}"),
        ("Frame rate", f"{details.get('fps', '—')} fps"),
    ]
    for key, label in (("evidence_id", "Evidence ID"), ("examiner", "Examiner"),
                       ("agency", "Agency")):
        if case_details.get(key):
            rows.append((label, str(case_details[key])))
    metrics = "".join(
        f'<div class="aee-history-metric"><span>{_html.escape(label)}</span>'
        f'<strong>{_html.escape(value)}</strong></div>'
        for label, value in rows[:4]
    )
    info = "".join(
        f'<div><dt>{_html.escape(label)}</dt><dd>{_html.escape(value)}</dd></div>'
        for label, value in rows[4:]
    )
    object_stats = details.get("summary_stats") if isinstance(details.get("summary_stats"), dict) else {}
    stats_html = "".join(
        f'<div class="aee-history-stat"><strong>{_html.escape(str(label))}</strong>'
        f'<span>{_html.escape(str(values.get("clips", 0)))} clips · '
        f'{_html.escape(str(values.get("duration", 0)))} sec · '
        f'avg {_html.escape(str(values.get("avg_confidence", 0)))}</span></div>'
        for label, values in object_stats.items() if isinstance(values, dict)
    )
    notes = str(details.get("analysis_notes") or "")
    notes_html = (f'<p class="aee-history-notes">{_html.escape(notes)}</p>' if notes else "")
    case_notes = str(details.get("case_notes") or "")
    case_notes_html = (f'<p class="aee-history-notes"><strong>Case notes:</strong> '
                       f'{_html.escape(case_notes)}</p>' if case_notes else "")
    stats_section = (f'<div class="aee-history-stat-list">{stats_html}</div>'
                     if stats_html else "")
    html = (f'<section class="aee-history-detail">{metrics}'
            f'<dl class="aee-history-metadata">{info}</dl>'
            f'{stats_section}'
            f'{notes_html}{case_notes_html}{warning_html}</section>')
    detection_rows = details.get("detections", [])
    detection_df = pd.DataFrame(detection_rows) if detection_rows else pd.DataFrame()
    track_rows = details.get("tracks", [])
    track_df = pd.DataFrame(track_rows) if track_rows else pd.DataFrame()
    return html, captures + heatmaps, files, detection_df, track_df


def _history_zip(entry_id=None):
    with _HISTORY_LOCK:
        entries = [
            (index, entry)
            for index, entry in enumerate(_read_analysis_history())
            if isinstance(entry, dict)
        ]
    if entry_id:
        selected = next(
            ((index, entry) for index, entry in entries
             if _history_entry_id(entry, index) == entry_id),
            None,
        )
        if selected is None:
            raise ValueError("The selected analysis is no longer in History.")
        entries = [selected]
    if not entries:
        raise ValueError("There are no saved analyses to export.")

    export_dir = OUTPUT_DIR / "history_exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    archive_name = (
        f"{safe_slug(entry_id)}.zip" if entry_id
        else "all-analysis-history.zip"
    )
    archive_path = export_dir / archive_name
    partial_path = archive_path.with_suffix(".partial")
    output_root = OUTPUT_DIR.resolve()
    exported_records = []

    try:
        with zipfile.ZipFile(partial_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for position, (history_index, entry) in enumerate(entries, 1):
                entry_id_value = _history_entry_id(entry, history_index)
                video_name = safe_slug(Path(str(entry.get("video") or "video")).stem)
                record_dir = (
                    f"AI Evidence Extractor History/Analyses/"
                    f"{position:04d}-{video_name}-{safe_slug(entry_id_value)[-12:]}"
                )
                record = dict(entry)
                details = dict(entry.get("details") or {})
                artifact_paths = {
                    "captures": details.get("captures", []),
                    "heatmaps": details.get("heatmaps", []),
                    "highlights": details.get("highlights", []),
                    "object_summaries": details.get("object_summaries", []),
                    "master_video": [details.get("master_video") or entry.get("master_video")],
                    "report_pdf": [details.get("report_pdf") or entry.get("report_pdf")],
                }
                archived = {}
                missing = []
                for key, paths in artifact_paths.items():
                    if not isinstance(paths, list):
                        paths = [paths]
                    archived_paths = []
                    for path_index, source in enumerate(paths, 1):
                        if not isinstance(source, str) or not source:
                            continue
                        source_path = Path(source).resolve()
                        if not source_path.is_relative_to(output_root):
                            missing.append(f"{key}: file is outside app output storage")
                            continue
                        if not source_path.is_file():
                            missing.append(f"{key}: {source_path.name} is missing")
                            continue
                        category = {
                            "object_summaries": "summaries",
                            "master_video": "reports",
                            "report_pdf": "reports",
                        }.get(key, key)
                        filename = f"{path_index:04d}-{safe_slug(source_path.name)}"
                        arcname = f"{record_dir}/{category}/{filename}"
                        archive.write(source_path, arcname=arcname)
                        archived_paths.append(arcname)
                    archived[key] = archived_paths

                for key in ("captures", "heatmaps", "highlights", "object_summaries"):
                    details[key] = archived[key]
                details["master_video"] = archived["master_video"][0] if archived["master_video"] else ""
                details["report_pdf"] = archived["report_pdf"][0] if archived["report_pdf"] else ""
                details["analysis_json"] = f"{record_dir}/analysis.json"
                details["zip_missing_artifacts"] = missing
                record["details"] = details
                analysis_json = json.dumps(
                    record, ensure_ascii=False, indent=2,
                    default=_history_json_default)
                archive.writestr(details["analysis_json"], analysis_json)
                exported_records.append({
                    "id": entry_id_value,
                    "video": entry.get("video", ""),
                    "analysis_json": details["analysis_json"],
                    "missing_artifacts": missing,
                })

            archive.writestr(
                "AI Evidence Extractor History/history_index.json",
                json.dumps(exported_records, ensure_ascii=False, indent=2),
            )
        os.replace(partial_path, archive_path)
    except Exception:
        log.exception("Could not create analysis history ZIP")
        try:
            partial_path.unlink(missing_ok=True)
        except OSError:
            log.exception("Could not remove partial analysis history ZIP")
        raise
    return str(archive_path.resolve()), len(entries), sum(
        len(record["missing_artifacts"]) for record in exported_records)


def _history_zip_ui(entry_id=None):
    try:
        archive_path, entry_count, missing_count = _history_zip(entry_id)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        log.exception("History ZIP export failed")
        status = (
            '<div class="aee-history-warning"><strong>ZIP export failed</strong> '
            f'{_html.escape(str(exc))}</div>'
        )
        return None, status
    status = (
        f'<div class="aee-history-export-status">ZIP ready: '
        f'{entry_count} analysis{"es" if entry_count != 1 else ""} included.'
    )
    if missing_count:
        status += f' {missing_count} generated file(s) were unavailable.</div>'
    else:
        status += '</div>'
    return archive_path, status


def _history_remove_ui(entry_id, refresh_count=0):
    try:
        remaining_count, cleanup_warnings = _remove_history_entry(entry_id)
    except (OSError, ValueError) as exc:
        log.exception("Could not remove analysis history entry")
        return (
            refresh_count,
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            None,
            '<div class="aee-history-warning"><strong>Could not remove analysis</strong> '
            f'{_html.escape(str(exc))}</div>',
        )
    status = (
        '<div class="aee-history-export-status">Analysis removed. '
        f'{remaining_count} saved analysis{"es" if remaining_count != 1 else ""} remain.'
    )
    if cleanup_warnings:
        status += (
            '<div class="aee-history-warning"><strong>Some generated files need cleanup</strong><ul>'
            + "".join(f"<li>{_html.escape(warning)}</li>" for warning in cleanup_warnings)
            + "</ul></div>"
        )
    status += "</div>"
    return (
        refresh_count + 1,
        '<section class="aee-history-empty"><strong>Analysis removed</strong>'
        '<span>Select another saved video to inspect its evidence and reports.</span></section>',
        [],
        [],
        pd.DataFrame(),
        pd.DataFrame(),
        None,
        status,
    )


def _history_clear_ui(refresh_count=0):
    try:
        removed_count, cleanup_warnings = _clear_analysis_history()
    except OSError as exc:
        log.exception("Could not clear analysis history")
        return (
            refresh_count,
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            '<div class="aee-history-warning"><strong>Could not clear history</strong> '
            f'{_html.escape(str(exc))}</div>',
        )
    status = f'<div class="aee-history-export-status">{removed_count} saved analysis(es) cleared.'
    if cleanup_warnings:
        status += (
            '<div class="aee-history-warning"><strong>Some generated files need cleanup</strong><ul>'
            + "".join(f"<li>{_html.escape(warning)}</li>" for warning in cleanup_warnings)
            + "</ul></div>"
        )
    status += "</div>"
    return (
        refresh_count + 1,
        '<section class="aee-history-empty"><strong>History cleared</strong>'
        '<span>Completed video analyses will appear here.</span></section>',
        [],
        [],
        pd.DataFrame(),
        pd.DataFrame(),
        status,
    )


def _history_refresh_ui(refresh_count=0):
    return refresh_count + 1


def _copy_history_artifact(source, destination):
    if not source:
        return ""
    source_path = Path(source)
    if not source_path.is_file() or source_path.stat().st_size <= 0:
        return ""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)
    return str(destination.resolve())


def _history_json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported history value: {type(value).__name__}")


def _record_analysis_history(results, master_path, report_path, case_number=None,
                             high_accuracy=False, *, case_meta=None, case_notes="",
                             tracker_rows=None, cam_items=None, work_root=None):
    if not results:
        return []
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    run_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:10]
    run_dir = HISTORY_DIR / run_id
    storage_warnings = []
    try:
        shared_dir = run_dir / "reports"
        saved_master = _copy_history_artifact(
            master_path, shared_dir / Path(master_path).name) if master_path else ""
        saved_report = _copy_history_artifact(
            report_path, shared_dir / Path(report_path).name) if report_path else ""
    except OSError as exc:
        log.exception("Could not save shared history report artifacts")
        storage_warnings.append(f"Report artifact copy failed: {exc}")
        saved_master = saved_report = ""
    additions = []
    for index, result in enumerate(results, 1):
        video_dir = run_dir / f"{index:03d}-{safe_slug(Path(result.video_name).stem)[:60]}"
        video_warnings = list(storage_warnings)

        def copy_group(paths, group):
            saved = []
            for asset_index, path in enumerate(paths, 1):
                try:
                    target = video_dir / group / f"{asset_index:04d}-{Path(path).name}"
                    copied = _copy_history_artifact(path, target)
                    if copied:
                        saved.append(copied)
                    elif path:
                        video_warnings.append(f"Could not find generated file: {Path(path).name}")
                except OSError as exc:
                    log.exception("Could not store history artifact %s", path)
                    video_warnings.append(f"Could not save {Path(path).name}: {exc}")
            return saved

        saved_captures = copy_group(result.captures, "captures")
        saved_summaries = copy_group(list(result.object_summaries.values()), "summaries")
        video_cam_items = [item for item in (cam_items or [])
                           if item.get("video") == result.video_name]
        saved_heatmaps = copy_group([item.get("heat") for item in video_cam_items], "heatmaps")
        saved_heatmaps += copy_group([item.get("blend") for item in video_cam_items], "heatmaps")
        highlights_dir = Path(work_root) / sname(result.video_name) / "highlights" if work_root else None
        saved_highlights = copy_group(
            sorted(highlights_dir.glob("*.mp4")) if highlights_dir and highlights_dir.is_dir() else [],
            "highlights",
        )
        video_tracks = [row for row in (tracker_rows or [])
                        if row.get("Video") == result.video_name]
        details = {
            "source_path": result.source_path,
            "analysis_notes": result.notes,
            "case_details": dict(case_meta or {}),
            "case_notes": case_notes,
            "width": result.metadata.width,
            "height": result.metadata.height,
            "fps": result.metadata.fps,
            "frame_count": result.metadata.frame_count,
            "summary_stats": result.summary_stats,
            "detections": result.detections,
            "tracks": video_tracks,
            "captures": saved_captures,
            "heatmaps": saved_heatmaps,
            "highlights": saved_highlights,
            "object_summaries": saved_summaries,
            "master_video": saved_master,
            "report_pdf": saved_report,
            "storage_warnings": video_warnings,
        }
        entry = {
            "id": f"{run_id}-{index:03d}",
            "analyzed_at": now,
            "video": result.video_name,
            "source_path": result.source_path,
            "duration_seconds": round(float(result.metadata.duration or 0), 2),
            "detections": len(result.detections),
            "high_accuracy": bool(high_accuracy),
            "case": (case_number or "").strip(),
            "master_video": saved_master,
            "report_pdf": saved_report,
            "details": details,
        }
        try:
            video_dir.mkdir(parents=True, exist_ok=True)
            analysis_json = video_dir / "analysis.json"
            entry["details"]["analysis_json"] = str(analysis_json.resolve())
            analysis_json.write_text(
                json.dumps(entry, ensure_ascii=False, indent=2,
                           default=_history_json_default), encoding="utf-8")
        except (OSError, TypeError, ValueError) as exc:
            log.exception("Could not save full history details for %s", result.video_name)
            entry["details"]["storage_warnings"].append(
                f"Could not save the analysis detail file: {exc}")
        additions.append(entry)
    with _HISTORY_LOCK:
        entries = [entry for entry in _read_analysis_history()
                   if isinstance(entry, dict)]
        entries.extend(additions)
        partial = HISTORY_FILE.with_suffix(".partial")
        try:
            HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            partial.write_text(json.dumps(
                entries, ensure_ascii=False, indent=2,
                default=_history_json_default),
                               encoding="utf-8")
            os.replace(partial, HISTORY_FILE)
        except Exception as exc:
            log.exception("Could not save analysis history")
            try:
                partial.unlink(missing_ok=True)
            except OSError:
                pass
            return [f"History index could not be saved: {exc}"]
    return [
        f"{entry['video']}: {warning}"
        for entry in additions
        for warning in entry.get("details", {}).get("storage_warnings", [])
    ]


class CFG:
    IOU, IMG, FPS = 0.45, 640, 1.0
    MERGE_IOU, TRACKER = 0.5, "bytetrack.yaml"
    MIN_AREA, FUSION_MIN, CAP_MAX = 150, 0.35, 500
    INFERENCE_DEVICE = "auto"  # "auto", "cpu", or "cuda"
    INFERENCE_HALF = True      # use FP16 on CUDA when enabled
    MAX_ACTIVE_MODELS = 0      # 0 = all; otherwise use first N manifest models
    CLIP_PAD, HL_PAD, HL_THRESH = 2.0, 5.0, 6
    W_WPN, W_MULTI, W_AUD = 15, 25, 6
    W_WPN_PERSON, W_WPN_FACE, W_PERSON_ONLY = 20, 15, -5
    MERGE_W, MERGE_H, MERGE_FPS = 1280, 720, 30
    SUPPORTED = {".mp4",".mov",".avi",".mkv",".flv",".wmv",".m4v",".webm",
                 ".mpg",".mpeg",".3gp",".3g2",".mts",".m2ts",".ts",".vob",
                 ".ogv",".rm",".rmvb",".asf",".divx",".f4v",".mxf"}
    SAMPLE_FPS_EST = 0.35
    CAM_MAX = 48
    PER_CLASS_CONF = {"Firearm": 0.59}
    PER_MODEL_CONF = {
        "weapon_detection": 0.71,
        "weapon_yolov8":   0.75,
        "firearm_yolov8n": 0.75,
        "cypher_weapon":   {"__weapons__": 0.50},
    }


CMAP = {"gun":"Firearm","pistol":"Firearm","handgun":"Firearm","firearm":"Firearm",
        "rifle":"Firearm","shotgun":"Firearm","revolver":"Firearm","weapon":"Weapon",
        "knife":"Knife","blade":"Knife","dagger":"Knife","machete":"Knife",
        "melee_weapon":"Knife","sword":"Knife","blunt_weapon":"Blunt","bat":"Blunt",
        "club":"Blunt","explosive":"Explosive","grenade":"Explosive","bomb":"Explosive",
        "fire_smoke":"Fire/Smoke",
        "face":"Face","person":"Person","human":"Person","people":"Person"}
WEAPONS = {"Firearm","Knife","Blunt","Explosive","Fire/Smoke","Weapon"}
CONTEXT = {"Person", "Face"}

WEAPON_ICONS = {"Firearm":"","Knife":"","Blunt":"","Explosive":"",
                "Fire/Smoke":"","Weapon":"",
                "Person":"","Face":"","Loud Sound":""}
WEAPON_COLORS = {"Firearm":"#ef4444","Knife":"#f59e0b","Blunt":"#eab308",
                 "Explosive":"#dc2626","Fire/Smoke":"#f97316","Weapon":"#ef4444",
                 "Person":"#8b5cf6","Face":"#a78bfa","Loud Sound":"#22d3ee"}
WEAPON_THREAT = {"Firearm":"critical","Explosive":"critical",
                 "Knife":"high","Weapon":"high","Fire/Smoke":"high","Blunt":"medium"}
THREAT_ORDER = {"critical":4,"high":3,"medium":2,"low":1,"info":0}
THREAT_COLORS = {"critical":"#dc2626","high":"#ef4444","medium":"#f59e0b",
                 "low":"#10b981","info":"#64748b"}

def norm(r): return CMAP.get(str(r).lower().strip(), str(r).title())
def icon_for(label): return WEAPON_ICONS.get(label, "◈")
def color_for(label): return WEAPON_COLORS.get(label, "#8b5cf6")


def threat_score(d, tracker=None, video_name=None):
    lbl = d.get("label", "")
    if lbl not in WEAPONS:
        return "info", 0, []
    reasons = []
    base = WEAPON_THREAT.get(lbl, "low")
    score = {"critical":55,"high":38,"medium":22,"low":10}.get(base, 0)
    reasons.append(f"class · {lbl}")
    conf = float(d.get("confidence", 0))
    score += conf * 22
    if conf >= 0.7: reasons.append(f"high conf · {conf:.2f}")
    fus = float(d.get("fusion_prob", 1.0))
    score += (fus - 0.5) * 20
    if fus >= 0.85: reasons.append(f"strong fusion · {fus:.2f}")
    if tracker is not None and video_name and d.get("track_id", -1) >= 0:
        tr = tracker.tracks.get((video_name, d["track_id"], lbl))
        if tr:
            dur = tr["last_seen"] - tr["first_seen"]
            if dur >= 5: score += 18; reasons.append(f"persistent · {dur:.1f}s")
            elif dur >= 2: score += 9; reasons.append(f"persistent · {dur:.1f}s")
    if score >= 80: level = "critical"
    elif score >= 60: level = "high"
    elif score >= 40: level = "medium"
    else: level = "low"
    return level, int(min(100, round(score))), reasons


def ts(s):
    s = max(0, float(s)); h, r = divmod(int(s), 3600); m, sec = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{sec:06.3f}" if h else f"{m:02d}:{sec:06.3f}"

def fdur(s):
    s = max(0, int(s))
    if s < 60: return f"{s}s"
    m, s = divmod(s, 60)
    return f"{m}m {s}s" if m < 60 else f"{m//60}h {m%60}m"

def fbytes(n):
    for u in ("B","KB","MB","GB","TB"):
        if n < 1024: return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"

def sname(p):
    return re.sub(r"[^a-zA-Z0-9._-]+","_", Path(p).stem).strip("_") or "video"

def safe_slug(p):
    return re.sub(r"[^a-zA-Z0-9._-]+","_", str(p).strip()).strip("_")

def _quiet_subprocess_kwargs():
    """Keep FFmpeg/FFprobe child consoles hidden on Windows."""
    if os.name != "nt":
        return {}
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    return {
        "creationflags": subprocess.CREATE_NO_WINDOW,
        "startupinfo": startupinfo,
    }

def iou(a, b):
    xA,yA=max(a[0],b[0]),max(a[1],b[1]); xB,yB=min(a[2],b[2]),min(a[3],b[3])
    iw,ih=max(0,xB-xA),max(0,yB-yA); i=iw*ih
    if i==0: return 0.
    return i/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i+1e-9)

def sh(cmd, cancel_event=None):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace",
                            **_quiet_subprocess_kwargs())
    while True:
        try:
            stdout, stderr = proc.communicate(timeout=0.25); break
        except subprocess.TimeoutExpired:
            if cancel_event is not None and cancel_event.is_set():
                proc.terminate()
                try: stdout, stderr = proc.communicate(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill(); stdout, stderr = proc.communicate()
                raise InterruptedError("Analysis cancelled by the user.")
    result = subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
    if result.returncode: raise RuntimeError(f"ffmpeg: {result.stderr[-400:]}")
    return result

def has_audio(p):
    try:
        r = subprocess.run(["ffprobe","-v","error","-select_streams","a:0",
            "-show_entries","stream=codec_type","-of","csv=p=0",str(p)],
            capture_output=True, text=True, timeout=10,
            **_quiet_subprocess_kwargs())
        return "audio" in r.stdout.lower()
    except Exception:
        return False

def probe_dur(p):
    try:
        r = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
            "-of","csv=p=0",str(p)], capture_output=True, text=True, timeout=10,
            **_quiet_subprocess_kwargs())
        return float(r.stdout.strip() or 0)
    except Exception:
        return 0.

def probe_video(path):
    info = {"path":str(path),"name":Path(path).name,"size":0,"duration":0.,
            "width":0,"height":0,"fps":0.,"codec":"?","has_audio":False,
            "valid":False,"error":None}
    try:
        p = Path(path)
        if not p.exists(): info["error"] = "File missing"; return info
        info["size"] = p.stat().st_size
        r = subprocess.run(["ffprobe","-v","error","-show_entries",
                    "format=duration:stream=width,height,r_frame_rate,codec_name,codec_type",
                    "-of","json", str(p)], capture_output=True, text=True, timeout=15,
                    **_quiet_subprocess_kwargs())
        data = json.loads(r.stdout or "{}")
        info["duration"] = float(data.get("format", {}).get("duration", 0) or 0)
        for s in data.get("streams", []):
            if s.get("codec_type") == "video" and info["width"] == 0:
                info["width"] = int(s.get("width", 0) or 0)
                info["height"] = int(s.get("height", 0) or 0)
                info["codec"] = s.get("codec_name", "?")
                fr = s.get("r_frame_rate", "0/1")
                try:
                    num, den = fr.split("/"); info["fps"] = float(num)/max(1., float(den))
                except Exception: pass
            elif s.get("codec_type") == "audio":
                info["has_audio"] = True
        info["valid"] = info["width"] > 0 and info["duration"] > 0
    except Exception as e:
        info["error"] = str(e)[:120]
    return info

def merge_ranges(events, dur, pad):
    if not events: return []
    rng = sorted((max(0, e["timestamp"]-pad), min(dur, e["timestamp"]+pad)) for e in events)
    out = [list(rng[0])]
    for s, e in rng[1:]:
        if s <= out[-1][1]: out[-1][1] = max(out[-1][1], e)
        else: out.append([s, e])
    return [(round(s,3), round(e,3)) for s, e in out]


def _resolve_conf_gate(model_name, label):
    mo = CFG.PER_MODEL_CONF.get(model_name)
    gate = None
    if isinstance(mo, (int, float)):
        gate = float(mo)
    elif isinstance(mo, dict):
        if label in mo:
            gate = float(mo[label])
        elif "__weapons__" in mo and label in WEAPONS:
            gate = float(mo["__weapons__"])
    if gate is None:
        gate = CFG.PER_CLASS_CONF.get(label)
    return gate


def _resolve_inference_device():
    requested = str(CFG.INFERENCE_DEVICE).strip().lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("CFG.INFERENCE_DEVICE must be 'auto', 'cpu', or 'cuda'")
    if requested == "cpu":
        return "cpu", False
    try:
        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False
    if not cuda_available:
        if requested == "cuda":
            log.warning("CUDA requested but unavailable; using CPU inference")
        return "cpu", False
    return 0, bool(CFG.INFERENCE_HALF)


class Ensemble:
    _i = None
    def __init__(self):
        self.models = []
        self.device, self.half = _resolve_inference_device()
        try:
            self.device_name = (
                torch.cuda.get_device_name(self.device)
                if self.device != "cpu" else "CPU"
            )
        except Exception as exc:
            log.warning("Could not identify the inference device: %s", exc)
            self.device_name = str(self.device)
        if MANIFEST.exists():
            for e in json.loads(MANIFEST.read_text()):
                model_path = _resolve_model_path(e["path"])
                if not model_path.is_file():
                    log.warning("model file is missing: %s", model_path)
                    continue
                try:
                    m = YOLO(str(model_path))
                    size_mb = model_path.stat().st_size / 1e6
                    self.models.append({
                        "yolo": m, "name": e["name"],
                        "conf": e.get("conf", 0.15),
                        "base_conf": e.get("conf", 0.15),
                        "names": m.names,
                        "classes": list(m.names.values()) if isinstance(m.names, dict)
                                   else list(m.names),
                        "size_mb": round(size_mb, 1),
                        "source": e.get("source", ""),
                        "enabled": True, "infer_ms": 0.0,
                    })
                except Exception as ex:
                    log.warning("load %s: %s", e["name"], ex)
        model_limit = max(0, int(CFG.MAX_ACTIVE_MODELS))
        for index, model in enumerate(self.models):
            model["enabled"] = model_limit == 0 or index < model_limit
        active_count = sum(1 for model in self.models if model["enabled"])
        precision = "FP16" if self.half else "FP32"
        log.info("Ensemble: %d/%d active models · device=%s (%s) · %s",
                 active_count, len(self.models), self.device_name,
                 self.device, precision)

    @classmethod
    def get(cls):
        if cls._i is None: cls._i = cls()
        return cls._i

    def reset(self):
        for m in self.models:
            try:
                model = m["yolo"]
                if hasattr(model, "predictor"):
                    model.predictor = None
            except Exception as e:
                log.debug("reset %s: %s", m["name"], e)

    def set_enabled(self, names_enabled):
        active = set(names_enabled or [])
        for m in self.models: m["enabled"] = m["name"] in active

    def detect(self, frame, high_accuracy=False):
        out = []
        for e in self.models:
            if not e.get("enabled", True): continue
            t0 = time.perf_counter()
            try:
                # The application assigns cross-model track IDs in Tracker below.
                # Per-model Ultralytics tracking duplicates that work and its IDs
                # are discarded when results from the ensemble are merged.
                r = e["yolo"].predict(source=frame, conf=e["conf"], iou=CFG.IOU,
                    imgsz=CFG.IMG, verbose=False,
                    device=self.device, half=self.half,
                    augment=bool(high_accuracy))[0]
            except Exception as ex:
                log.debug("%s: %s", e["name"], ex); continue
            e["infer_ms"] = (time.perf_counter() - t0) * 1000.
            if r.boxes is None: continue
            for b in r.boxes:
                cid = int(b.cls[0]); nm = e["names"]
                raw = nm.get(cid, cid) if isinstance(nm, dict) else nm[cid]
                label = norm(raw)
                conf = float(b.conf[0])
                gate = _resolve_conf_gate(e["name"], label)
                if gate is not None and conf < gate:
                    continue
                out.append({"label": label, "confidence": conf,
                    "bbox": tuple(map(int, b.xyxy[0].tolist())),
                    "source": e["name"],
                    "track_id": -1})
        return out

    @property
    def info(self):
        active = [m for m in self.models if m.get("enabled", True)]
        return {"loaded": bool(active), "n_models": len(active),
                "n_total": len(self.models),
                "device": self.device_name,
                "precision": "FP16" if self.half else "FP32",
                "names": [m["name"] for m in active],
                "classes": sorted({n for m in active for n in m["classes"]}),
                "members": [{"name": m["name"], "enabled": m["enabled"],
                             "conf": m["conf"], "size_mb": m["size_mb"],
                             "classes": m["classes"], "source": m["source"],
                             "infer_ms": m["infer_ms"]} for m in self.models]}


class Tracker:
    def __init__(self):
        self.tracks = {}; self.frames = 0; self.reid_events = []
        self._last_timestamp_seen = {}; self._next_id = 1

    def update(self, dets, t, frame_idx, video_name):
        """Assign stable IDs to merged ensemble detections using box association.

        The ensemble uses predict() for each model, so model-local tracker IDs do
        not exist. Associate detections across sampled frames here instead.
        """
        self.frames += 1
        max_gap = max(3.0, 3.0 / max(float(CFG.FPS), 0.1))
        candidates = []
        for d_idx, d in enumerate(dets):
            box = d.get("bbox")
            if not box or len(box) != 4:
                continue
            for key, tr in self.tracks.items():
                if key[0] != video_name or tr["label"] != d.get("label"):
                    continue
                gap = float(t) - tr["last_seen"]
                if gap < 0 or gap > max_gap:
                    continue
                prev_box = tr["history"][-1][1]
                overlap = iou(box, prev_box)
                cx = (box[0] + box[2]) / 2.0; cy = (box[1] + box[3]) / 2.0
                px = (prev_box[0] + prev_box[2]) / 2.0
                py = (prev_box[1] + prev_box[3]) / 2.0
                scale = max(1.0, math.hypot(prev_box[2]-prev_box[0],
                                            prev_box[3]-prev_box[1]))
                distance = math.hypot(cx-px, cy-py) / scale
                if overlap >= 0.05 or distance <= 0.8:
                    score = 0.65 * overlap + 0.35 * max(0.0, 1.0-distance)
                    candidates.append((score, d_idx, key))

        assigned_dets = set(); assigned_tracks = set(); matches = {}
        for score, d_idx, key in sorted(candidates, reverse=True):
            if score < 0.30 or d_idx in assigned_dets or key in assigned_tracks:
                continue
            matches[d_idx] = key
            assigned_dets.add(d_idx); assigned_tracks.add(key)

        for d_idx, d in enumerate(dets):
            if not d.get("bbox") or len(d["bbox"]) != 4:
                d["track_id"] = -1
                continue
            key = matches.get(d_idx)
            if key is None:
                tid = self._next_id; self._next_id += 1
                key = (video_name, tid, d["label"])
                self.tracks[key] = {"video":video_name,"track_id":tid,
                    "label":d["label"],"source":d.get("source", "ensemble"),
                    "first_seen":t,"last_seen":t,"frames":0,"conf_sum":0.0,
                    "max_conf":0.0,"history":[]}
            tr = self.tracks[key]
            prev = self._last_timestamp_seen.get(key)
            if prev is not None and (t-prev) > max(1.0, 1.25/max(float(CFG.FPS), 0.1)):
                self.reid_events.append({"video":video_name,"track_id":tr["track_id"],
                    "label":d["label"],"at":t,"gap_sec":t-prev})
            tr["last_seen"] = t; tr["frames"] += 1
            tr["conf_sum"] += float(d.get("confidence", 0.0))
            tr["max_conf"] = max(tr["max_conf"], float(d.get("confidence", 0.0)))
            tr["history"].append((t, tuple(d["bbox"])))
            self._last_timestamp_seen[key] = t
            d["track_id"] = tr["track_id"]

    def summaries(self):
        rows = []
        for (vn, tid, lbl), tr in self.tracks.items():
            dur = tr["last_seen"] - tr["first_seen"]
            avg = tr["conf_sum"] / max(1, tr["frames"])
            _proxy = {"label":lbl,"confidence":tr["max_conf"],
                      "fusion_prob":1.0,"track_id":tid}
            lvl, sc, _ = threat_score(_proxy, self, vn)
            rows.append({"Video":vn,"Track":tid,"Class":lbl,"Source":tr["source"],
                "First":ts(tr["first_seen"]),"Last":ts(tr["last_seen"]),
                "Duration_s":round(dur,2),"Frames":tr["frames"],
                "Avg_Conf":round(avg,3),"Max_Conf":round(tr["max_conf"],3),
                "Threat":lvl.title(),"Risk":sc,"Weapon":lbl in WEAPONS})
        return rows

    def panel_html(self, video_filter=None):
        rows = self.summaries()
        if video_filter and video_filter != "All":
            rows = [r for r in rows if r["Video"] == video_filter]
        if not rows:
            return ('<div class="aee-empty"><div class="aee-empty-icon">🧭</div>'
                    '<div class="aee-empty-title">No tracks yet</div>'
                    '<div class="aee-empty-sub">Object tracks appear here after analysis.'
                    '</div></div>')
        n_tracks = len(rows)
        n_weapon = sum(1 for r in rows if r["Weapon"])
        n_long = sum(1 for r in rows if r["Duration_s"] >= 3.0)
        avg_dur = sum(r["Duration_s"] for r in rows) / n_tracks
        longest = max(rows, key=lambda r: r["Duration_s"])
        reids = len(self.reid_events)
        max_dur = max(r["Duration_s"] for r in rows) or 1.0
        bars = ""
        for r in sorted(rows, key=lambda x: -x["Duration_s"])[:40]:
            pct = min(100, r["Duration_s"] / max_dur * 100)
            cls = "wpn" if r["Weapon"] else "ctx"
            ic = icon_for(r["Class"])
            bars += (f'<div class="aee-trackbar {cls}">'
                     f'<span class="aee-trackbar-id">#{r["Track"]}</span>'
                     f'<span class="aee-trackbar-lbl">{ic} {r["Class"]}</span>'
                     f'<div class="aee-trackbar-outer">'
                     f'<div class="aee-trackbar-fill" style="width:{pct:.1f}%"></div>'
                     f'</div>'
                     f'<span class="aee-trackbar-dur">{r["Duration_s"]:.2f}s</span>'
                     f'<span class="aee-trackbar-frames">{r["Frames"]}f</span>'
                     f'</div>')
        return f"""
        <div class="aee-track-stats">
          <div class="aee-stat"><span class="aee-stat-icon">🧭</span>
            <div class="aee-stat-val">{n_tracks}</div>
            <div class="aee-stat-lbl">Unique tracks</div></div>
          <div class="aee-stat"><span class="aee-stat-icon">🎯</span>
            <div class="aee-stat-val">{n_weapon}</div>
            <div class="aee-stat-lbl">Weapon tracks</div></div>
          <div class="aee-stat"><span class="aee-stat-icon">⏱</span>
            <div class="aee-stat-val">{avg_dur:.2f}s</div>
            <div class="aee-stat-lbl">Avg duration</div></div>
          <div class="aee-stat"><span class="aee-stat-icon">🔁</span>
            <div class="aee-stat-val">{reids}</div>
            <div class="aee-stat-lbl">Re-ID events</div></div>
        </div>
        <div class="aee-track-summary">
          <div>Longest track: <b>#{longest['Track']} · {longest['Class']}
            · {longest['Duration_s']:.2f}s</b></div>
          <div>Persistent (&gt;3s): <b>{n_long}</b></div>
        </div>
        <div class="aee-track-timeline">
          <h4>Track Lifetimes <span>one bar per track · widest = longest-lived</span></h4>
          {bars}
        </div>"""


class Fusion:
    PATH = MODEL_DIR / "fusion_clf.pkl"
    def __init__(self):
        self.clf = None
        if self.PATH.exists():
            try: self.clf = pickle.loads(self.PATH.read_bytes())
            except Exception as e: log.warning("fusion load: %s", e)
    def prob(self, d, cluster):
        if self.clf is None: return 1.0
        try:
            confs = [c["confidence"] for c in cluster]
            same = sum(1 for c in cluster if c["label"] == d["label"])
            b = d["bbox"]; w, h = b[2]-b[0], b[3]-b[1]
            X = np.array([[len(cluster), max(confs), sum(confs)/len(confs),
                same/len(cluster), int(d["label"] in WEAPONS),
                (w*h)/(640*640), w/max(h,1)]])
            return float(self.clf.predict_proba(X)[0, 1])
        except Exception:
            return 1.0

FUSION = Fusion()

_UPLOAD_CACHE = {}
_ACTIVE_CANCEL = threading.Event()
_ACTIVE_LOCK = threading.Lock()
_BATCH_LOCK = threading.Lock()


def _stable_copy(src_path):
    if not src_path: return None
    p = Path(src_path)
    if not p.is_file(): return None
    stable_dir = APP_DATA / "work" / "uploads"
    try:
        digest = hashlib.sha256()
        with p.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        file_dir = stable_dir / digest.hexdigest()[:20]
        file_dir.mkdir(parents=True, exist_ok=True)
        dest = file_dir / p.name
        if not dest.exists() or dest.stat().st_size != p.stat().st_size:
            shutil.copy2(p, dest)
        _UPLOAD_CACHE[digest.hexdigest()] = str(dest)
        return str(dest)
    except Exception:
        log.exception("Could not preserve uploaded video %s", p.name)
        return None


def _cluster(dets):
    cl = []
    for d in sorted(dets, key=lambda x: -x["confidence"]):
        hit = False
        for c in cl:
            if any(iou(d["bbox"], x["bbox"]) > 0.4 for x in c):
                c.append(d); hit = True; break
        if not hit: cl.append([d])
    return cl

def merge_and_filter(dets):
    if not dets: return []
    dets = [d for d in dets
            if (d["bbox"][2]-d["bbox"][0])*(d["bbox"][3]-d["bbox"][1]) >= CFG.MIN_AREA]
    kept = []
    for c in _cluster(dets):
        by = defaultdict(list)
        for d in c: by[d["label"]].append(d)
        for lbl, g in by.items():
            best = max(g, key=lambda x: x["confidence"])
            p = FUSION.prob(best, c)
            if p >= CFG.FUSION_MIN:
                b = dict(best); b["fusion_prob"] = round(p, 3); kept.append(b)
    return kept


try:
    from pytorch_grad_cam import EigenCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    _CAM_OK = True
except Exception as _e:
    log.warning("pytorch_grad_cam unavailable: %s", _e)
    _CAM_OK = False

_CAM_LAYER_CACHE = {}
_CAM_RESULT_UNSET = object()


class _CAMModelWrapper(torch.nn.Module):
    def __init__(self, detection_model):
        super().__init__()
        self.detection_model = detection_model
    def forward(self, x):
        out = self.detection_model(x)
        while isinstance(out, (tuple, list)) and len(out) > 0:
            out = out[0]
        return out


def _candidate_layers(model):
    seq = model.model.model
    n = len(seq)
    for i in [n-2, n-3, n-4, n-5, n-6, n-7, n-8, n-10]:
        if 0 <= i < n: yield i


def _reshape(t, h=40, w=40):
    while isinstance(t, (tuple, list)) and len(t) > 0:
        t = t[0]
    if t is None or not isinstance(t, torch.Tensor):
        raise ValueError(f"_reshape: expected Tensor, got {type(t).__name__}")
    if len(t.shape) == 3:
        t = t.reshape(t.size(0), h, w, t.size(2)).permute(0, 3, 1, 2)
    return t


def grad_cam(model, frame, bbox=None):
    if not _CAM_OK: return None, None
    key = (id(model), len(model.model.model))
    seq = model.model.model
    rgb = cv2.cvtColor(cv2.resize(frame, (640, 640)), cv2.COLOR_BGR2RGB)
    rf  = np.float32(rgb) / 255.
    tensor = torch.from_numpy(rf).permute(2, 0, 1).unsqueeze(0)
    try:
        parameter = next(model.model.parameters())
        tensor = tensor.to(device=parameter.device, dtype=parameter.dtype)
    except (AttributeError, StopIteration):
        tensor = tensor.float()
    indices = [_CAM_LAYER_CACHE[key]] if key in _CAM_LAYER_CACHE \
              else list(_candidate_layers(model))
    wrapped = _CAMModelWrapper(model.model); wrapped.eval()
    last_err = None
    for li in indices:
        try:
            layer = seq[li]
            cam = EigenCAM(model=wrapped, target_layers=[layer],
                           reshape_transform=_reshape)
            g = cam(input_tensor=tensor, targets=None)[0]
            _CAM_LAYER_CACHE[key] = li
            g_norm = (g - g.min()) / max(1e-8, g.max() - g.min())
            heat = cv2.applyColorMap((g_norm * 255).astype(np.uint8),
                                     cv2.COLORMAP_JET)
            blend = show_cam_on_image(rf, g, use_rgb=True)
            blend = cv2.cvtColor(blend, cv2.COLOR_RGB2BGR)
            heat = cv2.resize(heat,  (frame.shape[1], frame.shape[0]))
            blend = cv2.resize(blend, (frame.shape[1], frame.shape[0]))
            return heat, blend
        except Exception as e:
            last_err = e; continue
    log.warning("grad_cam: all layers failed (last: %s)", last_err)
    return None, None


class CAMCache:
    def __init__(self):
        self.items = []

    def add(self, frame, det, model, out_dir, video, idx,
            cam_result=_CAM_RESULT_UNSET):
        if len(self.items) >= CFG.CAM_MAX: return None
        if cam_result is _CAM_RESULT_UNSET:
            cam_result = grad_cam(model, frame, det["bbox"])
        heat, blend = cam_result
        if heat is None: return None
        out_dir.mkdir(parents=True, exist_ok=True)
        safe = sname(det["label"])
        stem = f"cam_{video}_{idx:04d}_{int(det['timestamp']*1000):08d}ms_{safe}"
        hp = out_dir / f"{stem}_heat.jpg"
        bp = out_dir / f"{stem}_blend.jpg"
        cv2.imwrite(str(hp), heat, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(str(bp), blend, [cv2.IMWRITE_JPEG_QUALITY, 90])
        rec = {"heat":str(hp),"blend":str(bp),"source":det["source"],
               "label":det["label"],"t":det["timestamp"],"video":video,
               "conf":det["confidence"]}
        self.items.append(rec)
        return rec

    def panel_html(self):
        if not self.items:
            return ('<div class="aee-empty">'
                    '<div class="aee-empty-icon">🔍</div>'
                    '<div class="aee-empty-title">No CAM heatmaps yet</div>'
                    '<div class="aee-empty-sub">'
                    'EigenCAM activation maps are generated <b>per weapon detection</b> '
                    'during analysis. They appear here as soon as the pipeline finds '
                    'a weapon and completes one forward pass.<br><br>'
                    'If the pipeline ran but produced no heatmaps, verify that the '
                    'uploaded clip actually contains a detectable weapon — heavy '
                    'blur, motion blur, or extreme occlusion will yield no detections.'
                    '</div></div>')
        srcs = sorted({i["source"] for i in self.items})
        chips = "".join(f'<span class="aee-class-chip">{s}</span>' for s in srcs)
        return f"""
        <div class="aee-cam-stats">
          <div class="aee-stat"><span class="aee-stat-icon">🔍</span>
            <div class="aee-stat-val">{len(self.items)}</div>
            <div class="aee-stat-lbl">CAM heatmaps</div></div>
          <div class="aee-stat"><span class="aee-stat-icon">🧠</span>
            <div class="aee-stat-val">{len(srcs)}</div>
            <div class="aee-stat-lbl">Source models</div></div>
          <div class="aee-stat"><span class="aee-stat-icon">🎯</span>
            <div class="aee-stat-val">{len({i['label'] for i in self.items})}</div>
            <div class="aee-stat-lbl">Weapon classes</div></div>
        </div>
        <div class="aee-cam-sources">
          <div class="aee-cam-sources-title">Contributing models</div>
          <div>{chips}</div>
        </div>"""


def save_capture(frame, det, ens, out_dir, idx, explain=True, tracker=None,
                 video_name=None, cam_result=_CAM_RESULT_UNSET):
    out_dir.mkdir(parents=True, exist_ok=True)
    vis = frame.copy()
    if explain and ens.models:
        if cam_result is _CAM_RESULT_UNSET:
            cam_result = grad_cam(ens.models[0]["yolo"], frame, det["bbox"])
        _, cam_blend = cam_result
        if cam_blend is not None:
            vis = cv2.addWeighted(cam_blend, 0.4, vis, 0.6, 0)
    x1, y1, x2, y2 = det["bbox"]
    lvl, risk, _ = threat_score(det, tracker, video_name)
    col_hex = THREAT_COLORS.get(lvl, "#64748b")
    col = tuple(int(col_hex.lstrip("#")[i:i+2], 16) for i in (4, 2, 0))
    cv2.rectangle(vis, (x1, y1), (x2, y2), col, 3)
    txt = f"{icon_for(det['label'])} {det['label']} {det['confidence']:.2f}"
    if "fusion_prob" in det: txt += f" fuse={det['fusion_prob']:.2f}"
    if det.get("track_id", -1) >= 0: txt += f" #{det['track_id']}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(vis, (x1, y1-th-10), (x1+tw+12, y1), col, -1)
    cv2.putText(vis, txt, (x1+6, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
    tb = f"{lvl.upper()} · RISK {risk}/100"
    (tw, th), _ = cv2.getTextSize(tb, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(vis, (8, 8), (tw+22, th+22), col, -1)
    cv2.putText(vis, tb, (16, th+15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
    banner = f"{ts(det['timestamp'])} | {det['label']} | {det['confidence']:.2f} | {det['source']}"
    (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.rectangle(vis, (8, 40), (tw+20, 40+th+14), (0,0,0), -1)
    cv2.putText(vis, banner, (16, 40+th+8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    fn = out_dir / f"cap_{idx:04d}_{int(det['timestamp']*1000):08d}ms_{det['label'].replace(' ','_')}.jpg"
    cv2.imwrite(str(fn), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return str(fn)


def analyze_audio(p, dur):
    if not p.exists(): return []
    try:
        frame_length, hop_length = 2048, 512
        block_frames = 60 * 16000
        carry = np.empty(0, dtype=np.float32)
        sample_cursor = 0
        rms_parts, time_parts = [], []
        with sf.SoundFile(str(p)) as audio:
            sr = int(audio.samplerate)
            block_frames = max(frame_length, 60 * sr)
            while True:
                block = audio.read(block_frames, dtype="float32", always_2d=True)
                if block.size == 0:
                    break
                mono = block.mean(axis=1, dtype=np.float32)
                data = np.concatenate((carry, mono)) if carry.size else mono
                if data.size >= frame_length:
                    starts = np.arange(0, data.size - frame_length + 1, hop_length)
                    squared = np.square(data, dtype=np.float64)
                    cumulative = np.concatenate(([0.0], np.cumsum(squared)))
                    energies = ((cumulative[starts + frame_length]
                                 - cumulative[starts]) / frame_length)
                    rms_parts.append(np.sqrt(np.maximum(energies, 0.0)))
                    frame_centers = (sample_cursor - carry.size + starts
                                     + frame_length / 2.0)
                    time_parts.append(frame_centers / sr)
                    consumed = int(starts[-1] + hop_length)
                    carry = data[consumed:].astype(np.float32, copy=False)
                else:
                    carry = data.astype(np.float32, copy=False)
                sample_cursor += mono.size
        if not rms_parts:
            return []
        rms = np.concatenate(rms_parts)
        t = np.concatenate(time_parts)
        base = float(np.median(rms)) + 1e-8
        thr = max(base*3., float(np.percentile(rms, 90)))
        ev = [{"timestamp":float(tt),"label":"Loud Sound",
               "confidence":min(.99, float(v/(thr+1e-8))/3.)}
              for v, tt in zip(rms, t) if tt <= dur and v >= thr]
        m = []
        for e in ev:
            if not m or e["timestamp"]-m[-1]["timestamp"] > 2: m.append(e)
        return m
    except Exception as e:
        log.warning("audio: %s", e); return []


def extract_clip(vp, s, e, out, cancel_event=None):
    dur = max(0.1, e-s); out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    for cmd in (
        ["ffmpeg","-y","-ss",f"{s:.3f}","-i",str(vp),"-t",f"{dur:.3f}",
         "-c:v","libx264","-preset","veryfast","-crf","23","-pix_fmt","yuv420p",
         "-c:a","aac","-b:a","128k","-movflags","+faststart",
         "-avoid_negative_ts","make_zero",str(out)],
        ["ffmpeg","-y","-ss",f"{s:.3f}","-i",str(vp),"-t",f"{dur:.3f}",
         "-c","copy","-avoid_negative_ts","make_zero",str(out)]):
        try:
            sh(cmd, cancel_event=cancel_event)
            if out.exists() and out.stat().st_size > 1024: return True
        except InterruptedError: return False
        except Exception: continue
    return False


def normalize_clip(src, dst):
    vf = (f"scale={CFG.MERGE_W}:{CFG.MERGE_H}:force_original_aspect_ratio=decrease,"
          f"pad={CFG.MERGE_W}:{CFG.MERGE_H}:(ow-iw)/2:(oh-ih)/2:color=black,"
          f"setsar=1,fps={CFG.MERGE_FPS}")
    if has_audio(src):
        sh(["ffmpeg","-y","-i",str(src),"-vf",vf,"-c:v","libx264","-preset","veryfast",
            "-crf","23","-pix_fmt","yuv420p","-c:a","aac","-b:a","128k","-ar","44100",
            "-ac","2","-movflags","+faststart","-shortest",str(dst)])
    else:
        sh(["ffmpeg","-y","-i",str(src),"-f","lavfi","-i",
            "anullsrc=channel_layout=stereo:sample_rate=44100","-vf",vf,
            "-map","0:v:0","-map","1:a:0","-c:v","libx264","-preset","veryfast",
            "-crf","23","-pix_fmt","yuv420p","-c:a","aac","-b:a","128k","-ar","44100",
            "-ac","2","-movflags","+faststart","-shortest",str(dst)])


def merge_clips(clips, out, work):
    valid = [c for c in clips if Path(c).exists() and Path(c).stat().st_size > 0]
    if not valid: return None
    if len(valid) == 1: shutil.copy(valid[0], out); return str(out)
    work.mkdir(parents=True, exist_ok=True); nrm = []
    for i, c in enumerate(valid):
        n = work / f"n_{i:04d}.mp4"
        try:
            normalize_clip(Path(c), n)
            if n.exists() and n.stat().st_size > 0: nrm.append(n)
        except Exception: pass
    if not nrm: return None
    lst = work / "list.txt"
    lst.write_text("\n".join(f"file '{n.resolve()}'" for n in nrm))
    try:
        sh(["ffmpeg","-y","-f","concat","-safe","0","-i",str(lst),"-c","copy",
            "-movflags","+faststart",str(out)])
        if out.exists() and out.stat().st_size > 0: return str(out)
    except Exception: pass
    return None


@dataclass
class Meta:
    fps: float; frame_count: int; width: int; height: int; duration: float

@dataclass
class Result:
    video_name: str; source_path: str; metadata: Meta
    detections: list = field(default_factory=list)
    captures: list = field(default_factory=list)
    object_summaries: dict = field(default_factory=dict)
    summary_stats: dict = field(default_factory=dict)
    notes: str = ""


class Processor:
    def __init__(self):
        self.ens = Ensemble.get()
        self.tracker = Tracker()
        self.camcache = CAMCache()

    @staticmethod
    def _prefetch_frames(capture, cancel_event=None, queue_size=4):
        frame_queue = queue.Queue(maxsize=queue_size)
        stop_event = threading.Event()
        end = object()

        def enqueue(item):
            while not stop_event.is_set():
                try:
                    frame_queue.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    continue
            return False

        def decode():
            index = 0
            consecutive_failures = 0
            try:
                while not stop_event.is_set():
                    if cancel_event is not None and cancel_event.is_set():
                        break
                    ok, frame = capture.read()
                    reported_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
                    if not enqueue((index, ok, frame, reported_ms)):
                        break
                    index += 1
                    consecutive_failures = (
                        0 if ok and frame is not None else consecutive_failures + 1
                    )
                    if consecutive_failures > 5:
                        break
            except Exception as exc:
                enqueue((None, exc))
            finally:
                capture.release()
                enqueue(end)

        decoder = threading.Thread(
            target=decode, name="aee-video-decoder", daemon=True)
        decoder.start()
        try:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    return
                item = frame_queue.get()
                if item is end:
                    return
                if item[0] is None:
                    raise RuntimeError(f"Video decoding failed: {item[1]}") from item[1]
                yield item
        finally:
            stop_event.set()
            decoder.join()

    def process_stream(self, vp, root, cancel_event=None, high_accuracy=False):
        video = Path(vp)
        if video.suffix.lower() not in CFG.SUPPORTED:
            yield ("error", "Unsupported format"); return
        name = video.name
        work_name = sname(video)
        out = root / work_name
        out.mkdir(parents=True, exist_ok=True)

        yield ("stage", "meta", 0.0, "Reading metadata")
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            yield ("error", "Cannot open video"); return
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        cap.release()
        if not math.isfinite(fps) or fps <= 0:
            fps = 25.0
        probed_duration = probe_dur(video)
        frame_duration = fc / fps if fc > 0 else 0.0
        duration = max(probed_duration, frame_duration, 0.0)
        if fc <= 0 and duration > 0:
            fc = int(math.ceil(duration * fps))
        meta = Meta(fps, fc, W, H, duration)
        yield ("stage", "meta", 1.0, f"{meta.duration:.1f}s @ {fps:.1f}fps")

        self.ens.reset()
        yield ("stage", "audio", 0.0, "Extracting audio")
        aud = out / "audio.wav"; aev = []
        try:
            sh(["ffmpeg","-y","-i",str(video),"-vn","-ac","1","-ar","16000",
                "-f","wav",str(aud)], cancel_event=cancel_event)
            if cancel_event is not None and cancel_event.is_set():
                yield ("cancelled", None); return
            yield ("stage", "audio", 0.5, "Analyzing loudness")
            aev = analyze_audio(aud, meta.duration)
        except InterruptedError:
            yield ("cancelled", None); return
        except Exception:
            if cancel_event is not None and cancel_event.is_set():
                yield ("cancelled", None); return
        yield ("stage", "audio", 1.0, f"{len(aev)} audio events")

        yield ("stage", "detect", 0.0, "Running ensemble tracking")
        caps_dir = out / "captures"; caps_dir.mkdir(exist_ok=True)
        cam_dir = out / "cam"; cam_dir.mkdir(exist_ok=True)
        dets, caps = [], []

        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            cap.release()
            yield ("error", "Cannot reopen video for frame analysis")
            return
        sample_period = 1.0 / max(CFG.FPS, 0.01)
        total = max(1, int(math.ceil(meta.duration / sample_period)) + 1)
        idx = proc = 0
        t0 = time.time()
        fails = 0
        cam_count = 0
        next_sample_t = 0.0
        last_frame_t = -1.0
        try:
            for idx, ok, fr, reported_ms in self._prefetch_frames(
                    cap, cancel_event=cancel_event):
                if cancel_event is not None and cancel_event.is_set():
                    yield ("cancelled", None); return
                if not ok or fr is None:
                    fails += 1
                    if fails > 5:
                        if (meta.duration > 0 and last_frame_t >= 0
                                and meta.duration - last_frame_t
                                > max(10.0, sample_period * 3)):
                            yield ("error",
                                   f"Video decoding stopped at {ts(last_frame_t)} "
                                   f"of {ts(meta.duration)}; the remaining section "
                                   "could not be analyzed.")
                            return
                        break
                    continue
                fails = 0
                reported_t = reported_ms / 1000.0
                if (math.isfinite(reported_t) and reported_t >= 0
                        and (idx == 0 or reported_t > last_frame_t)):
                    frame_t = reported_t
                else:
                    frame_t = idx / meta.fps
                t = max(last_frame_t, frame_t, 0.0)
                last_frame_t = t
                if t + 1e-6 >= next_sample_t:
                    raw_dets = self.ens.detect(fr, high_accuracy=high_accuracy)
                    frame_dets = merge_and_filter(raw_dets)
                    self.tracker.update(frame_dets, t, idx, name)
                    for d in frame_dets:
                        d["timestamp"] = round(t, 3); d["kind"] = "object"
                        dets.append(d)
                        if d["label"] in WEAPONS:
                            yield ("detection", d)
                            need_capture = len(caps) < CFG.CAP_MAX
                            need_cam = cam_count < CFG.CAM_MAX and bool(self.ens.models)
                            cam_result = _CAM_RESULT_UNSET
                            if self.ens.models and (need_capture or need_cam):
                                try:
                                    cam_result = grad_cam(
                                        self.ens.models[0]["yolo"], fr, d["bbox"])
                                except Exception as ce:
                                    log.warning("CAM inference failed: %s", ce)
                                    cam_result = (None, None)
                            if need_capture:
                                cp = save_capture(fr, d, self.ens, caps_dir, len(caps),
                                                  tracker=self.tracker, video_name=name,
                                                  cam_result=cam_result)
                                if cp: caps.append(cp)
                            if need_cam:
                                try:
                                    rec = self.camcache.add(fr, d,
                                        self.ens.models[0]["yolo"],
                                        cam_dir, name, cam_count,
                                        cam_result=cam_result)
                                    if rec:
                                        cam_count += 1
                                        yield ("cam", rec)
                                except Exception as ce:
                                    log.warning("CAM add failed: %s", ce)
                    proc += 1
                    while next_sample_t <= t:
                        next_sample_t += sample_period
                if proc and proc % 3 == 0:
                    yield ("substep", {"frame":idx,"total_frames":meta.frame_count,
                                       "sampled":proc,"total_sampled":total,
                                       "detections":len(dets),
                                       "fps_processed":proc/max(0.001,time.time()-t0)})
                    yield ("stage", "detect", min(proc/total, 0.995),
                           f"sampled {proc:,}/{total:,} · {len(dets):,} detections")
        finally:
            cap.release()
        yield ("stage", "detect", 1.0,
               f"Finished full video · {proc:,} sampled frames · {len(dets):,} detections")

        for e in aev:
            if e["confidence"] >= .5:
                dets.append({"timestamp":e["timestamp"],"label":e["label"],
                    "confidence":e["confidence"],"kind":"audio",
                    "source":"audio","bbox":(),"fusion_prob":1.0,"track_id":-1})

        yield ("stage", "highlight", 0.0, "Scoring events")
        ev = self._score(dets)
        hl = out / "highlights"; hl.mkdir(exist_ok=True); clips = []
        for i, (s, e) in enumerate(merge_ranges(ev, meta.duration, CFG.HL_PAD), 1):
            if cancel_event is not None and cancel_event.is_set():
                yield ("cancelled", None); return
            t = hl / f"highlight_{i:03d}.mp4"
            if extract_clip(video, s, e, t, cancel_event=cancel_event):
                clips.append(str(t))
        yield ("stage", "highlight", 1.0, f"{len(clips)} highlights")

        yield ("stage", "summary", 0.0, "Building object reels")
        summ, stats = self._summaries(video, dets, out, meta.duration)
        yield ("stage", "summary", 1.0, f"{len(summ)} object summaries")

        uniq_w = sorted({d["label"] for d in dets if d.get("kind") == "object"
                         and d["label"] in WEAPONS})
        n_person = sum(1 for d in dets if d.get("label") == "Person")
        n_face = sum(1 for d in dets if d.get("label") == "Face")
        parts = [f"{len(clips)} highlights" if clips else "No highlights"]
        if summ: parts.append(f"Summaries: {', '.join(summ)}")
        elif uniq_w: parts.append(f"Detected: {', '.join(uniq_w)}")
        else: parts.append("No weapons detected")
        if n_person or n_face:
            parts.append(f"Context: {n_person} person · {n_face} face")
        r = Result(name, str(video), meta, dets, caps, summ, stats, " • ".join(parts))
        yield ("result", r)

    def _score(self, dets):
        by = defaultdict(lambda: {"w":0,"a":0,"p":0,"f":0})
        for d in dets:
            s = int(d["timestamp"])
            if d.get("kind") == "audio": by[s]["a"] += 1
            elif d["label"] in WEAPONS: by[s]["w"] += 1
            elif d["label"] == "Person": by[s]["p"] += 1
            elif d["label"] == "Face": by[s]["f"] += 1
        ev = []
        for s, b in by.items():
            sc = 0
            if b["w"]: sc += CFG.W_WPN
            if b["w"] >= 2: sc += CFG.W_MULTI
            if b["w"] and b["p"]: sc += CFG.W_WPN_PERSON
            if b["w"] and b["f"]: sc += CFG.W_WPN_FACE
            if b["p"] and not b["w"]: sc += CFG.W_PERSON_ONLY
            if b["a"]: sc += CFG.W_AUD
            if sc >= CFG.HL_THRESH:
                ev.append({"timestamp": float(s), "score": sc})
        return ev

    def _summaries(self, video, dets, out, dur):
        by = defaultdict(list)
        for d in dets:
            if d.get("kind") == "object" and d["label"] in WEAPONS:
                by[d["label"]].append(d)
        if not by: return {}, {}
        base = out / "object_summaries"; base.mkdir(parents=True, exist_ok=True)
        sm, st = {}, {}
        for lbl, ds in by.items():
            safe = sname(lbl)
            rng = merge_ranges([{"timestamp":d["timestamp"]} for d in ds],
                                dur, CFG.CLIP_PAD)[:25]
            cl = []
            for i, (s, e) in enumerate(rng):
                if e - s < .5: continue
                c = base / f"{safe}_{i:03d}.mp4"
                if extract_clip(video, s, e, c): cl.append(str(c))
            if not cl: continue
            m = merge_clips(cl, base / f"summary_{safe}.mp4", base / f"_w_{safe}")
            if m:
                sm[lbl] = m
                cf = [d["confidence"] for d in ds]
                st[lbl] = {"detections":len(ds),
                           "avg_confidence":round(sum(cf)/len(cf),4),
                           "max_confidence":round(max(cf),4),
                           "clips":len(cl),"duration":round(probe_dur(m),2)}
        return sm, st


def build_pdf(results, model_info, out_path, analyst_notes=None, tracker=None,
              case_meta=None, case_notes=""):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.lib import colors
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Table, TableStyle, PageBreak)
    except Exception:
        return None
    try:
        styles = getSampleStyleSheet()
        h1 = ParagraphStyle("h1", parent=styles["Title"], fontSize=22,
                            textColor=colors.HexColor("#0f172a"))
        h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13,
                            textColor=colors.HexColor("#1e40af"))
        h3 = ParagraphStyle("h3", parent=styles["Heading3"], fontSize=11,
                            textColor=colors.HexColor("#7c3aed"))
        body = ParagraphStyle("b", parent=styles["BodyText"], fontSize=10)
        muted = ParagraphStyle("m", parent=styles["BodyText"], fontSize=9,
                               textColor=colors.HexColor("#64748b"))
        doc = SimpleDocTemplate(str(out_path), pagesize=A4, leftMargin=18*mm,
                                rightMargin=18*mm, topMargin=20*mm, bottomMargin=18*mm)
        st = [Paragraph("AI Evidence Extractor — Forensic Report", h1),
              Paragraph(f"Generated {datetime.now():%Y-%m-%d %H:%M:%S}", muted),
              Spacer(1, 14)]

        cm = case_meta or {}
        if any(cm.get(k) for k in ("case_number","examiner","agency","evidence_id")):
            case_rows = [["Field","Value"]]
            for lbl, key in [("Case Number","case_number"),("Evidence ID","evidence_id"),
                             ("Examiner","examiner"),("Agency","agency")]:
                v = (cm.get(key) or "").strip()
                if v: case_rows.append([lbl, v])
            if len(case_rows) > 1:
                ct = Table(case_rows, colWidths=[60*mm, 110*mm])
                ct.setStyle(TableStyle([
                    ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#0f172a")),
                    ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                    ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
                    ("FONTSIZE",(0,0),(-1,-1),9),
                    ("GRID",(0,0),(-1,-1),.3,colors.HexColor("#cbd5e1"))]))
                st.append(ct); st.append(Spacer(1, 14))

        summary = [["Metric","Value"],
                   ["Videos analysed",str(len(results))],
                   ["Total detections",str(sum(len(r.detections) for r in results))],
                   ["Ensemble models",str(model_info.get("n_models", 0))],
                   ["Models",", ".join(model_info.get("names", []))]]
        t = Table(summary, colWidths=[60*mm, 110*mm])
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1e40af")),
            ("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("FONTSIZE",(0,0),(-1,-1),9),
            ("GRID",(0,0),(-1,-1),.3,colors.HexColor("#cbd5e1"))]))
        st.append(t)

        if case_notes and case_notes.strip():
            st.append(Spacer(1, 12))
            st.append(Paragraph("Analyst Case Notes", h3))
            st.append(Paragraph(case_notes.strip().replace("\n","<br/>"), body))

        for r in results:
            st.append(PageBreak())
            st.append(Paragraph(f"Video: {r.video_name}", h2))
            st.append(Paragraph(f"Duration {ts(r.metadata.duration)} · "
                f"{r.metadata.width}×{r.metadata.height} · {r.metadata.fps:.2f} fps", body))
            st.append(Spacer(1, 8)); st.append(Paragraph(r.notes, body))

            person_count = sum(1 for d in r.detections if d.get("label") == "Person")
            face_count = sum(1 for d in r.detections if d.get("label") == "Face")
            weapon_count = sum(1 for d in r.detections if d.get("label") in WEAPONS)
            _by_sec = defaultdict(set)
            for d in r.detections:
                if d.get("label") in WEAPONS: _by_sec[int(d["timestamp"])].add("w")
                elif d.get("label") in CONTEXT: _by_sec[int(d["timestamp"])].add("p")
            correlated = sum(1 for v in _by_sec.values() if "w" in v and "p" in v)

            st.append(Spacer(1, 10))
            st.append(Paragraph("Scene Context", h2))
            ctx_rows = [["Metric","Value"],
                        ["Person detections",str(person_count)],
                        ["Face detections",str(face_count)],
                        ["Weapon detections",str(weapon_count)],
                        ["Correlated seconds (weapon + person)",str(correlated)]]
            ctx = Table(ctx_rows, colWidths=[90*mm, 80*mm])
            ctx.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#7c3aed")),
                ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
                ("FONTSIZE",(0,0),(-1,-1),9),
                ("GRID",(0,0),(-1,-1),.3,colors.HexColor("#cbd5e1"))]))
            st.append(ctx)

            if tracker is not None:
                trows = [r_ for r_ in tracker.summaries() if r_["Video"] == r.video_name]
                if trows:
                    st.append(Spacer(1, 10))
                    st.append(Paragraph("Object track analytics", h3))
                    tr_tbl = [["Track","Class","First","Last","Dur(s)","Frames","Risk","Threat"]]
                    for tr in trows[:40]:
                        tr_tbl.append([f"#{tr['Track']}", tr["Class"], tr["First"],
                                       tr["Last"], f"{tr['Duration_s']:.2f}",
                                       str(tr["Frames"]), str(tr["Risk"]),
                                       tr.get("Threat","—")])
                    tt = Table(tr_tbl, colWidths=[16*mm, 22*mm, 22*mm, 22*mm,
                                                    16*mm, 16*mm, 14*mm, 20*mm])
                    tt.setStyle(TableStyle([
                        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#0f766e")),
                        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
                        ("FONTSIZE",(0,0),(-1,-1),7),
                        ("GRID",(0,0),(-1,-1),.25,colors.HexColor("#cbd5e1"))]))
                    st.append(tt)

            if r.detections:
                st.append(Spacer(1, 10)); st.append(Paragraph("Evidence Log", h2))
                rows = [["Time","Type","Object","Conf","Fusion","Source","Track","Threat","Note"]]
                for d in r.detections[:300]:
                    key = f"{d['timestamp']}|{d.get('label','')}|{d.get('track_id',-1)}"
                    note = (analyst_notes or {}).get(key, "")[:60]
                    lvl, sc, _ = threat_score(d, tracker, r.video_name)
                    lvl_disp = f"{lvl.title()} {sc}" if sc else "—"
                    rows.append([ts(d["timestamp"]),
                        "audio" if d.get("kind") == "audio" else "object",
                        d["label"], f"{d['confidence']:.2f}",
                        f"{d.get('fusion_prob', 1.0):.2f}", d.get("source", ""),
                        str(d.get("track_id", "-")), lvl_disp, note or "—"])
                tb = Table(rows, colWidths=[18*mm,12*mm,20*mm,10*mm,12*mm,20*mm,10*mm,18*mm,34*mm],
                           repeatRows=1)
                tb.setStyle(TableStyle([
                    ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#334155")),
                    ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                    ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
                    ("FONTSIZE",(0,0),(-1,-1),7),
                    ("GRID",(0,0),(-1,-1),.25,colors.HexColor("#cbd5e1"))]))
                st.append(tb)
        doc.build(st)
        return str(out_path) if out_path.exists() else None
    except Exception as e:
        log.warning("pdf: %s", e); return None


# ============================================================
# CSS (unchanged)
# ============================================================
CSS = """
:root{
  --bg:#04060c;--surface:#0a0f1c;--surface-2:#0e1424;--surface-3:#141a2e;
  --border:rgba(148,163,184,.08);--border-strong:rgba(148,163,184,.16);
  --text:#e6edf7;--text-muted:#8b9bb4;--text-subtle:#5a6b85;
  --blue:#3b82f6;--violet:#8b5cf6;--cyan:#22d3ee;
  --ok:#10b981;--warn:#f59e0b;--err:#ef4444;
  --shell-bg-1:#0a1020;--shell-bg-2:#04060c;
  --hero-grad-1:#f8fafc;--hero-grad-2:#c7d2fe;--hero-grad-3:#c4b5fd;--hero-grad-4:#67e8f9;
}
html[data-theme="light"]{
  --bg:#FFF0CC;--surface:#FFF8E9;--surface-2:#F8E7C8;--surface-3:#EFD9B7;
  --border:rgba(121,98,83,.2);--border-strong:rgba(121,98,83,.42);
  --text:#554237;--text-muted:#796253;--text-subtle:#796253;
  --blue:#F28C32;--violet:#B58C99;--cyan:#A58D91;
  --card-top:#796253;--card-glass:#A58D91;--card-transition:#DDA16C;
  --card-glow:#F28C32;--card-lavender:#B58C99;
  --card-text:#FFF0CC;--card-text-muted:rgba(255,240,204,.8);
  --shell-bg-1:#FFF8E9;--shell-bg-2:#FFF0CC;
  --hero-grad-1:#796253;--hero-grad-2:#A58D91;--hero-grad-3:#B58C99;--hero-grad-4:#DDA16C;
  --body-background-fill:#FFF0CC;
  --body-text-color:#796253;
  --body-text-color-subdued:#796253;
  --background-fill-primary:#FFF8E9;
  --background-fill-secondary:#F8E7C8;
  --background-fill-tertiary:#EFD9B7;
  --block-background-fill:#FFF8E9;
  --block-border-color:rgba(121,98,83,.2);
  --border-color-primary:rgba(121,98,83,.25);
  --border-color-accent:#DDA16C;
  --color-accent:#F28C32;
  --color-accent-soft:#F8E7C8;
  --input-background-fill:#FFF8E9;
  --input-background-fill-focus:#FFFDF7;
  --input-border-color:rgba(121,98,83,.35);
  --input-border-color-focus:#DDA16C;
  --input-placeholder-color:#796253;
  --button-primary-background-fill:#F28C32;
  --button-primary-background-fill-hover:#DDA16C;
  --button-primary-text-color:#49382D;
  --button-secondary-background-fill:#F8E7C8;
  --button-secondary-background-fill-hover:#EFD9B7;
  --button-secondary-text-color:#796253;
}
html,body,.gradio-container{
  background:var(--bg)!important;color:var(--text)!important;
  font-family:'Inter',system-ui,-apple-system,sans-serif!important;
  -webkit-font-smoothing:antialiased;
  -moz-osx-font-smoothing:grayscale;
}
.gradio-container{max-width:1760px!important;width:100%!important;margin:0 auto!important;padding:0 28px 80px!important;}
.gradio-container::before{
  content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
  background:
    radial-gradient(1100px 700px at 12% -12%,rgba(59,130,246,.14),transparent 55%),
    radial-gradient(900px 600px at 88% -4%,rgba(139,92,246,.12),transparent 55%),
    radial-gradient(1200px 800px at 50% 112%,rgba(34,211,238,.08),transparent 60%);
}
.gradio-container>*{position:relative;z-index:1;}
.gradio-container button{font-family:inherit;transition:filter .16s ease,border-color .16s ease,
  box-shadow .16s ease,transform .16s ease;}
.gradio-container button:hover:not(:disabled){filter:brightness(1.06);}
.gradio-container button:active:not(:disabled){transform:translateY(1px);}
.gradio-container button:focus-visible,.gradio-container input:focus-visible,
.gradio-container select:focus-visible,.gradio-container textarea:focus-visible{
  outline:2px solid rgba(255,110,104,.88)!important;outline-offset:2px!important;}
.gradio-container .wrap.generating{
  border:1px solid rgba(96,165,250,.34)!important;
  border-radius:14px!important;
  background:rgba(7,12,24,.78)!important;
  backdrop-filter:blur(12px) saturate(1.2);
  box-shadow:0 18px 50px -24px rgba(37,99,235,.52),inset 0 1px rgba(255,255,255,.08);
  animation:aee-gradio-loading-in .18s ease-out both!important;
}
.gradio-container .wrap.generating::before{
  content:'';width:34px;height:34px;flex:none;margin-bottom:12px;
  border:3px solid rgba(148,163,184,.2);border-top-color:#60a5fa;
  border-right-color:#a78bfa;border-radius:50%;
  box-shadow:0 0 24px rgba(59,130,246,.22);
  animation:aee-gradio-loading-spin .8s linear infinite;
}
.gradio-container .wrap.generating .loading{
  color:var(--text)!important;font-family:inherit!important;
  font-size:.9rem!important;font-weight:650!important;letter-spacing:.01em;
}
.gradio-container .wrap.generating .progress-level-inner{
  color:var(--text-muted)!important;font-family:inherit!important;font-size:.78rem!important;
}
.gradio-container .wrap.generating .progress-bar-wrap{
  height:7px!important;border:1px solid rgba(148,163,184,.2)!important;
  border-radius:999px!important;background:rgba(148,163,184,.12)!important;
}
.gradio-container .wrap.generating .progress-bar{
  border-radius:999px!important;
  background:linear-gradient(90deg,#3b82f6,#8b5cf6,#22d3ee)!important;
}
html[data-theme="light"] .gradio-container .wrap.generating{
  background:rgba(255,255,255,.88)!important;
  border-color:rgba(59,130,246,.24)!important;
  box-shadow:0 18px 50px -24px rgba(37,99,235,.3),inset 0 1px #fff;
}
@keyframes aee-gradio-loading-spin{to{transform:rotate(360deg);}}
@keyframes aee-gradio-loading-in{from{opacity:0;transform:scale(.985);}
  to{opacity:1;transform:scale(1);}}
@media(prefers-reduced-motion:reduce){
  .gradio-container .wrap.generating{animation:none!important;}
  .gradio-container .wrap.generating::before{animation:none;}
}
.aee-workflow{backdrop-filter:blur(14px);box-shadow:0 14px 42px -34px rgba(0,0,0,.9);}
.aee-settings-card,.aee-upload-shell,.aee-config-shell .aee-card{
  transition:border-color .18s ease,box-shadow .18s ease,background .18s ease;}
.aee-settings-card:hover,.aee-upload-shell:hover,.aee-config-shell .aee-card:hover{
  border-color:rgba(148,163,184,.22);}

.aee-hero{position:relative;width:100%;max-width:1600px;margin:0 auto;text-align:left;
  padding:38px 8px 24px;display:flex;flex-direction:column;align-items:flex-start;}
.aee-brand{display:flex;align-items:center;gap:15px;margin-bottom:18px;}
.aee-brand-mark{width:54px;height:54px;display:grid;place-items:center;border-radius:17px;
  background:linear-gradient(145deg,rgba(59,130,246,.2),rgba(139,92,246,.18));
  border:1px solid rgba(129,140,248,.34);box-shadow:0 12px 35px -18px rgba(59,130,246,.8);}
.aee-brand-mark svg{width:31px;height:31px;}
.aee-brand-copy{display:flex;flex-direction:column;gap:3px;}
.aee-brand-name{font-size:clamp(1.45rem,3vw,2rem);font-weight:800;letter-spacing:-.045em;color:var(--text);}
.aee-brand-caption{font-size:.78rem;color:var(--text-subtle);letter-spacing:.03em;}
.aee-hero .aee-hero-badge{margin:0 0 12px;padding:6px 12px;font-size:.67rem;}
.aee-hero .aee-hero-description{color:var(--text-muted);font-size:.94rem;max-width:760px;
  margin:0 0 16px;line-height:1.65;}
.aee-workflow{width:100%;max-width:1600px;margin:0 auto 22px;padding:13px 18px;display:flex;align-items:center;
  gap:10px;border:1px solid var(--border);border-radius:15px;
  background:linear-gradient(120deg,rgba(15,23,42,.68),rgba(17,24,39,.48));}
.aee-workflow-step{display:flex;align-items:center;gap:9px;color:var(--text-muted);font-size:.78rem;font-weight:650;white-space:nowrap;}
.aee-workflow-number{display:grid;place-items:center;width:25px;height:25px;border-radius:9px;
  color:#bfdbfe;background:rgba(59,130,246,.16);border:1px solid rgba(96,165,250,.24);font-size:.68rem;}
.aee-workflow-line{height:1px;flex:1;min-width:12px;background:linear-gradient(90deg,var(--border-strong),transparent);}
.aee-workspace{width:100%!important;max-width:1600px;margin:0 auto;align-items:stretch!important;gap:20px!important;}
.aee-workspace>.column{flex:1 1 0!important;width:0!important;min-width:0!important;}
.aee-config-shell{min-width:0!important;display:flex!important;}
.aee-config-shell>div{width:100%;}
@media(max-width:720px){
  .aee-hero{padding:30px 4px 20px;}
  .aee-brand-mark{width:46px;height:46px;border-radius:14px;}
  .aee-workflow{gap:6px;padding:10px 12px;}
  .aee-workflow-step{gap:5px;font-size:.67rem;}
  .aee-workflow-number{width:21px;height:21px;border-radius:7px;}
  .aee-workflow-line{min-width:5px;}
  .gradio-container{padding:0 14px 48px!important;}
  .aee-workspace{gap:14px!important;}
}
@media(max-width:900px){
  .aee-workspace{flex-direction:column!important;}
  .aee-workspace>.column{flex:none!important;width:100%!important;min-width:0!important;}
}
html[data-theme="light"] .aee-workflow{background:linear-gradient(120deg,#fff,#f1f5f9);}

.aee-upload-shell{min-width:0!important;}
.aee-config-shell .aee-card{height:100%;box-sizing:border-box;}
.aee-tabs [role="tabpanel"]{padding:6px 2px 2px;}
.aee-hero-badge{
  display:inline-flex;align-items:center;gap:8px;padding:7px 16px;
  border-radius:100px;
  background:linear-gradient(135deg,rgba(59,130,246,.10),rgba(139,92,246,.08));
  border:1px solid rgba(96,165,250,.28);
  color:#bfdbfe;font-size:.72rem;font-weight:600;letter-spacing:.14em;
  text-transform:uppercase;margin-bottom:22px;
  backdrop-filter:blur(10px);
}
html[data-theme="light"] .aee-hero-badge{color:#1e40af;}
.aee-pulse{
  width:7px;height:7px;border-radius:50%;background:var(--ok);
  box-shadow:0 0 12px var(--ok),0 0 0 0 var(--ok);
  animation:pulse 2s ease-in-out infinite;
}
@keyframes pulse{
  0%{box-shadow:0 0 12px var(--ok),0 0 0 0 rgba(16,185,129,.6);}
  50%{opacity:.65;}
  100%{box-shadow:0 0 12px var(--ok),0 0 0 10px rgba(16,185,129,0);}
}
.aee-hero h1{
  font-size:clamp(2.2rem,5vw,3.6rem);font-weight:900;letter-spacing:-.05em;
  line-height:1.02;margin-bottom:16px;
  background:linear-gradient(115deg,var(--hero-grad-1) 0%,var(--hero-grad-2) 40%,
    var(--hero-grad-3) 65%,var(--hero-grad-4) 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;
}
.aee-hero p{color:var(--text-muted);font-size:1rem;max-width:760px;
  margin:0 auto 24px;line-height:1.7;}
.aee-hero-pills{display:flex;gap:8px;justify-content:center;flex-wrap:wrap;}
.aee-pill{
  padding:6px 14px;border-radius:100px;
  background:linear-gradient(180deg,rgba(15,23,42,.55),rgba(9,12,22,.65));
  border:1px solid var(--border);font-size:.74rem;color:var(--text-muted);
  font-family:'JetBrains Mono',monospace;
  backdrop-filter:blur(8px);
  transition:all .2s ease;
}
.aee-pill:hover{border-color:var(--border-strong);color:var(--text);}
html[data-theme="light"] .aee-pill{
  background:linear-gradient(180deg,#ffffff,#f8fafc);
}
.aee-toolbar{position:absolute;top:24px;right:28px;display:flex;gap:8px;z-index:10;}
.aee-tool{
  width:42px;height:42px;border-radius:12px;
  background:linear-gradient(180deg,rgba(15,23,42,.55),rgba(9,12,22,.65));
  border:1px solid var(--border);display:flex;align-items:center;justify-content:center;
  cursor:pointer;font-size:1.05rem;color:var(--text-muted);text-decoration:none;
  transition:all .22s cubic-bezier(.4,0,.2,1);
  backdrop-filter:blur(10px);
}
.aee-tool:hover{
  background:linear-gradient(180deg,rgba(30,41,59,.75),rgba(15,23,42,.85));
  color:var(--text);border-color:rgba(96,165,250,.45);
  transform:translateY(-2px);
  box-shadow:0 12px 26px -12px rgba(59,130,246,.55);
}
html[data-theme="light"] .aee-tool{
  background:linear-gradient(180deg,#ffffff,#f8fafc);
}
.aee-profile-tool{padding:3px!important;border-radius:50%!important;overflow:hidden;
  border-color:rgba(148,163,184,.28)!important;background:rgba(15,23,42,.7)!important;}
.aee-profile-tool>[data-aee-profile-avatar]{width:100%;height:100%;display:grid;place-items:center;
  overflow:hidden;border-radius:50%;color:#dbeafe;
  background:linear-gradient(145deg,rgba(59,130,246,.42),rgba(139,92,246,.4));
  font-size:.76rem;font-weight:800;letter-spacing:.02em;}
.aee-profile-tool img{display:block;width:100%;height:100%;border-radius:50%;object-fit:cover;}
.aee-profile-tool svg{width:19px;height:19px;color:#cbd5e1;}
.aee-update-app{width:auto;min-height:42px;
  display:inline-flex;align-items:center;justify-content:center;gap:8px;padding:0 14px;
  border:1px solid rgba(96,165,250,.28);border-radius:13px;color:#dbeafe;
  background:linear-gradient(145deg,rgba(30,64,120,.92),rgba(17,32,62,.94));
  box-shadow:0 10px 28px -18px rgba(37,99,235,.8);backdrop-filter:blur(14px);
  cursor:pointer;font:650 .76rem Inter,system-ui,sans-serif;
  transition:transform .18s,border-color .18s,box-shadow .18s,background .18s;}
.aee-update-app[hidden]{display:none!important;}
.aee-update-app:hover{transform:translateY(-1px);border-color:rgba(96,165,250,.62);
  box-shadow:0 14px 30px -16px rgba(59,130,246,.62);
  background:linear-gradient(145deg,rgba(37,99,235,.82),rgba(30,64,120,.96));}
.aee-update-app svg{width:16px;height:16px;flex:none;}
html[data-theme="light"] .aee-profile-tool{background:rgba(255,255,255,.86)!important;}
html[data-theme="light"] .aee-profile-tool>[data-aee-profile-avatar]{color:#1e3a8a;}
html[data-theme="light"] .aee-update-app{color:#1d4ed8;
  background:linear-gradient(145deg,rgba(255,255,255,.96),rgba(239,246,255,.96));
  border-color:rgba(59,130,246,.22);}
html[data-theme="light"] .aee-update-app:hover{background:#eff6ff;border-color:rgba(59,130,246,.5);}
@media(max-width:640px){
  .aee-hero{padding:88px 4px 20px;}
  .aee-workflow{flex-wrap:wrap;justify-content:space-between;}
  .aee-workflow-step{flex:1 1 calc(50% - 12px);white-space:normal;}
  .aee-workflow-line{display:none;}
  .aee-toolbar{position:fixed;top:12px;right:12px;gap:6px;}
  .aee-toolbar .aee-tool{width:36px;height:36px;border-radius:11px;}
  .aee-toolbar .aee-update-app{width:auto!important;min-height:36px;padding:0 8px;gap:6px;font-size:.69rem;}
  .aee-toolbar .aee-profile-tool{border-radius:50%!important;}
}

.aee-card{
  background:linear-gradient(180deg,rgba(15,23,42,.65),rgba(10,14,26,.78))!important;
  border:1px solid var(--border)!important;border-radius:18px!important;
  padding:24px!important;
  backdrop-filter:blur(12px);
}
.aee-card-title{font-weight:700;font-size:1rem;color:var(--text);letter-spacing:-.01em;}
.aee-card-sub{color:var(--text-subtle);font-size:.8rem;margin-top:4px;line-height:1.55;}
.aee-update-card{position:relative;max-width:900px;margin:20px auto;padding:clamp(22px,4vw,36px);
  overflow:hidden;border:1px solid rgba(96,165,250,.24);border-radius:24px;
  background:radial-gradient(600px 240px at 100% 0%,rgba(59,130,246,.16),transparent 68%),
    linear-gradient(150deg,rgba(15,23,42,.9),rgba(8,12,23,.96));
  box-shadow:0 24px 80px -42px rgba(37,99,235,.45);}
.aee-update-eyebrow{display:flex;align-items:center;gap:9px;color:#93c5fd;font-size:.72rem;
  font-weight:750;letter-spacing:.15em;text-transform:uppercase;}
.aee-update-dot{width:8px;height:8px;border-radius:50%;background:#38bdf8;box-shadow:0 0 16px #38bdf8;}
.aee-update-title{margin:15px 0 8px;color:var(--text);font-size:clamp(1.45rem,3vw,2rem);
  font-weight:800;letter-spacing:-.045em;}
.aee-update-copy{max-width:680px;color:var(--text-muted);font-size:.94rem;line-height:1.7;}
.aee-version-pill{display:inline-flex;align-items:center;margin:15px 0 0;padding:6px 11px;
  border:1px solid var(--border-strong);border-radius:999px;background:rgba(148,163,184,.08);
  color:var(--text-muted);font-family:'JetBrains Mono',monospace;font-size:.76rem;}
.aee-update-status{max-width:900px;margin:14px auto 0;padding:16px 18px;border:1px solid var(--border);
  border-radius:16px;background:rgba(15,23,42,.52);color:var(--text-muted);line-height:1.65;}
.aee-update-status strong{color:var(--text);}
.aee-update-status[data-tone="success"]{border-color:rgba(16,185,129,.32);background:rgba(16,185,129,.07);}
.aee-update-status[data-tone="available"]{border-color:rgba(59,130,246,.42);background:rgba(59,130,246,.09);}
.aee-update-status[data-tone="error"]{border-color:rgba(239,68,68,.32);background:rgba(239,68,68,.07);}
.aee-update-progress{display:grid;gap:7px;margin-top:14px;}
.aee-update-progress-label{display:flex;justify-content:space-between;gap:12px;
  color:var(--text-muted);font-size:.74rem;font-weight:650;}
.aee-update-progress-track{height:9px;overflow:hidden;border:1px solid rgba(148,163,184,.2);
  border-radius:999px;background:rgba(148,163,184,.12);}
.aee-update-progress-fill{height:100%;border-radius:inherit;
  background:linear-gradient(90deg,#38bdf8,#6366f1,#a78bfa);
  box-shadow:0 0 16px rgba(99,102,241,.4);transition:width .22s ease;}
.aee-update-progress-indeterminate .aee-update-progress-fill{
  width:35%!important;animation:aee-update-progress-sweep 1.25s ease-in-out infinite alternate;}
@keyframes aee-update-progress-sweep{from{transform:translateX(0)}
  to{transform:translateX(185%)}}
.aee-update-notes{margin:10px 0 0;color:var(--text-muted);white-space:pre-wrap;}
.aee-update-dialog-shell{position:fixed!important;inset:0!important;z-index:12000!important;
  padding:22px!important;
  background:rgba(2,6,23,.68)!important;backdrop-filter:blur(12px)!important;}
.aee-update-dialog-shell>div{position:absolute!important;left:50%!important;top:50%!important;
  transform:translate(-50%,-50%)!important;width:min(620px,calc(100vw - 32px))!important;max-height:min(82vh,720px)!important;
  overflow:auto!important;padding:0!important;border:0!important;background:transparent!important;box-shadow:none!important;}
.aee-update-dialog{position:relative;padding:clamp(24px,4vw,36px);border:1px solid rgba(96,165,250,.3);
  border-radius:26px;background:radial-gradient(520px 240px at 100% 0%,rgba(59,130,246,.18),transparent 68%),
  linear-gradient(150deg,#111b30,#080d18);box-shadow:0 35px 110px -34px rgba(0,0,0,.85),0 0 0 1px rgba(148,163,184,.06) inset;}
.aee-update-dialog-head{display:flex;align-items:flex-start;justify-content:space-between;gap:18px;margin-bottom:20px;}
.aee-update-dialog-kicker{color:#93c5fd;font-size:.72rem;font-weight:800;letter-spacing:.16em;text-transform:uppercase;}
.aee-update-dialog-title{margin:6px 0 0;color:var(--text);font-size:1.45rem;font-weight:800;letter-spacing:-.04em;}
.aee-update-dialog-copy{margin:0 0 18px;color:var(--text-muted);line-height:1.7;}
.aee-update-dialog-actions{margin-top:20px;}
.aee-update-dialog-close{min-width:42px!important;width:42px!important;height:42px!important;border:1px solid var(--border)!important;
  border-radius:14px!important;background:rgba(148,163,184,.08)!important;color:var(--text)!important;}
.aee-update-dialog-shell .aee-update-status{max-width:none;margin:0;}
html[data-theme="light"] .aee-update-card{background:radial-gradient(600px 240px at 100% 0%,rgba(59,130,246,.12),transparent 68%),linear-gradient(150deg,#fff,#f4f7fc);}
html[data-theme="light"] .aee-update-status{background:rgba(255,255,255,.8);}
html[data-theme="light"] .aee-update-dialog{background:radial-gradient(520px 240px at 100% 0%,rgba(59,130,246,.1),transparent 68%),linear-gradient(150deg,#fff,#f4f7fc);}
.aee-nav-toggle{position:fixed;z-index:10002;left:16px;top:20px;width:48px;height:48px;display:grid;place-items:center;
  border:1px solid var(--border-strong);border-radius:15px;color:var(--text);cursor:pointer;
  background:linear-gradient(145deg,rgba(30,41,59,.96),rgba(10,15,28,.96));
  box-shadow:0 12px 32px -16px rgba(2,6,23,.85);transition:transform .2s,border-color .2s,box-shadow .2s;}
.aee-nav-toggle:hover{transform:translateY(-2px);border-color:rgba(96,165,250,.6);box-shadow:0 14px 35px -15px rgba(59,130,246,.5);}
.aee-nav-toggle svg{width:21px;height:21px;}
.aee-nav-backdrop{position:fixed;inset:0;z-index:10000;background:rgba(2,6,23,.55);backdrop-filter:blur(3px);
  opacity:0;visibility:hidden;transition:opacity .24s,visibility .24s;}
.aee-nav-drawer{position:fixed;inset:0 auto 0 0;z-index:10001;width:min(340px,88vw);padding:24px 18px;
  display:flex;flex-direction:column;gap:22px;overflow-y:auto;transform:translateX(-105%);transition:transform .28s cubic-bezier(.2,.75,.25,1);
  background:linear-gradient(180deg,#0c1426,#070b14);border-right:1px solid rgba(148,163,184,.18);
  box-shadow:24px 0 80px -34px rgba(0,0,0,.8);}
.aee-nav-open .aee-nav-drawer{transform:translateX(0);}
.aee-nav-open .aee-nav-backdrop{opacity:1;visibility:visible;}
.aee-nav-open{overflow:hidden;}
.aee-nav-head{display:flex;align-items:center;gap:12px;padding:8px 6px 18px;border-bottom:1px solid var(--border);}
.aee-nav-logo{width:42px;height:42px;display:grid;place-items:center;border-radius:13px;
  background:linear-gradient(145deg,#111b36,#090d18);border:1px solid rgba(147,197,253,.28);
  box-shadow:0 8px 25px -12px rgba(59,130,246,.8);}
.aee-nav-logo svg{width:27px;height:27px;}
.aee-nav-brand{min-width:0;flex:1;}
.aee-nav-brand strong{display:block;color:#f1f5f9;font-size:.91rem;letter-spacing:-.02em;}
.aee-nav-brand span{display:block;margin-top:3px;color:#8493ac;font-size:.7rem;}
.aee-nav-close{width:34px;height:34px;border:1px solid rgba(148,163,184,.18);border-radius:10px;
  color:#aebbd0;background:rgba(148,163,184,.06);cursor:pointer;font-size:1.1rem;}
.aee-nav-group-label{margin:0 10px 8px;color:#64748b;font-size:.65rem;font-weight:750;letter-spacing:.14em;text-transform:uppercase;}
.aee-nav-items{display:grid;gap:6px;}
.aee-nav-item{width:100%;display:flex;align-items:center;gap:12px;padding:12px 13px;border:1px solid transparent;
  border-radius:13px;text-align:left;color:#b9c5d8;background:transparent;cursor:pointer;font:600 .86rem Inter,system-ui,sans-serif;
  transition:background .18s,border-color .18s,color .18s,transform .18s;}
.aee-nav-item:hover{transform:translateX(2px);color:#f8fafc;border-color:rgba(96,165,250,.22);background:rgba(59,130,246,.11);}
.aee-nav-item svg{width:19px;height:19px;flex:none;color:#8fb9ff;}
.aee-nav-spacer{flex:1;min-height:12px;}
.aee-nav-foot{padding:14px 10px;border-top:1px solid var(--border);color:#687993;font-size:.72rem;line-height:1.6;}
.aee-utility-window-shell{display:none;position:fixed!important;inset:0!important;z-index:11000!important;
  width:100vw!important;height:100vh!important;height:100dvh!important;max-height:none!important;
  overflow:hidden!important;overscroll-behavior:contain;box-sizing:border-box!important;
  padding:0!important;border:0!important;border-radius:0!important;opacity:0;
  background:radial-gradient(1000px 640px at 92% 0%,rgba(79,70,229,.17),transparent 70%),
    radial-gradient(1000px 650px at 0% 100%,rgba(14,165,233,.12),transparent 70%),
    linear-gradient(145deg,#0b1220,#060910 76%)!important;
  transition:opacity .24s ease;}
.aee-utility-window-shell.aee-utility-window-open{opacity:1;}
.aee-utility-window-shell .column.wrap{position:absolute!important;inset:0!important;
  width:100vw!important;max-width:100vw!important;height:100%!important;
  max-height:100dvh!important;margin:0!important;padding:0 28px 42px!important;
  overflow-y:auto!important;overflow-x:hidden!important;overscroll-behavior-y:contain;
  -webkit-overflow-scrolling:touch;box-sizing:border-box!important;
  background:transparent!important;transform:translateY(12px) scale(.995);
  transition:transform .28s cubic-bezier(.2,.75,.25,1);}
.aee-utility-window-shell.aee-utility-window-open .column.wrap{transform:translateY(0) scale(1);}
.aee-utility-window-active{overflow:hidden!important;}
.aee-utility-window-header{position:sticky;top:0;z-index:12;max-width:none;
  margin:0 -28px 30px;padding:0 clamp(20px,4vw,58px);
  display:flex;align-items:center;gap:20px;min-height:86px;
  border:0;border-bottom:1px solid rgba(148,163,184,.14);border-radius:0;
  background:rgba(10,16,28,.92);box-shadow:0 12px 32px -28px rgba(0,0,0,.8);
  backdrop-filter:blur(18px) saturate(1.15);}
.aee-utility-window-header>div:first-child{flex:1;min-width:0;}
.aee-utility-window-shell .aee-project-page-heading{width:100%;max-width:1180px;
  margin:0 auto;gap:14px;align-items:center;}
.aee-utility-window-shell .aee-project-page-mark{width:44px;height:44px;border-radius:13px;
  color:#bfdbfe;border-color:rgba(96,165,250,.2);
  background:rgba(37,99,235,.13);box-shadow:none;}
.aee-utility-window-shell .aee-project-page-mark svg{width:20px;height:20px;}
.aee-utility-window-shell .aee-project-page-heading small{display:block;width:auto;
  max-width:100%;margin:0 0 3px;padding:0;border:0;border-radius:0;
  color:#8aa4c5;background:none;font-size:.59rem;letter-spacing:.12em;}
.aee-utility-window-shell .aee-project-page-heading h1{margin:0;color:#f1f5f9;
  font-size:1.38rem;font-weight:700;letter-spacing:-.035em;line-height:1.2;}
.aee-utility-window-shell .aee-project-page-heading p{margin:3px 0 0;
  color:#94a3b8;font-size:.75rem;line-height:1.4;}
.aee-utility-window-close{position:relative;z-index:1;min-width:112px!important;height:40px!important;
  padding:0 13px!important;border:1px solid rgba(148,163,184,.2)!important;
  border-radius:10px!important;color:#dbe3ef!important;background:rgba(148,163,184,.06)!important;
  box-shadow:none;transition:border-color .18s,background .18s,color .18s!important;}
.aee-utility-window-close:hover{border-color:rgba(148,163,184,.38)!important;
  color:#fff!important;background:rgba(148,163,184,.12)!important;}
html[data-theme="light"] .aee-utility-window-shell{background:rgba(15,23,42,.32)!important;}
html[data-theme="light"] .aee-utility-window-shell .column.wrap{
  background:transparent!important;}
html[data-theme="light"] .aee-utility-window-shell{
  background:radial-gradient(1000px 640px at 92% 0%,rgba(79,70,229,.09),transparent 70%),
    radial-gradient(1000px 650px at 0% 100%,rgba(14,165,233,.07),transparent 70%),
    linear-gradient(145deg,#f8faff,#edf2fa 76%)!important;}
html[data-theme="light"] .aee-utility-window-header{
  background:rgba(255,255,255,.94);border-bottom-color:rgba(100,116,139,.17);
  box-shadow:0 12px 28px -28px rgba(15,23,42,.42);}
html[data-theme="light"] .aee-utility-window-shell .aee-project-page-heading h1{color:#172033;}
html[data-theme="light"] .aee-utility-window-shell .aee-project-page-heading p{color:#64748b;}
html[data-theme="light"] .aee-utility-window-shell .aee-project-page-heading small{
  color:#64748b;}
html[data-theme="light"] .aee-utility-window-shell .aee-project-page-mark{
  color:#2563eb;background:rgba(37,99,235,.08);border-color:rgba(37,99,235,.14);}
html[data-theme="light"] .aee-utility-window-close{color:#334155!important;
  background:rgba(100,116,139,.055)!important;border-color:rgba(100,116,139,.2)!important;}
html[data-theme="light"] .aee-utility-window-close:hover{color:#172033!important;
  background:rgba(100,116,139,.1)!important;border-color:rgba(100,116,139,.32)!important;}
.aee-utility-window-shell .aee-update-card{max-width:1180px;margin:0 auto 24px;padding:clamp(30px,5vw,64px);
  border-radius:30px;box-shadow:0 30px 90px -55px rgba(37,99,235,.6);}
.aee-utility-window-shell .aee-update-copy{max-width:760px;}
.aee-utility-window-shell .aee-update-status{max-width:1180px;margin:0 auto;padding:18px 22px;}
.aee-utility-window-shell .aee-update-actions{width:100%;max-width:1180px;margin:18px auto 0;gap:12px;}
.aee-utility-window-shell .aee-update-actions button{min-height:48px;border-radius:13px!important;
  font-weight:700!important;box-shadow:0 10px 28px -20px rgba(37,99,235,.9);}
.aee-utility-window-shell .aee-update-actions>div{flex:1 1 220px;}
.aee-hidden-trigger{display:none!important;}
.aee-account-shell .column.wrap{display:flex!important;flex-direction:column!important;}
.aee-account-shell .column.wrap>.aee-account-auth-form{box-sizing:border-box!important;
  width:min(480px,calc(100vw - 36px))!important;max-width:480px;
  margin:clamp(18px,5vh,52px) auto 24px;padding:clamp(28px,5vw,54px) clamp(18px,6vw,26px) 34px;
  border:1px solid rgba(148,163,184,.2);border-radius:27px;position:relative;overflow:hidden;
  background:radial-gradient(440px 200px at 50% -5%,rgba(148,163,184,.17),transparent 75%),
    radial-gradient(460px 300px at 0% 55%,rgba(37,99,235,.15),transparent 78%),
    linear-gradient(155deg,#151719 0%,#101113 60%,#0e0f11 100%);
  box-shadow:0 32px 90px -52px rgba(0,0,0,.94),0 1px 0 rgba(255,255,255,.07) inset;}
.aee-account-shell .column.wrap>.aee-account-auth-form .aee-account-auth-form{
  width:100%!important;max-width:100%!important;margin:0!important;padding:0!important;
  border:0!important;border-radius:0!important;background:transparent!important;
  box-shadow:none!important;overflow:visible!important;}
.aee-account-shell .aee-account-auth-form .styler{overflow:visible!important;
  width:100%!important;max-width:100%!important;padding:0!important;
  border:0!important;background:transparent!important;box-shadow:none!important;}
.aee-account-auth-form .gr-group{overflow:visible!important;
  border:0!important;background:transparent!important;box-shadow:none!important;}
.aee-account-intro{position:relative;margin:0 0 46px;padding:0;text-align:center;}
.aee-account-intro h2{margin:0;color:#f7f8fa;font-size:clamp(2rem,4.2vw,2.55rem);
  font-weight:500;letter-spacing:-.045em;line-height:1.08;}
.aee-account-intro p{max-width:390px;margin:12px auto 0;color:#a0a2a7;
  font-size:.82rem;line-height:1.55;font-weight:400;}
.aee-account-auth-form .aee-account-name-field{margin-bottom:0;}
.aee-account-auth-form label{width:100%!important;min-width:0!important;}
.aee-account-auth-form label>span[data-testid="block-info"]{position:absolute!important;width:1px!important;height:1px!important;
  padding:0!important;margin:-1px!important;overflow:hidden!important;clip:rect(0,0,0,0)!important;
  white-space:nowrap!important;border:0!important;}
.aee-account-auth-form .input-container{width:100%!important;min-width:0!important;}
.aee-account-auth-form input{min-height:44px!important;padding:0 18px!important;
  width:100%!important;min-width:0!important;box-sizing:border-box!important;
  border:1px solid rgba(59,130,246,.66)!important;border-radius:999px!important;
  color:#f4f6fa!important;background:rgba(11,12,14,.8)!important;
  box-shadow:0 1px 0 rgba(255,255,255,.025) inset!important;transition:border-color .18s,box-shadow .18s!important;}
.aee-account-auth-form input:focus{border-color:#3b82f6!important;
  box-shadow:0 0 0 3px rgba(59,130,246,.14)!important;}
.aee-account-auth-form input::placeholder{color:#767b85!important;font-size:.78rem;}
.aee-account-auth-form .aee-account-actions{display:flex;flex-direction:column;gap:7px;margin-top:46px;}
.aee-account-auth-form .aee-account-actions>div{width:100%;flex:none;}
.aee-account-auth-form button.aee-account-submit{width:100%!important;min-height:48px;border:1px solid rgba(255,255,255,.045)!important;
  border-radius:999px!important;color:#f5f6f8!important;background:linear-gradient(180deg,#282a2d,#202123)!important;
  font-size:.9rem!important;font-weight:500!important;letter-spacing:0;
  box-shadow:0 8px 22px -16px rgba(0,0,0,.8)!important;transition:transform .18s,filter .18s!important;}
.aee-account-auth-form button.aee-account-submit:hover{transform:translateY(-1px);filter:brightness(1.12);}
.aee-account-auth-form button.aee-account-mode-switch{width:100%!important;min-height:31px;padding:2px 5px!important;
  border:0!important;color:#b7b9bf!important;background:transparent!important;
  box-shadow:none!important;font-size:.76rem!important;font-weight:400!important;}
.aee-account-auth-form button.aee-account-mode-switch:hover{color:#fff!important;background:transparent!important;}
.aee-account-auth-form button.aee-account-google{width:100%!important;min-height:44px;
  border:1px solid rgba(255,255,255,.12)!important;border-radius:999px!important;
  color:#e5e7eb!important;background:rgba(255,255,255,.055)!important;
  font-size:.8rem!important;font-weight:550!important;transition:border-color .18s,background .18s!important;}
.aee-account-auth-form button.aee-account-google:hover{border-color:rgba(148,163,184,.34)!important;
  background:rgba(255,255,255,.1)!important;}
.aee-account-auth-form .aee-account-divider{margin:14px auto 11px;color:#85878d;font-size:.69rem;
  font-weight:400;letter-spacing:0;text-transform:none;}
.aee-account-auth-form .aee-account-divider:before,.aee-account-auth-form .aee-account-divider:after{
  background:linear-gradient(90deg,transparent,rgba(148,163,184,.17),transparent);}
.aee-account-auth-form button.aee-account-guest{width:100%!important;min-height:40px;border:1px solid rgba(255,255,255,.055)!important;
  border-radius:999px!important;color:#aeb0b5!important;background:rgba(255,255,255,.025)!important;
  font-size:.75rem!important;font-weight:500!important;transition:border-color .18s,background .18s,color .18s!important;}
.aee-account-auth-form button.aee-account-guest:hover{border-color:rgba(59,130,246,.42)!important;
  color:#fff!important;background:rgba(59,130,246,.07)!important;}
.aee-account-auth-privacy{max-width:410px;margin:13px auto 0;text-align:center;
  color:#777a82;font-size:.68rem;line-height:1.55;}
.aee-account-auth-privacy strong{color:#a6a8ad;font-weight:600;}
.aee-account-setup-hint{margin:13px auto 0;text-align:center;color:#8c9098;
  font-size:.68rem;line-height:1.55;}
.aee-account-setup-hint strong{color:#c3c7cf;font-weight:650;}
.aee-account-auth-form .aee-account-status{margin:0 0 18px;padding:12px 14px;font-size:.76rem;}
.aee-account-auth-form .aee-account-status[hidden]{display:none!important;}
body.aee-access-required #aee-account-window{z-index:12000!important;
  background:radial-gradient(860px 520px at 8% 100%,rgba(20,42,82,.75),transparent 76%),
    radial-gradient(820px 520px at 95% 100%,rgba(17,37,72,.62),transparent 78%),
    linear-gradient(145deg,#08090b 0%,#090a0d 55%,#0e1729 100%)!important;}
body.aee-access-required #aee-account-window .column.wrap{display:flex!important;
  align-items:center!important;justify-content:center!important;overflow-y:auto!important;
  padding:26px 16px!important;}
body.aee-access-required #aee-account-window .aee-utility-window-header{display:none!important;}
body.aee-access-required #aee-account-window .column.wrap>.aee-account-auth-form{margin:0 auto!important;}
body.aee-access-required #aee-account-window .aee-account-auth-form .aee-account-intro{margin-bottom:42px;}
body.aee-access-required .aee-navigation{display:none!important;}
body.aee-access-required .aee-toolbar .aee-profile-tool{display:none!important;}
.aee-account-status{max-width:600px;margin:0 auto 18px;padding:14px 18px;border:1px solid rgba(96,165,250,.2);
  border-radius:14px;color:#bfdbfe;background:rgba(37,99,235,.09);font-size:.85rem;line-height:1.55;}
.aee-account-status[data-kind="success"]{color:#a7f3d0;border-color:rgba(16,185,129,.24);background:rgba(16,185,129,.08);}
.aee-account-status[data-kind="error"]{color:#fecaca;border-color:rgba(248,113,113,.26);background:rgba(239,68,68,.08);}
.aee-account-status[data-kind="neutral"]{color:var(--text-muted);border-color:var(--border);background:rgba(148,163,184,.06);}
.aee-account-profile{max-width:1080px;display:flex;align-items:center;gap:18px;margin:0 auto 20px;padding:20px;
  border:1px solid rgba(148,163,184,.16);border-radius:18px;background:rgba(15,23,42,.42);}
.aee-account-avatar{width:68px;height:68px;flex:none;display:grid;place-items:center;overflow:hidden;
  border:1px solid rgba(96,165,250,.36);border-radius:22px;color:#dbeafe;
  background:linear-gradient(145deg,rgba(37,99,235,.32),rgba(79,70,229,.22));font-size:1.25rem;font-weight:800;}
.aee-account-avatar img{width:100%;height:100%;object-fit:cover;}
.aee-account-identity{min-width:0;}
.aee-account-identity strong{display:block;overflow-wrap:anywhere;color:var(--text);font-size:1rem;font-weight:800;}
.aee-account-identity span{display:block;margin-top:5px;overflow-wrap:anywhere;color:var(--text-muted);font-size:.82rem;}
.aee-account-identity .aee-account-mode{display:inline-flex;margin-top:7px;overflow-wrap:normal;}
.aee-account-identity .aee-account-mode>span{display:inline;margin:0;overflow-wrap:normal;}
.aee-account-mode{display:inline-flex;align-items:center;gap:7px;margin-top:11px;padding:6px 10px;
  border:1px solid rgba(16,185,129,.2);border-radius:999px;color:#6ee7b7;background:rgba(16,185,129,.07);
  font-size:.68rem;font-weight:750;letter-spacing:.04em;}
.aee-account-mode:before{content:'';width:7px;height:7px;border-radius:50%;background:#34d399;
  box-shadow:0 0 10px rgba(52,211,153,.7);}
.aee-account-mode[data-provider="google"]{border-color:rgba(66,133,244,.24);
  color:#dbeafe;background:rgba(66,133,244,.09);}
.aee-account-mode[data-provider="google"]:before{display:none;}
.aee-account-mode[data-provider="google"] svg{width:14px;height:14px;flex:none;}
.aee-account-profile-form{max-width:1080px;margin:0 auto;}
.aee-settings-layout{max-width:1280px;margin:0 auto;align-items:flex-start;gap:18px!important;}
.aee-settings-layout>.aee-settings-sidebar{position:sticky;top:92px;flex:0 0 208px!important;
  max-width:208px!important;min-width:0!important;}
.aee-settings-layout>.aee-settings-main{min-width:0!important;flex:1 1 0!important;gap:16px!important;}
.aee-settings-nav{display:grid;gap:5px;padding:12px;border:1px solid rgba(148,163,184,.14);
  border-radius:17px;background:rgba(15,23,42,.66);box-shadow:0 18px 44px -34px rgba(0,0,0,.78);}
.aee-settings-nav-label{padding:4px 9px 9px;color:#818da2;font-size:.66rem;font-weight:800;
  letter-spacing:.14em;text-transform:uppercase;}
.aee-settings-nav button{width:100%;display:flex;align-items:center;gap:10px;padding:10px 11px;
  border:1px solid transparent;border-radius:11px;color:#b8c3d5;background:transparent;
  text-align:left;font:600 .81rem Inter,system-ui,sans-serif;cursor:pointer;
  transition:color .16s,border-color .16s,background .16s,transform .16s;}
.aee-settings-nav button:hover{color:#f8fafc;border-color:rgba(148,163,184,.12);
  background:rgba(148,163,184,.08);transform:translateX(1px);}
.aee-settings-nav button[aria-current="page"]{color:#f8fafc;border-color:rgba(248,113,113,.2);
  background:linear-gradient(90deg,rgba(248,113,113,.13),rgba(148,163,184,.045));}
.aee-settings-language-control{position:relative;display:block;color:#b8c3d5;}
.aee-settings-language-control>summary{min-height:42px;box-sizing:border-box;
  display:flex;align-items:center;gap:10px;padding:10px 11px;list-style:none;
  border:1px solid rgba(148,163,184,.12);border-radius:11px;
  background:rgba(148,163,184,.045);font-size:.81rem;font-weight:650;
  cursor:pointer;transition:border-color .16s,background .16s,color .16s;}
.aee-settings-language-control>summary::-webkit-details-marker{display:none;}
.aee-settings-language-control>summary:hover,
.aee-settings-language-control[open]>summary{color:#f8fafc;
  border-color:rgba(148,163,184,.24);background:rgba(148,163,184,.09);}
.aee-settings-language-control>summary>svg:first-child{width:16px;height:16px;
  flex:none;color:#a9b7cb;}
.aee-settings-language-control>summary>span{flex:1;}
.aee-settings-language-chevron{width:14px;height:14px;flex:none;
  color:#8492a8;transition:transform .16s;}
.aee-settings-language-control[open] .aee-settings-language-chevron{transform:rotate(180deg);}
.aee-settings-language-menu{position:absolute;z-index:20;top:calc(100% + 6px);
  right:0;left:0;display:grid;gap:4px;padding:6px;
  border:1px solid rgba(148,163,184,.2);border-radius:12px;
  background:#172033;box-shadow:0 18px 40px -20px rgba(0,0,0,.75);}
.aee-settings-language-menu button{width:100%;min-height:38px!important;
  display:flex;align-items:center;justify-content:space-between;gap:10px;
  padding:0 10px!important;border:1px solid transparent!important;
  border-radius:8px!important;color:#cbd5e1!important;background:transparent!important;
  text-align:left;font:600 .78rem Inter,system-ui,sans-serif!important;cursor:pointer;}
.aee-settings-language-menu button:hover,
.aee-settings-language-menu button[aria-pressed="true"]{
  color:#fff!important;border-color:rgba(148,163,184,.14)!important;
  background:rgba(148,163,184,.1)!important;}
.aee-settings-language-menu button:focus-visible{outline:2px solid #ff7770!important;
  outline-offset:1px!important;}
.aee-settings-language-menu button[aria-pressed="true"]:after{
  content:"✓";color:#fca5a5;font-weight:800;}
.aee-settings-language-native{display:none!important;}
.aee-settings-nav button:focus-visible,.aee-settings-theme-button:focus-visible,
.aee-settings-layout button:focus-visible,.aee-settings-layout input:focus-visible,
.aee-settings-layout select:focus-visible{outline:2px solid #ff7770!important;outline-offset:2px!important;}
.aee-settings-nav svg{width:16px;height:16px;flex:none;color:#a9b7cb;}
.aee-settings-panel{min-width:0;scroll-margin-top:90px;}
.aee-account-shell .aee-project-page-heading{gap:13px;}
.aee-account-shell .aee-project-page-heading h1{font-size:1.55rem;letter-spacing:-.045em;}
.aee-settings-profile-header{position:relative!important;align-items:stretch!important;
  gap:14px!important;margin:0 0 16px!important;}
.aee-settings-profile-header{
  border:0!important;background:transparent!important;box-shadow:none!important;}
.aee-settings-profile-header>#aee-account-profile-source{flex:1 1 auto!important;width:auto!important;min-width:0!important;}
.aee-settings-photo-actions{position:absolute!important;z-index:3;right:16px;bottom:15px;
  flex:none!important;width:126px!important;min-width:0!important;
  align-items:stretch!important;justify-content:flex-end!important;gap:8px!important;margin:0!important;}
.aee-settings-photo-button,.aee-settings-photo-button button{
  width:100%!important;min-height:40px!important;padding:0 13px!important;
  border:1px solid rgba(255,255,255,.12)!important;border-radius:10px!important;
  color:#f5f7fb!important;background:rgba(16,24,40,.72)!important;
  box-shadow:0 10px 25px -20px rgba(0,0,0,.9)!important;font-size:.75rem!important;
  font-weight:650!important;white-space:nowrap!important;}
.aee-settings-photo-button button:hover{border-color:rgba(255,255,255,.24)!important;
  background:rgba(31,41,59,.92)!important;}
.aee-settings-photo-preview{width:70px!important;min-width:70px!important;max-width:70px!important;
  margin:0!important;}
.aee-settings-photo-preview .image-container{min-height:68px!important;max-height:68px!important;
  border:1px solid rgba(255,255,255,.16)!important;border-radius:12px!important;
  background:rgba(48,54,71,.56)!important;}
.aee-settings-photo-preview .image-container img{object-fit:cover!important;}
.aee-settings-photo-preview .wrap{padding:3px!important;}
.aee-settings-photo-preview label{display:none!important;}
.aee-photo-editor-modal{position:fixed;inset:0;z-index:10000;display:grid;
  place-items:center;box-sizing:border-box;overflow:auto;padding:18px;
  color:#f4f6f8;background:rgba(5,8,15,.74);backdrop-filter:blur(8px);
  font-family:Inter,system-ui,sans-serif;}
body.aee-photo-editor-open{overflow:hidden!important;}
.aee-photo-editor-modal[hidden]{display:none!important;}
.aee-photo-editor-window{width:min(760px,calc(100vw - 36px));
  height:min(720px,calc(100vh - 36px));height:min(720px,calc(100dvh - 36px));
  min-height:min(420px,calc(100dvh - 36px));display:grid;
  grid-template-rows:58px minmax(0,1fr) 64px;overflow:hidden;
  border:1px solid rgba(255,255,255,.12);border-radius:18px;
  color:#f4f6f8;background:#090b0e;
  box-shadow:0 28px 90px rgba(0,0,0,.58);}
.aee-photo-editor-header{display:flex;align-items:center;justify-content:space-between;
  gap:14px;padding:0 18px;border-bottom:1px solid rgba(255,255,255,.08);
  background:#121417;}
.aee-photo-editor-heading{display:flex;align-items:center;gap:18px;min-width:0;}
.aee-photo-editor-back,.aee-photo-editor-apply,.aee-photo-zoom-button{
  display:grid;place-items:center;border:0;cursor:pointer;font:inherit;}
.aee-photo-editor-back{width:36px;height:36px;border-radius:50%;color:#e7ebf0;
  background:transparent;font-size:23px;line-height:1;}
.aee-photo-editor-back:hover{background:rgba(255,255,255,.1);}
.aee-photo-editor-title{font-size:17px;font-weight:700;letter-spacing:-.015em;}
.aee-photo-editor-apply{min-width:82px;height:38px;padding:0 17px;border:1px solid rgba(255,93,85,.32);
  border-radius:10px;color:#fff;background:linear-gradient(120deg,#ef665f,#dc4c59);
  box-shadow:0 10px 24px -16px rgba(248,113,113,.9);font-size:13px;font-weight:750;}
.aee-photo-editor-apply:hover{background:linear-gradient(120deg,#f34d49,#d94150);}
.aee-photo-editor-stage{position:relative;display:grid;place-items:center;min-height:0;
  padding:14px 18px;background:#090b0e;}
.aee-photo-crop-canvas{display:block;width:min(100%,680px);height:min(100%,68vh);
  min-height:220px;touch-action:none;cursor:grab;}
.aee-photo-crop-canvas:active{cursor:grabbing;}
.aee-photo-editor-error{position:absolute;bottom:12px;left:50%;max-width:calc(100% - 32px);
  transform:translateX(-50%);padding:8px 12px;border:1px solid rgba(248,113,113,.35);
  border-radius:10px;color:#fecaca;background:rgba(69,20,25,.94);font-size:12px;}
.aee-photo-editor-error[hidden]{display:none;}
.aee-photo-editor-zoom{display:flex;align-items:center;justify-content:center;gap:13px;
  padding:0 20px;border-top:1px solid rgba(255,255,255,.06);background:#0b0d10;}
.aee-photo-zoom-button{width:30px;height:30px;border:1px solid #394149;border-radius:50%;
  color:#aab3bb;background:transparent;font-size:20px;line-height:1;}
.aee-photo-zoom-button:hover{border-color:#6b7781;color:#fff;}
.aee-photo-zoom-range{width:min(54vw,300px);height:4px;accent-color:#ff5d55;
  cursor:pointer;touch-action:pan-y;}
.aee-photo-transfer{display:none!important;}
.aee-photo-editor-result{display:none!important;}
.aee-photo-hidden-file{display:none!important;}
.aee-settings-profile-header .aee-account-profile{height:auto;margin:0;}
.aee-account-shell .aee-account-profile-form{
  border:0!important;background:transparent!important;box-shadow:none!important;padding:0!important;}
.aee-account-shell .aee-account-profile-form>.styler{
  border:0!important;background:transparent!important;box-shadow:none!important;padding:0!important;}
.aee-account-shell [id="aee-settings-profile-section"]>.styler,
.aee-account-shell .aee-settings-card>.styler{
  padding:0!important;border:0!important;background:transparent!important;box-shadow:none!important;}
.aee-account-shell [id="aee-settings-profile-section"]{
  border:0!important;background:transparent!important;box-shadow:none!important;}
.aee-account-profile-form{max-width:none;margin:0;}
.aee-account-shell .aee-account-profile{position:relative;min-height:148px;box-sizing:border-box;
  align-items:center;gap:18px;margin:0 0 16px;padding:20px 170px 20px 22px;
  overflow:hidden;border-color:rgba(148,163,184,.18);border-radius:20px;
  background:radial-gradient(440px 190px at 100% 0%,rgba(248,113,113,.13),transparent 72%),
    linear-gradient(125deg,#182238 0%,#131c2c 58%,#171d2c 100%);
  box-shadow:0 20px 48px -38px rgba(0,0,0,.9),0 1px 0 rgba(255,255,255,.045) inset;}
.aee-account-shell .aee-account-profile:before{content:"";position:absolute;left:0;top:22px;bottom:22px;
  display:block;width:3px;border-radius:0 4px 4px 0;
  background:linear-gradient(180deg,#fb7770,#e64e5e 72%,transparent);}
.aee-account-shell .aee-account-avatar,.aee-account-shell .aee-account-identity{position:relative;z-index:1;}
.aee-account-shell .aee-account-avatar{width:72px;height:72px;border-radius:22px;
  border:2px solid rgba(255,255,255,.72);color:#fff;
  background:linear-gradient(145deg,#fa766a,#c83e59);
  box-shadow:0 0 0 5px rgba(255,255,255,.055),0 12px 28px -14px rgba(248,113,113,.72);}
.aee-account-shell .aee-account-identity{min-width:0;padding:0;}
.aee-account-eyebrow{display:block!important;margin:0 0 5px!important;color:#fca5a5!important;
  font-size:.61rem!important;font-weight:800!important;letter-spacing:.15em!important;
  line-height:1.3!important;text-transform:uppercase;}
.aee-account-shell .aee-account-identity strong{font-size:1.1rem;letter-spacing:-.025em;}
.aee-account-shell .aee-account-identity .aee-account-detail{margin-top:4px;color:#b7c2d3;font-size:.78rem;}
.aee-account-shell .aee-account-identity .aee-account-joined{margin-top:3px;color:#8897ad;font-size:.7rem;}
.aee-account-shell .aee-account-mode{margin-top:9px;padding:5px 9px;font-size:.61rem;}
.aee-settings-profile-header .aee-settings-photo-actions{top:50%;bottom:auto;transform:translateY(-50%);}
.aee-settings-card{box-sizing:border-box;min-width:0;margin:0;padding:18px;
  border:1px solid rgba(148,163,184,.16);border-radius:19px;background:rgba(20,28,45,.78);
  box-shadow:0 18px 46px -40px rgba(0,0,0,.86),0 1px 0 rgba(255,255,255,.025) inset;
  transition:border-color .18s,background .18s,transform .18s;}
.aee-settings-card:hover{border-color:rgba(148,163,184,.23);background:rgba(22,31,50,.84);}
.aee-settings-card>.styler{border-radius:inherit!important;}
.aee-settings-card-heading{display:flex;align-items:flex-start;justify-content:space-between;gap:14px;
  margin:0 0 15px;padding:0 0 13px;border-bottom:1px solid rgba(148,163,184,.1);}
.aee-settings-card-heading>button{flex:0 0 auto!important;width:auto!important;min-width:104px!important;
  min-height:38px!important;padding:0 13px!important;border:1px solid rgba(248,113,113,.34)!important;
  border-radius:10px!important;color:#fff!important;background:linear-gradient(120deg,#ef665f,#dc4c59)!important;
  box-shadow:0 10px 24px -16px rgba(248,113,113,.9);}
.aee-settings-card-heading h2{margin:0;color:#edf2fa;font-size:.97rem;font-weight:740;letter-spacing:-.02em;}
.aee-settings-card-heading p{margin:5px 0 0;color:#909db1;font-size:.76rem;line-height:1.55;}
.aee-settings-fields{display:grid!important;grid-template-columns:repeat(3,minmax(0,1fr));
  width:100%!important;max-width:none!important;gap:14px!important;align-items:start!important;}
.aee-settings-fields>.block{flex:none!important;min-width:0!important;width:100%!important;max-width:none!important;}
.aee-settings-fields label span[data-testid="block-info"],
.aee-settings-fields label span:first-child{color:#a2aabc!important;font-size:.75rem!important;font-weight:550!important;}
.aee-settings-fields input,.aee-settings-fields select{
  min-height:40px!important;box-sizing:border-box!important;border:1px solid rgba(255,255,255,.08)!important;
  border-radius:9px!important;color:#f5f7fb!important;background:#303647!important;
  box-shadow:none!important;transition:border-color .16s,box-shadow .16s,background .16s!important;}
.aee-settings-fields input:hover,.aee-settings-fields select:hover{
  border-color:rgba(255,255,255,.17)!important;}
.aee-settings-fields input:focus,.aee-settings-fields select:focus{
  border-color:rgba(255,93,85,.78)!important;box-shadow:0 0 0 3px rgba(255,93,85,.13)!important;}
.aee-settings-fields input:focus-visible,.aee-settings-fields select:focus-visible{outline:0!important;}
.aee-settings-fields input::placeholder{color:#7e899d!important;}
.aee-settings-fields input[aria-invalid="true"]{
  border-color:#fb7185!important;}
.aee-settings-dirty{display:flex;align-items:center;gap:8px;margin:0 0 12px;
  color:#fbbf9f;font-size:.76rem;line-height:1.4;}
.aee-settings-dirty[hidden]{display:none!important;}
.aee-settings-dirty:before{width:7px;height:7px;border-radius:50%;content:"";
  background:#ff8b65;box-shadow:0 0 9px rgba(255,139,101,.45);}
.aee-settings-save-button{flex:0 0 auto!important;width:auto!important;min-width:112px!important;
  min-height:38px!important;padding:0 14px!important;border:1px solid rgba(255,93,85,.32)!important;
  border-radius:9px!important;color:#fff!important;background:#ff5d55!important;
  box-shadow:0 10px 24px -16px rgba(255,93,85,.86);transition:background .16s,transform .16s!important;}
.aee-settings-save-button:hover{background:#f34d49!important;transform:translateY(-1px);}
.aee-settings-save-button:active{transform:translateY(0);}
.aee-settings-save-button:disabled{opacity:.58!important;cursor:wait!important;transform:none!important;}
.aee-settings-language-card{display:flex;align-items:center;justify-content:space-between;gap:14px;}
.aee-settings-language-card strong{display:block;color:#edf2fa;font-size:.84rem;}
.aee-settings-language-card span{display:block;margin-top:5px;color:#909db1;font-size:.75rem;}
.aee-settings-language-badge{padding:7px 11px;border:1px solid rgba(255,255,255,.08);
  border-radius:9px;color:#e2e8f0;background:rgba(255,255,255,.04);font-size:.77rem;}
.aee-settings-fields .aee-settings-dirty{grid-column:1/-1;}
.aee-settings-card .aee-settings-option-list{max-width:none;margin:0;gap:8px!important;}
.aee-settings-card .aee-settings-option-list button{min-height:44px;border-radius:11px!important;}
#aee-account-signout,#aee-account-signout button,
button#aee-account-signout,.aee-settings-card .aee-account-signout button,
.aee-settings-card button.aee-account-signout{
  width:100%!important;min-height:46px!important;box-sizing:border-box!important;
  border:1px solid rgba(255,105,105,.42)!important;border-radius:11px!important;
  color:#fff!important;background:linear-gradient(115deg,#ef5350,#d9364f)!important;
  background-color:#d9364f!important;
  box-shadow:0 12px 26px -18px rgba(239,68,68,.82)!important;
  font-weight:700!important;transition:transform .18s,box-shadow .18s,filter .18s!important;}
#aee-account-signout button:hover,button#aee-account-signout:hover{
  transform:translateY(-1px);filter:brightness(1.07);
  box-shadow:0 15px 30px -17px rgba(239,68,68,.72)!important;}
#aee-account-signout button:focus-visible,button#aee-account-signout:focus-visible{
  outline:2px solid #fca5a5!important;outline-offset:3px!important;}
.aee-settings-card .aee-settings-option-list button{transition:transform .16s,border-color .16s,background .16s!important;}
.aee-settings-card .aee-settings-option-list button:hover{transform:translateY(-1px);}
.aee-settings-card .aee-account-profile-actions{max-width:none;margin:10px 0 0;}
.aee-history-page{max-width:1280px;margin:0 auto;padding:8px 0 30px;}
.aee-history-window .column.wrap{padding-bottom:56px!important;}
.aee-history-window .aee-history-page{width:100%;max-width:1320px;margin:0 auto;padding:0 0 34px;}
.aee-history-window .aee-history-heading{margin:4px 0 20px;padding:22px 24px;
  border:1px solid rgba(148,163,184,.13);border-radius:18px;
  background:linear-gradient(135deg,rgba(37,99,235,.12),rgba(139,92,246,.07) 58%,rgba(15,23,42,.62));
  box-shadow:0 18px 50px -42px rgba(59,130,246,.65);}
.aee-history-window .aee-history-heading h2{font-size:1.25rem;}
.aee-history-window .aee-history-heading p{max-width:760px;}
.aee-history-window .aee-history-toolbar{padding:16px;border:1px solid rgba(148,163,184,.12);
  border-radius:16px;background:rgba(15,23,42,.48);}
.aee-history-window .aee-history-gallery,
.aee-history-window .aee-history-detections{box-shadow:0 16px 40px -34px rgba(0,0,0,.75);}
.aee-history-window .aee-history-section-heading{margin:24px 0 12px;padding:0 2px;}
.aee-history-window .aee-history-section-heading h2{margin:0;color:#e2e8f0;
  font-size:1rem;font-weight:720;letter-spacing:-.02em;}
.aee-history-window .aee-history-section-heading p{margin:5px 0 0;color:#94a3b8;
  font-size:.76rem;line-height:1.5;}
html[data-theme="light"] .aee-history-window .aee-history-heading{
  border-color:rgba(100,116,139,.16);
  background:linear-gradient(135deg,rgba(219,234,254,.8),rgba(237,233,254,.64) 58%,#fff);}
html[data-theme="light"] .aee-history-window .aee-history-toolbar{border-color:rgba(100,116,139,.15);
  background:rgba(255,255,255,.78);}
html[data-theme="light"] .aee-history-window .aee-history-section-heading h2{color:#172033;}
html[data-theme="light"] .aee-history-window .aee-history-section-heading p{color:#64748b;}
.aee-history-heading{display:flex;align-items:flex-end;justify-content:space-between;
  gap:18px;margin:4px 0 18px;}
.aee-history-heading h2{margin:0;color:#f1f5f9;font-size:1.35rem;font-weight:750;letter-spacing:-.03em;}
.aee-history-heading p{margin:6px 0 0;color:#94a3b8;font-size:.82rem;line-height:1.55;}
.aee-history-toolbar{display:flex;align-items:center;justify-content:flex-end;gap:8px;margin-bottom:14px;}
.aee-history-toolbar>.block{flex:0 0 auto!important;min-width:0;}
.aee-history-toolbar label span{color:#aab5c6!important;font-size:.75rem!important;}
.aee-history-list{display:grid!important;gap:10px!important;margin:0 0 20px;}
.aee-history-row{align-items:center!important;gap:16px!important;padding:14px 16px;
  border:1px solid rgba(148,163,184,.14);border-radius:14px;
  background:linear-gradient(120deg,rgba(20,30,48,.9),rgba(15,23,42,.68));
  box-shadow:0 12px 32px -28px rgba(0,0,0,.8);}
.aee-history-row-info{min-width:0!important;gap:0!important;}
.aee-history-row-title{overflow:hidden;color:#e7edf7;font-size:.9rem;font-weight:700;
  text-overflow:ellipsis;white-space:nowrap;}
.aee-history-row-meta{margin-top:5px;color:#94a3b8;font-size:.72rem;line-height:1.45;}
.aee-history-row-actions{display:flex!important;flex:0 0 auto!important;justify-content:flex-end!important;
  gap:5px!important;width:auto!important;margin-left:auto!important;}
.aee-history-row-actions button{min-height:27px!important;padding:0 8px!important;
  border-radius:7px!important;font-size:.65rem!important;font-weight:650!important;}
.aee-history-row-actions .aee-history-details button,#aee-history-details button,
button#aee-history-details{border:1px solid rgba(125,164,255,.5)!important;color:#eff6ff!important;
  background:linear-gradient(135deg,#3b82f6,#4f46e5)!important;
  box-shadow:0 5px 14px -8px rgba(79,70,229,.85)!important;
  transition:transform .16s ease,box-shadow .16s ease,background .16s ease!important;}
.aee-history-row-actions .aee-history-details button:hover,#aee-history-details button:hover,
button#aee-history-details:hover{border-color:#bfdbfe!important;
  background:linear-gradient(135deg,#60a5fa,#6366f1)!important;
  box-shadow:0 8px 18px -8px rgba(79,70,229,.95)!important;transform:translateY(-1px);}
.aee-history-row-actions .aee-history-download button,#aee-history-download button,
button#aee-history-download{border:1px solid rgba(94,234,212,.42)!important;color:#ecfeff!important;
  background:linear-gradient(135deg,#0f766e,#2563eb)!important;
  box-shadow:0 5px 14px -8px rgba(13,148,136,.85)!important;
  transition:transform .16s ease,box-shadow .16s ease,background .16s ease!important;}
.aee-history-row-actions .aee-history-download button:hover,#aee-history-download button:hover,
button#aee-history-download:hover{border-color:#99f6e4!important;
  background:linear-gradient(135deg,#14b8a6,#3b82f6)!important;
  box-shadow:0 8px 18px -8px rgba(13,148,136,.95)!important;transform:translateY(-1px);}
.aee-history-row-actions .aee-history-remove button,#aee-history-remove button,
button#aee-history-remove{border:1px solid #f87171!important;
  color:#fff!important;background:linear-gradient(135deg,#ef4444,#b91c1c)!important;
  box-shadow:0 5px 14px -7px rgba(239,68,68,.9)!important;}
.aee-history-row-actions .aee-history-remove button:hover,#aee-history-remove button:hover,
button#aee-history-remove:hover{border-color:#fca5a5!important;
  background:linear-gradient(135deg,#f87171,#dc2626)!important;
  box-shadow:0 7px 18px -7px rgba(239,68,68,.95)!important;transform:translateY(-1px);}
html[data-theme="light"] .aee-history-row{border-color:rgba(100,116,139,.16);
  background:linear-gradient(120deg,#fff,#f8fafc);}
html[data-theme="light"] .aee-history-row-title{color:#172033;}
html[data-theme="light"] .aee-history-row-meta{color:#64748b;}
html[data-theme="light"] .aee-history-row-actions .aee-history-remove button,
html[data-theme="light"] #aee-history-remove button{color:#fff!important;}
.aee-history-toolbar input,.aee-history-toolbar select{
  min-height:42px!important;border:1px solid rgba(148,163,184,.17)!important;
  border-radius:10px!important;color:#e2e8f0!important;background:#172033!important;}
.aee-history-refresh button,#aee-history-refresh button,button#aee-history-refresh,
.aee-history-download-all button,#aee-history-download-all button,button#aee-history-download-all{
  min-height:30px!important;padding:0 10px!important;border-radius:9px!important;
  color:#f8fafc!important;font-size:.68rem!important;font-weight:700!important;
  transition:transform .16s ease,box-shadow .16s ease,background .16s ease!important;}
.aee-history-refresh button,#aee-history-refresh button,button#aee-history-refresh{
  border:1px solid rgba(148,163,184,.32)!important;
  background:linear-gradient(135deg,#334155,#1e293b)!important;
  box-shadow:0 5px 14px -9px rgba(15,23,42,.9)!important;}
.aee-history-refresh button:hover,#aee-history-refresh button:hover,button#aee-history-refresh:hover{
  border-color:#94a3b8!important;background:linear-gradient(135deg,#475569,#334155)!important;
  box-shadow:0 8px 18px -8px rgba(15,23,42,.95)!important;transform:translateY(-1px);}
.aee-history-download-all button,#aee-history-download-all button,button#aee-history-download-all{
  border:1px solid rgba(125,164,255,.45)!important;
  background:linear-gradient(135deg,#2563eb,#6d28d9)!important;
  box-shadow:0 6px 16px -9px rgba(79,70,229,.9)!important;}
.aee-history-download-all button:hover,#aee-history-download-all button:hover,
button#aee-history-download-all:hover{border-color:#c4b5fd!important;
  background:linear-gradient(135deg,#3b82f6,#7c3aed)!important;
  box-shadow:0 9px 20px -8px rgba(79,70,229,.95)!important;transform:translateY(-1px);}
.aee-history-clear button,#aee-history-clear button,button#aee-history-clear{
  min-height:30px!important;padding:0 11px!important;border-radius:9px!important;
  border:1px solid #f87171!important;color:#fff!important;
  background:linear-gradient(135deg,#ef4444,#b91c1c)!important;
  box-shadow:0 7px 18px -9px rgba(239,68,68,.9)!important;
  font-size:.68rem!important;font-weight:700!important;letter-spacing:.01em!important;
  transition:transform .16s ease,box-shadow .16s ease,background .16s ease!important;}
.aee-history-clear button:hover,#aee-history-clear button:hover,button#aee-history-clear:hover{
  border-color:#fca5a5!important;background:linear-gradient(135deg,#f87171,#dc2626)!important;
  box-shadow:0 9px 22px -8px rgba(239,68,68,.95)!important;transform:translateY(-1px);}
.aee-history-detail{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));
  gap:10px;margin:0 0 16px;padding:16px;border:1px solid rgba(148,163,184,.14);
  border-radius:16px;background:#111a2a;}
.aee-history-metric{min-width:0;padding:12px;border:1px solid rgba(148,163,184,.1);
  border-radius:11px;background:rgba(148,163,184,.045);}
.aee-history-metric span,.aee-history-metadata dt{display:block;color:#94a3b8;
  font-size:.68rem;font-weight:650;letter-spacing:.035em;}
.aee-history-metric strong{display:block;margin-top:5px;color:#e2e8f0;
  font-size:.9rem;overflow-wrap:anywhere;}
.aee-history-metadata{grid-column:1/-1;display:grid;
  grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:0;padding:12px 2px 2px;
  border-top:1px solid rgba(148,163,184,.1);}
.aee-history-metadata dd{margin:4px 0 0;color:#cbd5e1;font-size:.76rem;line-height:1.45;
  overflow-wrap:anywhere;}
.aee-history-stat-list{grid-column:1/-1;display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px;}
.aee-history-stat{padding:10px 12px;border:1px solid rgba(96,165,250,.14);
  border-radius:10px;background:rgba(37,99,235,.055);}
.aee-history-stat strong,.aee-history-stat span{display:block;font-size:.74rem;}
.aee-history-stat strong{color:#dbeafe;}
.aee-history-stat span{margin-top:4px;color:#9eacc1;}
.aee-history-notes{grid-column:1/-1;margin:0;color:#cbd5e1;font-size:.8rem;line-height:1.6;}
.aee-history-warning{grid-column:1/-1;padding:10px 12px;border:1px solid rgba(248,113,113,.2);
  border-radius:10px;color:#fecaca;background:rgba(239,68,68,.07);font-size:.74rem;}
.aee-history-warning ul{margin:5px 0 0;padding-left:18px;}
.aee-history-export-status{margin:6px 0 12px;padding:10px 12px;border:1px solid rgba(96,165,250,.18);
  border-radius:10px;color:#bfdbfe;background:rgba(37,99,235,.08);font-size:.78rem;}
.aee-history-empty{display:grid;gap:6px;padding:26px;border:1px dashed rgba(148,163,184,.2);
  border-radius:14px;color:#94a3b8;text-align:center;}
.aee-history-empty strong{color:#e2e8f0;font-size:.9rem;}
.aee-history-gallery{margin:0 0 16px;border:1px solid rgba(148,163,184,.12);
  border-radius:14px;background:rgba(15,23,42,.35);}
.aee-history-files{margin:0 0 16px;}
.aee-history-detections{overflow:auto;border:1px solid rgba(148,163,184,.12);
  border-radius:14px;background:rgba(15,23,42,.35);}
.aee-history-detections table{min-width:720px;}
html[data-theme="light"] .aee-history-heading h2{color:#172033;}
html[data-theme="light"] .aee-history-heading p{color:#64748b;}
html[data-theme="light"] .aee-history-toolbar label span{color:#475569!important;}
html[data-theme="light"] .aee-history-toolbar input,html[data-theme="light"] .aee-history-toolbar select{
  color:#172033!important;background:#fff!important;border-color:rgba(100,116,139,.2)!important;}
html[data-theme="light"] .aee-history-refresh button{color:#334155!important;
  background:rgba(100,116,139,.07)!important;border-color:rgba(100,116,139,.2)!important;}
@media(max-width:720px){
  .aee-history-row{align-items:stretch!important;flex-direction:column!important;}
  .aee-history-row-actions{width:100%!important;justify-content:flex-end!important;}
}
html[data-theme="light"] .aee-history-detail{border-color:rgba(100,116,139,.16);background:#fff;}
html[data-theme="light"] .aee-history-metric{border-color:rgba(100,116,139,.12);background:#f8fafc;}
html[data-theme="light"] .aee-history-metric span,html[data-theme="light"] .aee-history-metadata dt{color:#64748b;}
html[data-theme="light"] .aee-history-metric strong{color:#172033;}
html[data-theme="light"] .aee-history-metadata{border-color:rgba(100,116,139,.15);}
html[data-theme="light"] .aee-history-metadata dd,html[data-theme="light"] .aee-history-notes{color:#334155;}
html[data-theme="light"] .aee-history-stat{border-color:rgba(37,99,235,.13);background:#eff6ff;}
html[data-theme="light"] .aee-history-stat strong{color:#1e3a8a;}
html[data-theme="light"] .aee-history-stat span{color:#475569;}
html[data-theme="light"] .aee-history-export-status{color:#1e40af;background:#eff6ff;}
html[data-theme="light"] .aee-history-empty{border-color:rgba(100,116,139,.25);color:#64748b;}
html[data-theme="light"] .aee-history-empty strong{color:#172033;}
@media(max-width:760px){
  .aee-history-detail{grid-template-columns:repeat(2,minmax(0,1fr));padding:12px;}
  .aee-history-metadata{grid-template-columns:repeat(2,minmax(0,1fr));}
  .aee-history-toolbar{align-items:stretch;flex-direction:column;}
}
.aee-settings-card .aee-account-settings-copy{margin:0 0 14px;}
.aee-settings-card .aee-account-settings-copy strong{color:#edf2fa;}
.aee-settings-card .aee-account-settings-copy p{color:#aab5c6;}
.aee-settings-theme-button{min-height:42px;padding:0 14px;border:1px solid rgba(148,163,184,.18);
  border-radius:11px;color:#edf2fa;background:rgba(148,163,184,.08);font:650 .79rem Inter,system-ui,sans-serif;
  cursor:pointer;transition:background .16s,border-color .16s;}
.aee-settings-theme-button:hover{border-color:rgba(248,113,113,.34);background:rgba(248,113,113,.09);}
.aee-settings-layout .aee-account-status{max-width:none;margin:0 auto 14px;}
.aee-settings-layout .aee-account-settings{max-width:none;margin:12px 0 0;}
.aee-settings-layout .aee-account-settings .label-wrap{padding:14px 16px!important;}
.aee-settings-layout .aee-account-settings .wrap{padding:12px 16px 17px;}
.aee-settings-layout .aee-account-settings>.wrap{width:100%!important;max-width:100%!important;min-width:0!important;}
.aee-settings-layout .aee-account-settings .prose{max-width:100%;}
.aee-settings-layout .aee-account-settings .gr-code{max-width:100%;overflow:auto;}
body.aee-access-required #aee-account-window .aee-settings-layout>.aee-settings-sidebar{display:none!important;}
body.aee-access-required #aee-account-window .aee-settings-layout>.aee-settings-main{
  width:min(480px,calc(100vw - 36px))!important;max-width:480px;margin:10px auto 0;}
.aee-settings-section-label{max-width:1080px;margin:24px auto 10px;color:var(--text-muted);
  font-size:.7rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;}
.aee-settings-option-list{max-width:1080px;margin:0 auto;gap:9px!important;}
.aee-settings-option-list button{min-height:48px;border-radius:13px!important;text-align:left;
  font-weight:700!important;}
.aee-account-divider{display:flex;align-items:center;gap:14px;max-width:1080px;margin:18px auto;
  color:var(--text-muted);font-size:.74rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;}
.aee-account-divider:before,.aee-account-divider:after{content:'';height:1px;flex:1;
  background:linear-gradient(90deg,transparent,rgba(148,163,184,.28),transparent);}
.aee-account-shell .aee-account-settings{max-width:1080px;margin:18px auto 0;overflow-x:hidden!important;
  border:1px solid rgba(148,163,184,.17);
  border-radius:18px;background:rgba(15,23,42,.34);}
.aee-account-shell .aee-account-settings .label-wrap{padding:17px 20px!important;
  color:#e5e7eb!important;font-weight:700!important;}
.aee-account-shell .aee-account-settings .wrap{padding:14px 20px 20px;}
.aee-account-shell .aee-account-settings .prose{max-width:900px;color:#b8c0cd;line-height:1.65;}
.aee-account-shell .aee-account-settings a{color:#93c5fd!important;}
.aee-account-settings-copy{max-width:850px;margin:2px 0 18px;color:#b8c0cd;font-size:.86rem;line-height:1.65;}
.aee-account-settings-copy strong{color:#f1f5f9;font-size:.95rem;}
.aee-account-settings-copy p{margin:5px 0 8px;}
.aee-account-settings-copy a{color:#93c5fd!important;font-weight:650;}
.aee-account-shell .aee-account-settings>.wrap{box-sizing:border-box!important;
  width:100%!important;max-width:100%!important;min-width:0!important;}
.aee-account-shell .aee-account-settings>div:first-child{color:var(--text);font-weight:750;}
.aee-account-shell .aee-account-settings .prose{color:var(--text-muted);}
.aee-account-shell .aee-account-settings button{border-radius:12px!important;}
.aee-account-shell .aee-account-profile-form .aee-account-actions button{
  min-height:46px;border-radius:13px!important;font-weight:700!important;}
.aee-account-profile-actions{max-width:1080px;margin:10px auto 0;gap:10px;}
.aee-account-profile-actions>div{flex:1 1 180px;}
.aee-account-profile-actions button{min-height:44px;border-radius:13px!important;}
body.aee-access-required #aee-account-window .aee-account-settings{
  width:min(480px,calc(100vw - 36px));margin:10px auto 0;}
html[data-theme="light"] .aee-account-profile{background:rgba(255,255,255,.68);border-color:rgba(100,116,139,.17);}
html[data-theme="light"] .aee-account-shell .aee-account-settings{background:rgba(255,255,255,.72);border-color:rgba(100,116,139,.2);}
html[data-theme="light"] .aee-account-divider:before,html[data-theme="light"] .aee-account-divider:after{
  background:linear-gradient(90deg,transparent,rgba(100,116,139,.25),transparent);}
html[data-theme="light"] .aee-settings-nav{background:rgba(255,255,255,.82);border-color:rgba(100,116,139,.17);}
html[data-theme="light"] .aee-settings-nav button{color:#475569;}
html[data-theme="light"] .aee-settings-nav button:hover{color:#0f172a;background:rgba(100,116,139,.07);}
html[data-theme="light"] .aee-settings-nav button[aria-current="page"]{color:#0f172a;background:rgba(248,113,113,.1);}
html[data-theme="light"] .aee-settings-language-control{color:#334155;background:rgba(100,116,139,.055);}
html[data-theme="light"] .aee-settings-language-control>summary{color:#334155;
  border-color:rgba(100,116,139,.16);background:rgba(100,116,139,.045);}
html[data-theme="light"] .aee-settings-language-control>summary:hover,
html[data-theme="light"] .aee-settings-language-control[open]>summary{
  color:#172033;border-color:rgba(100,116,139,.27);background:rgba(100,116,139,.09);}
html[data-theme="light"] .aee-settings-language-menu{border-color:rgba(100,116,139,.2);
  background:#fff;box-shadow:0 18px 38px -22px rgba(15,23,42,.32);}
html[data-theme="light"] .aee-settings-language-menu button{color:#334155!important;}
html[data-theme="light"] .aee-settings-language-menu button:hover,
html[data-theme="light"] .aee-settings-language-menu button[aria-pressed="true"]{
  color:#172033!important;border-color:rgba(100,116,139,.15)!important;
  background:#f1f5f9!important;}
html[data-theme="light"] .aee-account-shell .aee-account-profile{
  border-color:rgba(100,116,139,.17);
  background:radial-gradient(440px 190px at 100% 0%,rgba(248,113,113,.09),transparent 72%),
    linear-gradient(125deg,#fff 0%,#f7f8fc 58%,#f1f4f9 100%);
  box-shadow:0 18px 44px -38px rgba(15,23,42,.28),0 1px 0 rgba(255,255,255,.85) inset;}
html[data-theme="light"] .aee-account-shell .aee-account-eyebrow{color:#c24149!important;}
html[data-theme="light"] .aee-account-shell .aee-account-identity .aee-account-detail{color:#475569;}
html[data-theme="light"] .aee-account-shell .aee-account-identity .aee-account-joined{color:#64748b;}
html[data-theme="light"] .aee-account-shell .aee-account-mode[data-provider="google"]{
  color:#1e3a8a;border-color:rgba(66,133,244,.22);background:rgba(66,133,244,.08);}
html[data-theme="light"] .aee-settings-card{background:rgba(255,255,255,.84);border-color:rgba(100,116,139,.17);}
html[data-theme="light"] .aee-settings-card-heading h2{color:#172033;}
html[data-theme="light"] .aee-settings-card-heading p{color:#64748b;}
html[data-theme="light"] .aee-settings-photo-button button{
  color:#172033!important;background:rgba(255,255,255,.82)!important;
  border-color:rgba(100,116,139,.2)!important;}
html[data-theme="light"] .aee-settings-photo-button button:hover{background:#fff!important;}
html[data-theme="light"] .aee-settings-fields input,html[data-theme="light"] .aee-settings-fields select{
  color:#172033!important;background:#f1f4f9!important;border-color:rgba(15,23,42,.12)!important;}
html[data-theme="light"] .aee-settings-fields input:focus,html[data-theme="light"] .aee-settings-fields select:focus{
  border-color:rgba(255,93,85,.78)!important;}
html[data-theme="light"] .aee-settings-theme-button{color:#172033;background:rgba(100,116,139,.07);}
html[data-theme="light"] .aee-settings-card .aee-account-settings-copy strong{color:#172033;}
html[data-theme="light"] .aee-settings-card .aee-account-settings-copy p{color:#475569;}
.aee-about-card,.aee-contact-card{position:relative;max-width:1180px;min-height: min(68vh,760px);
  display:flex;flex-direction:column;justify-content:center;margin:0 auto;padding:clamp(34px,6vw,76px);
  overflow:hidden;border:1px solid rgba(96,165,250,.2);border-radius:26px;
  background:radial-gradient(480px 240px at 100% 0%,rgba(59,130,246,.12),transparent 70%),
    linear-gradient(150deg,rgba(15,23,42,.94),rgba(8,12,23,.97));
  box-shadow:0 30px 90px -55px rgba(37,99,235,.55),0 1px 0 rgba(255,255,255,.04) inset;}
.aee-about-card:before,.aee-contact-card:before{content:'';position:absolute;left:0;top:28px;bottom:28px;width:3px;
  border-radius:0 6px 6px 0;background:linear-gradient(180deg,#38bdf8,#6366f1,transparent);opacity:.85;}
.aee-project-page-heading{max-width:100%;margin:0;display:flex;align-items:center;gap:15px;}
.aee-project-page-mark{width:54px;height:54px;display:grid;place-items:center;flex:none;border:1px solid rgba(96,165,250,.34);
  border-radius:17px;color:#bfdbfe;background:linear-gradient(145deg,rgba(37,99,235,.27),rgba(79,70,229,.18));
  box-shadow:0 12px 30px -18px rgba(59,130,246,.95),0 1px 0 rgba(255,255,255,.1) inset;}
.aee-project-page-mark svg{width:24px;height:24px;}
.aee-project-page-heading small{display:block;color:#7dd3fc;font-size:.67rem;font-weight:800;letter-spacing:.17em;text-transform:uppercase;}
.aee-project-page-heading h1{margin:4px 0 0;color:var(--text);font-size:clamp(1.35rem,2.5vw,1.7rem);font-weight:800;letter-spacing:-.04em;}
.aee-about-title,.aee-contact-title{margin:0 0 15px;color:var(--text);font-size:clamp(1.6rem,3.4vw,2.35rem);font-weight:850;letter-spacing:-.05em;}
.aee-about-copy,.aee-contact-copy{max-width:820px;color:var(--text-muted);font-size:1rem;line-height:1.85;}
.aee-about-copy p{margin:0 0 14px;}
.aee-about-copy p:last-child{margin-bottom:0;}
.aee-project-feature-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;
  max-width:100%;margin:18px auto 0;}
.aee-project-feature{padding:17px 18px;border:1px solid rgba(148,163,184,.15);border-radius:16px;
  background:linear-gradient(145deg,rgba(30,41,59,.58),rgba(15,23,42,.35));}
.aee-project-feature strong{display:block;color:var(--text);font-size:.88rem;font-weight:750;}
.aee-project-feature span{display:block;margin-top:6px;color:var(--text-muted);font-size:.8rem;line-height:1.55;}
.aee-update-steps{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-top:28px;}
.aee-update-step{display:flex;align-items:flex-start;gap:12px;min-width:0;padding:15px 16px;
  border:1px solid rgba(148,163,184,.15);border-radius:16px;background:rgba(15,23,42,.32);}
.aee-update-step-number{display:grid;place-items:center;flex:none;width:27px;height:27px;border-radius:9px;
  color:#bfdbfe;background:rgba(59,130,246,.16);font:700 .72rem 'JetBrains Mono',monospace;}
.aee-update-step strong{display:block;color:var(--text);font-size:.8rem;font-weight:750;}
.aee-update-step span:last-child{display:block;margin-top:4px;color:var(--text-muted);font-size:.73rem;line-height:1.5;}
.aee-about-eyebrow{display:flex;align-items:center;gap:9px;margin-bottom:19px;color:#93c5fd;
  font-size:.7rem;font-weight:800;letter-spacing:.16em;text-transform:uppercase;}
.aee-about-eyebrow-mark{display:grid;place-items:center;width:27px;height:27px;border:1px solid rgba(96,165,250,.28);
  border-radius:9px;color:#bfdbfe;background:rgba(59,130,246,.14);font-size:.9rem;}
.aee-about-title{max-width:760px;font-size:clamp(2rem,5vw,3.4rem);line-height:1.06;}
.aee-about-copy{max-width:760px;font-size:.96rem;line-height:1.8;}
.aee-project-feature-grid{grid-template-columns:repeat(4,minmax(0,1fr));gap:16px;margin-top:38px;}
.aee-project-feature{position:relative;min-height:132px;padding:19px 18px;overflow:hidden;
  transition:transform .2s ease,border-color .2s ease,background .2s ease;}
.aee-project-feature:hover{transform:translateY(-3px);border-color:rgba(96,165,250,.34);
  background:linear-gradient(145deg,rgba(37,99,235,.14),rgba(15,23,42,.42));}
.aee-project-feature-index{display:block;margin-bottom:18px;color:#7dd3fc;
  font:700 .68rem 'JetBrains Mono',monospace;letter-spacing:.08em;}
.aee-about-note{display:flex;align-items:flex-start;gap:13px;margin-top:22px;padding:17px 19px;
  border:1px solid rgba(16,185,129,.2);border-radius:16px;background:rgba(16,185,129,.055);}
.aee-about-note-mark{display:grid;place-items:center;flex:none;width:27px;height:27px;border-radius:9px;
  color:#6ee7b7;background:rgba(16,185,129,.12);font-weight:800;}
.aee-about-note strong{display:block;color:var(--text);font-size:.82rem;font-weight:750;}
.aee-about-note span{display:block;margin-top:4px;color:var(--text-muted);font-size:.78rem;line-height:1.6;}
.aee-contact-link{display:inline-flex;align-items:center;justify-content:center;gap:10px;margin-top:25px;padding:13px 18px;
  border:1px solid rgba(191,219,254,.2);border-radius:13px;color:#fff!important;text-decoration:none!important;font-weight:750;
  background:linear-gradient(120deg,#2563eb,#4f46e5);box-shadow:0 14px 34px -19px rgba(59,130,246,.9);
  transition:transform .18s,box-shadow .18s,filter .18s;}
.aee-contact-link:hover{transform:translateY(-2px);filter:brightness(1.08);box-shadow:0 18px 40px -18px rgba(59,130,246,.75);}
@media(max-width:900px){
  .aee-settings-fields{grid-template-columns:repeat(2,minmax(0,1fr));}
  .aee-project-feature-grid{grid-template-columns:repeat(2,minmax(0,1fr));}
  .aee-update-steps{grid-template-columns:1fr;}
  .aee-project-feature{min-height:118px;}
}
@media(max-width:640px){
  .aee-utility-window-shell .column.wrap{width:100vw!important;max-width:100vw!important;
    padding:0 16px 28px!important;}
  .aee-utility-window-header{margin:0 -16px 22px;padding:0 16px;min-height:72px;gap:10px;}
  .aee-utility-window-shell .aee-project-page-heading{gap:10px;}
  .aee-utility-window-shell .aee-project-page-mark{width:40px;height:40px;border-radius:12px;}
  .aee-utility-window-shell .aee-project-page-mark svg{width:19px;height:19px;}
  .aee-utility-window-shell .aee-project-page-heading small{margin-bottom:2px;font-size:.53rem;}
  .aee-utility-window-shell .aee-project-page-heading h1{font-size:1.13rem;}
  .aee-utility-window-shell .aee-project-page-heading p{max-width:230px;font-size:.65rem;}
  .aee-utility-window-close{min-width:92px!important;height:38px!important;padding:0 9px!important;
    border-radius:9px!important;font-size:.72rem!important;}
  .aee-project-page-mark{width:46px;height:46px;border-radius:14px;}
  .aee-update-actions{flex-wrap:wrap!important;}
  .aee-update-actions>div{flex:1 1 100%!important;}
  .aee-account-auth-layout{margin:16px auto 22px;}
  .aee-account-shell .column.wrap>.aee-account-auth-form{width:min(480px,calc(100vw - 28px))!important;
    margin:14px auto 20px;padding:34px 20px 26px;border-radius:23px;}
  .aee-account-intro{margin-bottom:34px;}
  .aee-account-auth-form .aee-account-actions{margin-top:34px;}
  body.aee-access-required #aee-account-window .column.wrap{padding:14px!important;}
  .aee-settings-layout{display:flex!important;flex-direction:column!important;gap:12px!important;}
  .aee-settings-layout>.aee-settings-sidebar{position:static;flex:0 0 auto!important;
    width:100%!important;max-width:none!important;}
  .aee-settings-nav{display:flex;flex-wrap:wrap;gap:6px;padding:8px;overflow:visible;}
  .aee-settings-nav-label{display:none;}
  .aee-settings-nav button{width:auto;min-width:max-content;padding:9px 11px;}
  .aee-settings-layout>.aee-settings-main{width:100%!important;flex:1 1 auto!important;}
  .aee-account-shell .aee-account-profile{min-height:138px;align-items:flex-start;
    padding:16px 100px 16px 15px;gap:11px;border-radius:16px;}
  .aee-account-shell .aee-account-profile:before{top:17px;bottom:17px;}
  .aee-account-shell .aee-account-avatar{width:50px;height:50px;border-radius:16px;}
  .aee-account-shell .aee-account-identity{padding-top:1px;}
  .aee-account-shell .aee-account-identity strong{font-size:.9rem;}
  .aee-account-shell .aee-account-identity .aee-account-detail{font-size:.68rem;}
  .aee-account-shell .aee-account-identity .aee-account-joined{font-size:.64rem;}
  .aee-account-shell .aee-account-eyebrow{font-size:.53rem!important;}
  .aee-account-shell .aee-account-mode{margin-top:7px;padding:4px 7px;font-size:.55rem;}
  .aee-settings-card{padding:15px;}
  .aee-settings-profile-header{gap:10px!important;}
  .aee-settings-profile-header>#aee-account-profile-source{
    flex:1 1 0!important;width:auto!important;min-width:0!important;}
  .aee-account-shell .aee-account-profile{padding-right:104px;}
  .aee-settings-photo-actions{flex:0 0 88px!important;width:88px!important;right:9px;top:50%;
    bottom:auto;transform:translateY(-50%);gap:7px!important;}
  .aee-settings-photo-button button{min-height:36px!important;padding:0 6px!important;font-size:.66rem!important;}
  .aee-photo-editor-header{height:54px;padding:0 12px;}
  .aee-photo-editor-heading{gap:11px;}
  .aee-photo-editor-title{font-size:16px;}
  .aee-photo-editor-stage{padding:10px;}
  .aee-photo-crop-canvas{height:min(70vh,560px);min-height:220px;}
  .aee-photo-editor-zoom{height:58px;padding:0 12px;gap:9px;}
  .aee-photo-zoom-range{width:min(58vw,300px);}
  .aee-settings-photo-preview{width:58px!important;min-width:58px!important;max-width:58px!important;}
  .aee-settings-photo-preview .image-container{min-height:56px!important;max-height:56px!important;}
  .aee-settings-fields{grid-template-columns:1fr;gap:8px!important;}
  .aee-settings-profile-photo{max-width:none;}
  .aee-about-card,.aee-contact-card{min-height:0;padding:30px 23px;border-radius:20px;}
  .aee-project-feature-grid{grid-template-columns:1fr;}
  .aee-project-feature{min-height:0;}
  .aee-project-feature-index{margin-bottom:10px;}
  .aee-about-note{padding:15px;}
}
html[data-theme="light"] .aee-nav-toggle{color:#1e293b;background:linear-gradient(145deg,#fff,#eff4fb);}
html[data-theme="light"] .aee-nav-drawer{background:linear-gradient(180deg,#fff,#f3f6fb);}
html[data-theme="light"] .aee-nav-brand strong{color:#0f172a;}
html[data-theme="light"] .aee-nav-item{color:#334155;}
html[data-theme="light"] .aee-nav-item:hover{color:#0f172a;background:rgba(59,130,246,.08);}
html[data-theme="light"] .aee-about-card,html[data-theme="light"] .aee-contact-card{background:radial-gradient(480px 240px at 100% 0%,rgba(59,130,246,.08),transparent 70%),linear-gradient(150deg,#fff,#f4f7fc);}
html[data-theme="light"] .aee-project-feature{background:linear-gradient(145deg,rgba(255,255,255,.9),rgba(241,245,249,.8));border-color:rgba(100,116,139,.16);}
html[data-theme="light"] .aee-update-step{background:rgba(255,255,255,.62);border-color:rgba(100,116,139,.16);}
html[data-theme="light"] .aee-project-feature:hover{background:linear-gradient(145deg,rgba(219,234,254,.8),rgba(255,255,255,.9));}
html[data-theme="light"] .aee-about-note{background:rgba(16,185,129,.045);}
@media(prefers-reduced-motion:reduce){
  .aee-utility-window-shell,.aee-utility-window-shell .column.wrap,
  .aee-project-feature,.aee-contact-link{transition:none;}
  .aee-utility-window-shell .column.wrap{transform:none;}
  .aee-project-feature:hover,.aee-contact-link:hover{transform:none;}
}
.aee-tabs [role="tablist"]{
  gap:4px!important;padding:5px!important;
  background:linear-gradient(180deg,rgba(15,23,42,.75),rgba(9,12,22,.85))!important;
  border:1px solid var(--border)!important;border-radius:14px!important;
  display:inline-flex!important;margin:24px 0 18px!important;flex-wrap:wrap!important;
  backdrop-filter:blur(10px);
}
.aee-tabs [role="tab"]{
  padding:10px 20px!important;border-radius:10px!important;
  font-size:.86rem!important;font-weight:600!important;
  color:var(--text-muted)!important;border:none!important;
  background:transparent!important;
  transition:all .2s cubic-bezier(.4,0,.2,1)!important;
}
.aee-tabs [role="tab"]:hover{color:var(--text)!important;}
.aee-tabs [role="tab"][aria-selected="true"]{
  background:linear-gradient(135deg,rgba(59,130,246,.25),rgba(139,92,246,.25))!important;
  color:var(--text)!important;
  box-shadow:0 6px 18px -10px rgba(59,130,246,.55),0 0 0 1px rgba(96,165,250,.25) inset;
}
.aee-stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
  gap:12px;margin:20px 0;}
.aee-stat{
  background:linear-gradient(180deg,rgba(20,27,45,.7),rgba(15,23,42,.82));
  border:1px solid var(--border);border-radius:14px;padding:18px;
  transition:all .2s ease;
}
.aee-stat:hover{transform:translateY(-2px);border-color:var(--border-strong);}
.aee-stat-icon{font-size:1.1rem;display:block;margin-bottom:10px;}
.aee-stat-val{
  font-size:1.75rem;font-weight:800;letter-spacing:-.025em;line-height:1;
  background:linear-gradient(135deg,#f8fafc,#c7d2fe);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
html[data-theme="light"] .aee-stat-val{
  background:linear-gradient(135deg,#0f172a,#4c1d95);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
.aee-stat-lbl{
  font-size:.7rem;color:var(--text-subtle);letter-spacing:.1em;
  text-transform:uppercase;margin-top:8px;font-weight:600;
}
.aee-empty{
  text-align:center;padding:60px 24px;border-radius:18px;
  background:var(--surface-2);border:1px dashed var(--border-strong);
}
.aee-empty-icon{font-size:2.4rem;margin-bottom:16px;opacity:.9;}
.aee-empty-title{font-size:1.1rem;font-weight:700;color:var(--text);margin-bottom:8px;}
.aee-empty-sub{font-size:.88rem;color:var(--text-muted);margin-bottom:28px;}
.aee-class-chip{
  display:inline-block;padding:3px 10px;border-radius:6px;
  background:linear-gradient(135deg,rgba(139,92,246,.12),rgba(139,92,246,.06));
  border:1px solid rgba(139,92,246,.28);
  font-size:.68rem;color:#c4b5fd;font-family:'JetBrains Mono',monospace;margin:2px;
}

.aee-upload-shell{
  background:
    radial-gradient(900px 240px at 50% -8%, rgba(59,130,246,.12), transparent 65%),
    linear-gradient(180deg, var(--shell-bg-1) 0%, var(--shell-bg-2) 100%) !important;
  border: 1px solid rgba(148,163,184,.11) !important;
  border-radius: 24px !important;
  padding: 22px !important;
  position: relative !important;
  overflow: visible !important;
  isolation: isolate;
  gap: 0 !important;
  box-shadow:
    0 1px 0 rgba(255,255,255,.06) inset,
    0 40px 100px -36px rgba(2,6,23,.98),
    0 0 140px -60px rgba(59,130,246,.45) !important;
  backdrop-filter: blur(14px);
}
.aee-upload-shell::before{
  content:'';position:absolute;inset:0;border-radius:24px;padding:1px;
  background:linear-gradient(155deg,
    rgba(96,165,250,.65) 0%, rgba(139,92,246,.22) 25%,
    transparent 55%, rgba(34,211,238,.5) 100%);
  -webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);
  -webkit-mask-composite:xor;mask-composite:exclude;
  pointer-events:none;z-index:0;
}
.aee-upload-shell::after{
  content:'';position:absolute;top:-200px;left:-140px;
  width:480px;height:480px;
  background:radial-gradient(circle,rgba(59,130,246,.22),transparent 62%);
  pointer-events:none;z-index:0;filter:blur(40px);
}
.aee-upload-shell>*{position:relative;z-index:2;}
#aee-upload-progress-host{
  display:block !important;width:100% !important;min-width:0 !important;
  margin:0 !important;padding:0 !important;
}
#aee-upload-progress-host .aee-up-progress{
  width:100%;box-sizing:border-box;
}

.aee-up-head{
  display:flex;align-items:center;gap:16px;
  padding-bottom:20px;margin-bottom:20px;
  border-bottom:1px solid rgba(148,163,184,.09);
  position:relative;
}
.aee-up-head::after{
  content:'';position:absolute;left:0;bottom:-1px;
  width:100px;height:1px;
  background:linear-gradient(90deg,#3b82f6,#8b5cf6,rgba(34,211,238,.6),transparent);
}
.aee-up-icon{
  width:52px;height:52px;flex-shrink:0;
  border-radius:15px;
  background:linear-gradient(140deg,#3b82f6 0%,#6366f1 55%,#8b5cf6 100%);
  box-shadow:
    0 16px 34px -14px rgba(59,130,246,.75),
    0 0 0 1px rgba(255,255,255,.18) inset,
    0 1px 0 rgba(255,255,255,.42) inset;
  display:flex;align-items:center;justify-content:center;
  position:relative;
}
.aee-up-icon svg{width:24px;height:24px;color:#fff;position:relative;z-index:2;}
.aee-up-icon::before{
  content:'';position:absolute;inset:-10px;border-radius:22px;
  background:radial-gradient(circle,rgba(96,165,250,.55),transparent 70%);
  filter:blur(14px);opacity:.65;z-index:0;
  animation:aee-icon-glow 3.6s ease-in-out infinite;
}
@keyframes aee-icon-glow{
  0%,100%{opacity:.5;transform:scale(.94);}
  50%    {opacity:.9;transform:scale(1.08);}
}
.aee-up-text{flex:1;min-width:0;}
.aee-up-title{
  font-size:1.15rem;font-weight:800;letter-spacing:-.025em;
  color:var(--text);line-height:1.2;margin-bottom:5px;
  display:flex;align-items:center;gap:10px;flex-wrap:wrap;
}
.aee-up-pill{
  font-size:.6rem;font-weight:700;padding:3px 10px;border-radius:100px;
  background:linear-gradient(135deg,rgba(59,130,246,.18),rgba(139,92,246,.16));
  border:1px solid rgba(96,165,250,.36);
  color:#93c5fd;letter-spacing:.14em;text-transform:uppercase;
  font-family:'JetBrains Mono',monospace;
}
html[data-theme="light"] .aee-up-pill{color:#1d4ed8;}
.aee-up-sub{font-size:.82rem;color:var(--text-muted);line-height:1.55;}
.aee-up-status{
  padding:7px 14px;border-radius:100px;
  background:linear-gradient(135deg,rgba(16,185,129,.14),rgba(16,185,129,.06));
  border:1px solid rgba(16,185,129,.4);
  color:#6ee7b7;font-size:.62rem;font-weight:800;
  letter-spacing:.16em;text-transform:uppercase;
  font-family:'JetBrains Mono',monospace;
  display:inline-flex;align-items:center;gap:7px;
  flex-shrink:0;
  box-shadow:0 8px 20px -12px rgba(16,185,129,.55);
}
.aee-up-status-dot{
  width:6px;height:6px;border-radius:50%;background:#10b981;
  box-shadow:0 0 10px #10b981;
  animation:pulse-dot 1.8s ease-in-out infinite;
}
@keyframes pulse-dot{
  0%,100%{opacity:1;transform:scale(1);}
  50%    {opacity:.5;transform:scale(1.25);}
}

.aee-upload-trigger,
.aee-upload-trigger button,
button.aee-upload-trigger {
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 10px !important;
  width: 100% !important;
  min-width: 0 !important;
  max-width: 100% !important;
  padding: 14px 26px !important;
  margin: 0 !important;
  background: linear-gradient(135deg, #3b82f6 0%, #6366f1 45%, #8b5cf6 100%) !important;
  background-size: 200% 200% !important;
  color: #fff !important;
  border: none !important;
  border-radius: 14px !important;
  font-family: 'Inter', sans-serif !important;
  font-size: .95rem !important;
  font-weight: 700 !important;
  letter-spacing: -.005em !important;
  cursor: pointer !important;
  position: relative !important;
  overflow: hidden !important;
  box-shadow:
    0 16px 34px -14px rgba(59,130,246,.85),
    0 0 0 1px rgba(255,255,255,.14) inset,
    0 1px 0 rgba(255,255,255,.34) inset !important;
  transition: all .28s cubic-bezier(.4,0,.2,1) !important;
  height: auto !important;
  min-height: 0 !important;
  line-height: 1.3 !important;
  text-transform: none !important;
  white-space: nowrap !important;
  outline: none !important;
  isolation: isolate;
}
.aee-upload-trigger svg,
.aee-upload-trigger button svg {
  width: 18px !important;height: 18px !important;flex-shrink: 0 !important;
}
.aee-upload-trigger::before,
.aee-upload-trigger button::before {
  content: '';
  position: absolute;inset: 0;
  background: linear-gradient(120deg,
    transparent 30%, rgba(255,255,255,.28) 50%, transparent 70%);
  transform: translateX(-100%);
  transition: transform .8s ease;
  pointer-events: none;z-index: 1;
}
.aee-upload-trigger:hover,
.aee-upload-trigger button:hover,
button.aee-upload-trigger:hover {
  background-position: 100% 0% !important;
  transform: translateY(-2px) !important;
  box-shadow:
    0 22px 46px -14px rgba(59,130,246,1),
    0 0 0 1px rgba(255,255,255,.2) inset,
    0 1px 0 rgba(255,255,255,.42) inset !important;
}
.aee-upload-trigger:hover::before,
.aee-upload-trigger button:hover::before { transform: translateX(100%); }
.aee-upload-trigger:active,
.aee-upload-trigger button:active,
button.aee-upload-trigger:active { transform: translateY(0) !important; }
.aee-upload-trigger > div,
.aee-upload-trigger .block,
.aee-upload-trigger .form,
.aee-upload-trigger .panel {
  background: transparent !important;border: none !important;
  padding: 0 !important;margin: 0 !important;box-shadow: none !important;
  min-height: 0 !important;gap: 0 !important;width: 100% !important;
}

/* ---------- Upload progress panel ---------- */
.aee-up-progress{
  display: none;
  margin-top: 14px;
  padding: 15px 18px;
  background:linear-gradient(135deg,rgba(15,23,42,.92),rgba(9,12,22,.96));
  border:1px solid rgba(96,165,250,.45);
  border-radius:12px;
  position:relative;
  overflow:hidden;
  box-shadow: 0 12px 32px -18px rgba(0,0,0,.9),
              0 0 0 1px rgba(255,255,255,.03) inset;
  animation:aee-fade-up .25s cubic-bezier(.4,0,.2,1);
  z-index: 5;
}
.aee-up-progress.aee-visible{display:block !important;}
.aee-up-progress.active{display:block !important;}
.aee-up-progress.complete{border-color:rgba(16,185,129,.5);}
.aee-up-progress.error{border-color:rgba(248,113,113,.6);
  background:linear-gradient(135deg,rgba(69,10,10,.9),rgba(24,8,14,.96));}
@keyframes aee-fade-up{
  from{opacity:0;transform:translateY(-4px);}
  to  {opacity:1;transform:translateY(0);}
}
.aee-up-progress-head{
  display:flex;align-items:center;justify-content:space-between;
  gap:12px;margin-bottom:10px;
}
.aee-up-progress-label{
  display:flex;align-items:center;gap:10px;
  font-size:.82rem;font-weight:600;color:#e6edf7;
  font-family:'Inter',sans-serif;
  min-width:0;
}
.aee-up-progress-text{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.aee-up-progress-spinner{
  width:14px;height:14px;
  border:2px solid rgba(96,165,250,.25);
  border-top-color:#60a5fa;
  border-radius:50%;
  animation:aee-spin .8s linear infinite;
  flex-shrink:0;
}
.aee-up-progress.complete .aee-up-progress-spinner,
.aee-up-progress.error .aee-up-progress-spinner{
  animation:none;border:0;width:16px;height:16px;
  display:flex;align-items:center;justify-content:center;
}
.aee-up-progress.complete .aee-up-progress-spinner::after{
  content:'✓';color:#34d399;font-size:16px;font-weight:900;
}
.aee-up-progress.error .aee-up-progress-spinner::after{
  content:'!';color:#fca5a5;font-size:15px;font-weight:900;
}
@keyframes aee-spin{to{transform:rotate(360deg);}}
.aee-up-progress-size{
  font-size:.72rem;color:#a8b5c9;
  font-family:'JetBrains Mono',monospace;
  font-weight:600;
  text-align:right;white-space:nowrap;
}
.aee-up-progress-track{
  height:8px;
  background:rgba(0,0,0,.6);
  border-radius:100px;
  overflow:hidden;
  border:1px solid var(--border);
  position:relative;
}
.aee-up-progress-bar{
  height:100%;
  border-radius:100px;
  background:linear-gradient(90deg,#3b82f6,#8b5cf6,#22d3ee,#3b82f6);
  background-size:250% 100%;
  animation:aee-stripe 1.4s linear infinite;
  width:0%;
  transition:width .25s ease;
  box-shadow:0 0 12px rgba(96,165,250,.7);
}
.aee-up-progress.indeterminate .aee-up-progress-bar{
  width:34% !important;
  animation:aee-progress-indeterminate 1.25s ease-in-out infinite;
}
.aee-up-progress.complete .aee-up-progress-bar{
  background:linear-gradient(90deg,#10b981,#34d399);
  animation:none;box-shadow:0 0 12px rgba(16,185,129,.55);
}
.aee-up-progress.error .aee-up-progress-bar{
  background:linear-gradient(90deg,#ef4444,#fb7185);
  animation:none;box-shadow:0 0 12px rgba(239,68,68,.5);
}
@keyframes aee-progress-indeterminate{
  0%{transform:translateX(-120%);}
  100%{transform:translateX(320%);}
}
@keyframes aee-stripe{
  0%   {background-position:0% 50%;}
  100% {background-position:250% 50%;}
}
.aee-up-progress-hint{
  margin-top:8px;
  font-size:.7rem;color:#8b9bb4;
  font-family:'JetBrains Mono',monospace;
  letter-spacing:.02em;
  display:flex;align-items:center;justify-content:space-between;gap:10px;
}
.aee-up-progress-pct{color:#93c5fd;font-weight:700;white-space:nowrap;}
.aee-up-progress-detail{
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;text-align:right;
}
.aee-up-progress.complete .aee-up-progress-pct{color:#6ee7b7;}
.aee-up-progress.error .aee-up-progress-pct{color:#fca5a5;}
html[data-theme="light"] .aee-up-progress{
  background:linear-gradient(135deg,rgba(248,250,252,.98),rgba(239,246,255,.98));
  border-color:rgba(59,130,246,.35);
}
html[data-theme="light"] .aee-up-progress-label{color:#0f172a;}
html[data-theme="light"] .aee-up-progress-size,
html[data-theme="light"] .aee-up-progress-hint{color:#475569;}
html[data-theme="light"] .aee-up-progress-track{background:rgba(148,163,184,.24);}
html[data-theme="light"] .aee-up-progress.error{
  background:linear-gradient(135deg,#fff7f7,#fff1f2);
  border-color:rgba(239,68,68,.45);
}
@media(max-width:520px){
  .aee-up-progress{padding:13px 14px;}
  .aee-up-progress-hint{align-items:flex-start;flex-direction:column;gap:4px;}
  .aee-up-progress-detail{text-align:left;max-width:100%;}
}

.aee-formats{
  margin-top: 14px;
  padding: 12px 16px;
  background:linear-gradient(135deg,rgba(15,23,42,.42),rgba(9,12,22,.52));
  border:1px solid var(--border);
  border-radius:12px;
}
.aee-formats-title{
  font-size:.66rem;font-weight:700;color:var(--text-subtle);
  letter-spacing:.14em;text-transform:uppercase;
  margin-bottom:8px;
  display:flex;align-items:center;gap:6px;
  font-family:'JetBrains Mono',monospace;
}
.aee-formats-list{display:flex;flex-wrap:wrap;gap:5px;}
.aee-format-chip{
  display:inline-block;
  padding:3px 9px;
  border-radius:6px;
  font-size:.66rem;font-weight:600;
  color:#a5b4fc;
  background:linear-gradient(135deg,rgba(96,165,250,.14),rgba(139,92,246,.10));
  border:1px solid rgba(96,165,250,.22);
  font-family:'JetBrains Mono',monospace;
  letter-spacing:.02em;
  transition:all .18s ease;
}
.aee-format-chip:hover{
  color:#dbeafe;
  border-color:rgba(96,165,250,.5);
  background:linear-gradient(135deg,rgba(96,165,250,.22),rgba(139,92,246,.16));
}
.aee-format-chip.more{
  color:#6ee7b7;
  background:linear-gradient(135deg,rgba(16,185,129,.14),rgba(16,185,129,.08));
  border-color:rgba(16,185,129,.3);
}
.aee-formats-note{
  margin-top:8px;
  font-size:.68rem;color:var(--text-subtle);
  font-family:'JetBrains Mono',monospace;
  display:flex;align-items:center;gap:6px;
}

.aee-actions{
  display:flex !important;align-items:center !important;
  gap:10px !important;margin-top:16px !important;
  padding-top:16px !important;
  border-top:1px solid rgba(148,163,184,.09) !important;
  position:relative;
}
.aee-actions::before{
  content:'';position:absolute;top:-1px;left:0;
  width:72px;height:1px;
  background:linear-gradient(90deg,rgba(139,92,246,.95),rgba(139,92,246,.3),transparent);
}
.aee-btn-reset{
  flex:0 0 auto !important;min-width:110px !important;
  background:linear-gradient(180deg,rgba(148,163,184,.07),rgba(148,163,184,.04)) !important;
  border:1px solid rgba(148,163,184,.20) !important;
  color:var(--text-muted) !important;
  border-radius:12px !important;
  font-size:.86rem !important;font-weight:600 !important;
  letter-spacing:-.005em !important;
  padding:13px 22px !important;
  box-shadow:0 1px 0 rgba(255,255,255,.03) inset !important;
  transition:all .22s cubic-bezier(.4,0,.2,1) !important;
}
.aee-btn-reset:hover{
  background:linear-gradient(180deg,rgba(148,163,184,.14),rgba(148,163,184,.08)) !important;
  border-color:rgba(96,165,250,.55) !important;
  color:var(--text) !important;
  transform:translateY(-1px) !important;
  box-shadow:0 12px 26px -14px rgba(96,165,250,.55) !important;
}
.aee-btn-run{
  flex:1 1 auto !important;min-width:0 !important;
  background:linear-gradient(135deg,#2563eb 0%,#4f46e5 55%,#6d28d9 100%) !important;
  background-size:200% 200% !important;
  border:none !important;color:#fff !important;
  border-radius:12px !important;
  font-size:.96rem !important;font-weight:800 !important;
  letter-spacing:-.005em !important;
  padding:14px 28px !important;
  position:relative !important;overflow:hidden !important;
  box-shadow:
    0 18px 38px -14px rgba(37,99,235,.85),
    0 0 0 1px rgba(255,255,255,.14) inset,
    0 1px 0 rgba(255,255,255,.32) inset !important;
  transition:all .28s cubic-bezier(.4,0,.2,1) !important;
}
.aee-btn-run::before{
  content:'';position:absolute;inset:0;
  background:linear-gradient(120deg,
    transparent 30%, rgba(255,255,255,.26) 50%, transparent 70%);
  transform:translateX(-100%);
  transition:transform .8s ease;pointer-events:none;
}
.aee-btn-run:hover{
  background-position:100% 0% !important;
  transform:translateY(-2px) !important;
  box-shadow:
    0 24px 50px -14px rgba(37,99,235,1),
    0 0 0 1px rgba(255,255,255,.2) inset,
    0 1px 0 rgba(255,255,255,.42) inset !important;
}
.aee-btn-run:hover::before{transform:translateX(100%);}
.aee-btn-run:active{transform:translateY(0) !important;}

.aee-file-summary{margin-top:16px;}
.aee-file-stats{
  display:flex;align-items:center;flex-wrap:wrap;gap:8px 14px;
  padding:13px 18px;
  background:linear-gradient(90deg,
    rgba(59,130,246,.10), rgba(139,92,246,.08) 55%, rgba(34,211,238,.06));
  border:1px solid rgba(96,165,250,.22);
  border-radius:12px;
  font-family:'JetBrains Mono',monospace;
  font-size:.78rem;color:var(--text-muted);
  position:relative;overflow:hidden;
  box-shadow:0 1px 0 rgba(255,255,255,.04) inset,0 8px 20px -16px rgba(59,130,246,.55);
}
.aee-file-stats::before{
  content:'';position:absolute;inset:0;
  background:linear-gradient(90deg,
    transparent,rgba(255,255,255,.05) 50%,transparent);
  animation:aee-sheen 4s ease-in-out infinite;
  pointer-events:none;
}
@keyframes aee-sheen{
  0%,100%{transform:translateX(-100%);}
  50%    {transform:translateX(100%);}
}
.aee-file-stats b{color:var(--text);font-weight:700;}
.aee-file-stats-sep{color:var(--text-subtle);opacity:.5;}
.aee-file-stat-ok{
  color:#6ee7b7;font-weight:700;
  text-shadow:0 0 10px rgba(16,185,129,.55);
}
.aee-file-stat-warn{color:#fbbf24;font-weight:700;}

html[data-theme="light"] .aee-upload-shell{
  background: linear-gradient(180deg,#ffffff,#eef2f8) !important;
}

@media (max-width:720px){
  .aee-up-status{display:none;}
  .aee-upload-shell{padding:18px !important;}
}

.gradio-container .block,
.gradio-container .form,
.gradio-container .panel{
  background:transparent!important;
  border:none!important;
  box-shadow:none!important;
  padding:0!important;
}
.gradio-container .gr-dataframe{
  border-radius:14px!important;overflow:hidden!important;
  border:1px solid var(--border)!important;background:var(--surface-2)!important;
}
.gradio-container .gr-dataframe thead th{
  background:var(--surface-3)!important;color:#a5b4fc!important;
  font-size:.7rem!important;text-transform:uppercase!important;
  letter-spacing:.1em!important;padding:12px 16px!important;font-weight:700!important;
}
.gradio-container .gallery{
  border-radius:14px!important;background:var(--surface-2)!important;
  border:1px solid var(--border)!important;padding:12px!important;
}
.gradio-container .gallery img{
  border-radius:10px!important;border:1px solid var(--border-strong)!important;
  transition:transform .2s ease,box-shadow .2s ease,border-color .2s ease!important;
}
.gradio-container .gallery img:hover{
  transform:scale(1.03)!important;border-color:rgba(96,165,250,.75)!important;
  box-shadow:0 14px 40px rgba(59,130,246,.5)!important;
}

.aee-load{
  padding:26px;border-radius:18px;
  background:linear-gradient(180deg,rgba(15,23,42,.88),rgba(10,14,26,.92));
  border:1px solid rgba(96,165,250,.32);
  box-shadow:0 30px 70px -34px rgba(2,6,23,.98);
}
.aee-load-top{
  display:grid;grid-template-columns:140px 1fr;gap:24px;
  align-items:center;margin-bottom:22px;
}
.aee-ring-wrap{position:relative;width:140px;height:140px;}
.aee-ring-svg{transform:rotate(-90deg);width:140px;height:140px;}
.aee-ring-bg{fill:none;stroke:var(--surface-3);stroke-width:10;}
.aee-ring-fg{
  fill:none;stroke:url(#aee-grad);stroke-width:10;stroke-linecap:round;
  transition:stroke-dashoffset .5s ease;
  filter:drop-shadow(0 0 10px rgba(96,165,250,.55));
}
.aee-ring-center{
  position:absolute;inset:0;display:flex;flex-direction:column;
  align-items:center;justify-content:center;
}
.aee-ring-pct{font-size:1.9rem;font-weight:900;letter-spacing:-.045em;color:var(--text);}
.aee-ring-lbl{
  font-size:.62rem;color:var(--text-subtle);letter-spacing:.14em;
  text-transform:uppercase;margin-top:6px;font-weight:700;
}
.aee-load-video{
  font-family:'JetBrains Mono',monospace;font-size:.82rem;color:#93c5fd;
  display:flex;align-items:center;gap:8px;
}
.aee-load-video-idx{
  color:var(--text-subtle);font-size:.72rem;padding:2px 8px;
  background:var(--surface-3);border-radius:6px;font-weight:700;
}
.aee-load-stage{font-size:1rem;font-weight:700;color:var(--text);margin-top:8px;}
.aee-load-msg{font-size:.86rem;color:var(--text-muted);margin-top:4px;}
.aee-bars{display:flex;flex-direction:column;gap:10px;margin-bottom:16px;}
.aee-bar-lbl{
  display:flex;justify-content:space-between;font-size:.72rem;
  color:var(--text-subtle);margin-bottom:6px;font-weight:600;
  letter-spacing:.06em;text-transform:uppercase;
  font-family:'JetBrains Mono',monospace;
}
.aee-bar-lbl b{color:var(--text);}
.aee-bar-outer{
  height:12px;background:rgba(0,0,0,.5);border-radius:100px;
  overflow:hidden;border:1px solid var(--border);
}
.aee-bar-outer.tall{height:18px;}
.aee-bar-fill{
  height:100%;border-radius:100px;transition:width .4s;
  background:linear-gradient(90deg,#3b82f6,#8b5cf6,#22d3ee,#3b82f6);
  background-size:250% 100%;animation:shim 2s linear infinite;
  box-shadow:0 0 12px rgba(59,130,246,.5);
}
.aee-bar-fill.cyan{
  background:linear-gradient(90deg,#06b6d4,#0891b2,#06b6d4);
  background-size:200% 100%;
}
@keyframes shim{0%{background-position:0% 50%}100%{background-position:300% 50%}}
.aee-stages{display:grid;grid-template-columns:repeat(5,1fr);gap:6px;margin:18px 0;}
.aee-stage{
  padding:10px 8px;border-radius:10px;background:var(--surface-2);
  border:1px solid var(--border);text-align:center;
}
.aee-stage.done{background:rgba(16,185,129,.10);border-color:rgba(16,185,129,.4);}
.aee-stage.active{
  background:linear-gradient(135deg,rgba(59,130,246,.25),rgba(139,92,246,.25));
  border-color:rgba(96,165,250,.85);
  box-shadow:0 6px 20px -10px rgba(59,130,246,.6);
}
.aee-stage-icon{font-size:1rem;display:block;margin-bottom:5px;}
.aee-stage-lbl{
  font-size:.7rem;font-weight:700;color:var(--text-muted);
  text-transform:uppercase;font-family:'JetBrains Mono',monospace;
}
.aee-stage.done .aee-stage-lbl{color:#6ee7b7;}
.aee-stage.active .aee-stage-lbl{color:#dbeafe;}
.aee-substep{
  display:grid;grid-template-columns:repeat(4,1fr);gap:12px;padding:14px 18px;
  border-radius:12px;background:var(--surface-2);border:1px solid var(--border);
  margin-bottom:16px;
}
.aee-sub-item{display:flex;flex-direction:column;gap:4px;}
.aee-sub-lbl{
  font-size:.62rem;color:var(--text-subtle);letter-spacing:.1em;
  text-transform:uppercase;font-weight:700;
  font-family:'JetBrains Mono',monospace;
}
.aee-sub-val{
  font-size:.94rem;color:var(--text);font-weight:800;
  font-family:'JetBrains Mono',monospace;
}
.aee-sub-val.accent{color:var(--blue);}
.aee-sub-val.ok{color:var(--ok);}
.aee-infer{
  padding:14px 18px;border-radius:12px;background:var(--surface-2);
  border:1px solid var(--border);margin-bottom:16px;
}
.aee-infer-title{
  font-size:.72rem;color:var(--text-subtle);font-weight:700;
  letter-spacing:.1em;text-transform:uppercase;margin-bottom:12px;
}
.aee-infer-row{
  display:grid;grid-template-columns:160px 1fr 64px;gap:12px;
  align-items:center;padding:5px 0;font-size:.76rem;
  font-family:'JetBrains Mono',monospace;
}
.aee-infer-name{color:var(--text-muted);overflow:hidden;text-overflow:ellipsis;}
.aee-infer-bar{height:6px;background:var(--surface-3);border-radius:100px;overflow:hidden;}
.aee-infer-fill{height:100%;border-radius:100px;
  background:linear-gradient(90deg,#06b6d4,#22d3ee);}
.aee-infer-ms{color:var(--text);text-align:right;font-weight:600;}
.aee-ticker{
  margin-top:4px;padding:14px 18px;border-radius:12px;
  background:var(--surface-2);border:1px solid var(--border);
}
.aee-ticker-title{
  font-size:.72rem;color:var(--text-subtle);font-weight:700;
  letter-spacing:.1em;text-transform:uppercase;margin-bottom:10px;display:flex;gap:8px;
}
.aee-ticker-dot{
  width:6px;height:6px;border-radius:50%;background:var(--err);
  box-shadow:0 0 8px var(--err);animation:pulse-dot 1s infinite;
}
.aee-ticker-list{display:flex;flex-direction:column;gap:5px;max-height:180px;
  overflow-y:auto;}
.aee-tick{
  display:grid;grid-template-columns:80px 1fr 60px 110px;gap:12px;padding:7px 12px;
  background:rgba(0,0,0,.28);border-radius:8px;font-size:.78rem;align-items:center;
  border-left:2px solid var(--blue);
}
.aee-tick.wpn{border-left-color:var(--err);}
.aee-tick .t{color:var(--text-subtle);font-family:'JetBrains Mono',monospace;font-size:.72rem;}
.aee-tick .l{color:var(--text);font-weight:600;}
.aee-tick .c{color:var(--ok);font-family:'JetBrains Mono',monospace;font-weight:600;
  text-align:right;}
.aee-tick .s{color:var(--text-subtle);font-size:.7rem;text-align:right;
  font-family:'JetBrains Mono',monospace;overflow:hidden;text-overflow:ellipsis;}
.aee-footer{
  text-align:center;padding:56px 0 20px;margin-top:60px;
  border-top:1px solid var(--border);color:var(--text-subtle);font-size:.75rem;
}
.aee-footer b{color:var(--text-muted);}

.aee-track-stats{
  display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:12px;margin:8px 0 18px;
}
.aee-track-summary{
  display:flex;gap:20px;flex-wrap:wrap;padding:14px 18px;
  background:var(--surface-2);border:1px solid var(--border);border-radius:12px;
  font-size:.85rem;color:var(--text-muted);margin-bottom:18px;
}
.aee-track-summary b{color:var(--text);font-family:'JetBrains Mono',monospace;}
.aee-track-timeline{
  padding:22px 24px;border-radius:14px;
  background:var(--surface-2);border:1px solid var(--border);
}
.aee-track-timeline h4{
  font-size:.9rem;font-weight:600;color:var(--text);
  margin-bottom:16px;display:flex;justify-content:space-between;align-items:baseline;
}
.aee-track-timeline h4 span{
  font-size:.7rem;color:var(--text-subtle);font-weight:400;
  text-transform:uppercase;letter-spacing:.1em;
  font-family:'JetBrains Mono',monospace;
}
.aee-trackbar{
  display:grid;grid-template-columns:60px 90px 1fr 60px 50px;gap:10px;
  align-items:center;padding:6px 0;font-size:.78rem;
  font-family:'JetBrains Mono',monospace;
}
.aee-trackbar-id{color:var(--text-subtle);font-weight:600;}
.aee-trackbar-lbl{color:var(--text);font-weight:600;overflow:hidden;text-overflow:ellipsis;}
.aee-trackbar-outer{height:8px;background:var(--surface-3);border-radius:100px;
  overflow:hidden;}
.aee-trackbar-fill{height:100%;border-radius:100px;
  background:linear-gradient(90deg,#3b82f6,#8b5cf6);}
.aee-trackbar.wpn .aee-trackbar-fill{background:linear-gradient(90deg,#ef4444,#f59e0b);}
.aee-trackbar.ctx .aee-trackbar-fill{background:linear-gradient(90deg,#8b5cf6,#a78bfa);}
.aee-trackbar-dur,.aee-trackbar-frames{color:var(--text-muted);text-align:right;}
.aee-cam-stats{
  display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:12px;margin:8px 0 18px;
}
.aee-cam-sources{
  padding:14px 18px;border-radius:12px;background:var(--surface-2);
  border:1px solid var(--border);margin-bottom:18px;
}
.aee-cam-sources-title{
  font-size:.72rem;color:var(--text-subtle);font-weight:700;
  letter-spacing:.1em;text-transform:uppercase;margin-bottom:10px;
}

.aee-threat-card{
  padding:24px;border-radius:18px;
  background:linear-gradient(180deg,rgba(15,23,42,.75),rgba(10,14,26,.85));
  border:1px solid var(--border);margin:14px 0;
}
.aee-threat-title{
  font-size:.95rem;font-weight:700;color:var(--text);
  margin-bottom:16px;display:flex;justify-content:space-between;align-items:baseline;
}
.aee-threat-title span{
  font-size:.7rem;color:var(--text-subtle);font-weight:400;
  text-transform:uppercase;letter-spacing:.1em;
  font-family:'JetBrains Mono',monospace;
}
.aee-threat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;}
.aee-threat-cell{
  padding:14px;border-radius:12px;
  background:var(--surface-2);border:1px solid var(--border);
  text-align:left;position:relative;overflow:hidden;
}
.aee-threat-cell::before{
  content:'';position:absolute;left:0;top:0;bottom:0;width:3px;
  background:var(--lvl-color,#64748b);
}
.aee-threat-cell .lvl{
  font-size:.68rem;font-weight:800;letter-spacing:.14em;
  text-transform:uppercase;color:var(--lvl-color,#64748b);
  margin-bottom:6px;font-family:'JetBrains Mono',monospace;
}
.aee-threat-cell .num{
  font-size:1.5rem;font-weight:900;color:var(--text);
  letter-spacing:-.035em;line-height:1;
}
.aee-threat-cell .sub{font-size:.68rem;color:var(--text-subtle);margin-top:4px;}

.aee-toast-wrap{
  position:fixed;bottom:28px;right:28px;z-index:9999;
  display:flex;flex-direction:column;gap:10px;pointer-events:none;
}
.aee-toast{
  display:flex;align-items:center;gap:12px;padding:13px 20px;
  background:linear-gradient(135deg,rgba(15,23,42,.96),rgba(9,12,22,.98));
  border:1px solid rgba(96,165,250,.4);
  border-radius:14px;min-width:250px;
  box-shadow:0 24px 48px -20px rgba(2,6,23,.98),
             0 0 0 1px rgba(255,255,255,.05) inset;
  color:#e6edf7;font-size:.85rem;font-weight:600;
  transform:translateX(120%);opacity:0;
  transition:all .38s cubic-bezier(.4,0,.2,1);
  pointer-events:auto;font-family:'Inter',sans-serif;
  backdrop-filter:blur(12px);
}
.aee-toast.show{transform:translateX(0);opacity:1;}
.aee-toast .ico{font-size:1.1rem;}
.aee-toast.success{border-color:rgba(16,185,129,.5);}
.aee-toast.error{border-color:rgba(239,68,68,.6);}
.aee-toast.warn{border-color:rgba(245,158,11,.6);}

.aee-cmd{position:fixed;inset:0;z-index:9998;display:none;}
.aee-cmd.open{display:block;}
.aee-cmd-backdrop{
  position:absolute;inset:0;background:rgba(2,6,23,.72);
  backdrop-filter:blur(8px);
}
.aee-cmd-panel{
  position:relative;max-width:560px;margin:14vh auto 0;
  background:linear-gradient(180deg,#0d1425,#070b16);
  border:1px solid rgba(96,165,250,.4);border-radius:16px;
  box-shadow:0 48px 100px -22px rgba(2,6,23,.98),
             0 0 0 1px rgba(255,255,255,.05) inset;
  overflow:hidden;
}
.aee-cmd-input{
  width:100%;padding:18px 22px;background:transparent;border:none;
  color:#e6edf7;font-family:'Inter',sans-serif;font-size:1rem;font-weight:600;
  outline:none;border-bottom:1px solid rgba(148,163,184,.14);
}
.aee-cmd-input::placeholder{color:#64748b;font-weight:400;}
.aee-cmd-list{max-height:340px;overflow-y:auto;padding:8px;}
.aee-cmd-item{
  padding:12px 16px;border-radius:10px;cursor:pointer;
  display:flex;align-items:center;gap:12px;
  color:#cbd5e1;font-size:.9rem;font-weight:500;
  transition:all .14s ease;
}
.aee-cmd-item:hover,.aee-cmd-item.active{
  background:linear-gradient(135deg,rgba(59,130,246,.25),rgba(139,92,246,.2));
  color:#f1f5f9;
}
.aee-cmd-item .kbd{
  margin-left:auto;padding:2px 8px;background:rgba(148,163,184,.10);
  border:1px solid rgba(148,163,184,.18);border-radius:5px;
  font-family:'JetBrains Mono',monospace;font-size:.68rem;color:#94a3b8;
}
.aee-cmd-foot{
  padding:10px 18px;border-top:1px solid rgba(148,163,184,.09);
  display:flex;gap:16px;font-size:.72rem;color:#64748b;
  font-family:'JetBrains Mono',monospace;
}
.aee-cmd-foot .kbd{
  padding:1px 6px;background:rgba(148,163,184,.10);
  border:1px solid rgba(148,163,184,.18);border-radius:4px;
}

*{scrollbar-width:thin;scrollbar-color:#3b82f6 #0b1120;}
::-webkit-scrollbar{width:10px;height:10px;}
::-webkit-scrollbar-track{background:var(--surface-2);border-radius:999px;}
::-webkit-scrollbar-thumb{
  background:linear-gradient(180deg,#3b82f6,#7c3aed);
  border:2px solid var(--surface-2);border-radius:999px;
}
::-webkit-scrollbar-thumb:hover{background:linear-gradient(180deg,#60a5fa,#a78bfa);}

.aee-heat-overlay{
  position: absolute !important;
  left: 8px !important;
  right: 8px !important;
  bottom: 46px !important;
  height: 78px !important;
  z-index: 30 !important;
  pointer-events: none !important;
  background: linear-gradient(180deg,
    rgba(2,6,23,0.04) 0%,
    rgba(2,6,23,0.30) 42%,
    rgba(2,6,23,0.82) 100%);
  border:1px solid rgba(226,232,240,.12);
  border-radius: 10px;
  overflow: hidden;
  opacity: 0;
  transition: opacity .3s ease;
  box-shadow: 0 8px 26px rgba(0,0,0,.36),
              inset 0 1px 0 rgba(255,255,255,.08);
}
.aee-heat-overlay.active{opacity:1;}
.aee-heat-overlay canvas{display:block;width:100%;height:100%;pointer-events:none;}
.aee-heat-playhead{
  position: absolute;
  top:0; bottom:0;
  width:2px;
  background:#fff;
  box-shadow: 0 0 8px #22d3ee, 0 0 2px #fff;
  pointer-events:none;
  left:0;
  will-change:left;
}
.aee-heat-legend{
  position:absolute;
  left:8px; right:8px; bottom:4px;
  display:flex; flex-wrap:wrap; align-items:center; gap:4px 7px;
  max-height:52%; overflow:auto;
  font-size:.6rem;
  font-family:'JetBrains Mono',monospace;
  color:#e2e8f0;
  text-shadow:0 1px 3px rgba(0,0,0,.95);
  pointer-events:auto;
  line-height:1.2;
}
.aee-heat-legend span{
  display:inline-flex;align-items:center;gap:4px;
  max-width:150px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
}
.aee-heat-legend i{
  width:7px;height:7px;border-radius:50%;
  display:inline-block;flex-shrink:0;
}
.aee-heat-legend .aee-heat-title{
  color:#f1f5f9;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
}
.aee-heat-agreement{color:#a5f3fc;}
.aee-heat-model-chip{
  display:inline-flex;align-items:center;gap:4px;
  max-width:190px;padding:3px 7px;border-radius:999px;
  color:#e2e8f0;background:rgba(15,23,42,.76);
  border:1px solid rgba(226,232,240,.16);
  font:inherit;text-shadow:inherit;cursor:pointer;
  transition:opacity .16s ease,background .16s ease,border-color .16s ease;
}
.aee-heat-model-chip:hover{background:rgba(30,41,59,.94);border-color:rgba(226,232,240,.34);}
.aee-heat-model-chip[aria-pressed="false"]{opacity:.48;}
.aee-heat-model-chip small{color:#cbd5e1;font-size:.86em;}
.aee-heat-jump{
  pointer-events:auto !important;
  cursor:pointer;
  color:#22d3ee !important;
  font-weight:700;
  text-decoration:none;
  padding:2px 8px;
  border-radius:5px;
  background:rgba(34,211,238,.14);
  border:1px solid rgba(34,211,238,.45);
  transition:all .18s ease;
  text-shadow:none !important;
  max-width:none !important;
  margin-left:auto;
}
.aee-heat-jump:hover{
  background:rgba(34,211,238,.26);
  color:#a5f3fc !important;
  border-color:rgba(34,211,238,.7);
  transform:translateY(-1px);
  box-shadow:0 4px 12px -4px rgba(34,211,238,.6);
}
@media (max-width:720px){
  .aee-heat-overlay{height:64px;bottom:40px;left:6px;right:6px;}
  .aee-heat-legend{font-size:.52rem;gap:3px 5px;left:5px;right:5px;bottom:3px;}
  .aee-heat-model-chip{padding:2px 5px;}
  .aee-heat-jump{padding:1px 6px;}
}
@media(max-width:380px){
  .aee-toolbar{gap:4px!important;}
  .aee-toolbar .aee-tool{width:32px!important;height:32px!important;border-radius:10px!important;}
  .aee-toolbar .aee-update-app{width:auto!important;min-height:32px!important;padding:0 6px!important;gap:4px!important;
    font-size:.64rem!important;}
  .aee-toolbar .aee-update-app svg{width:14px!important;height:14px!important;}
}
html[dir="rtl"],html[dir="rtl"] body,html[dir="rtl"] .gradio-container{
  direction:rtl;font-family:Tahoma,Arial,sans-serif!important;}
html[dir="rtl"] .gradio-container [dir="ltr"]{direction:rtl!important;}
html[dir="rtl"] .gradio-container .aee-account-profile{direction:rtl;}
html[dir="rtl"] .aee-hero,html[dir="rtl"] .aee-project-page-heading,
html[dir="rtl"] .aee-history-heading,html[dir="rtl"] .aee-history-section-heading,
html[dir="rtl"] .aee-update-copy,html[dir="rtl"] .aee-settings-card,
html[dir="rtl"] .aee-settings-nav,html[dir="rtl"] .aee-account-shell,
html[dir="rtl"] .aee-history-window,html[dir="rtl"] .aee-update-window,
html[dir="rtl"] .aee-about-window,html[dir="rtl"] .aee-contact-window{
  text-align:right;}
html[dir="rtl"] .aee-nav-toggle{right:16px;left:auto;}
html[dir="rtl"] .aee-nav-drawer{right:0;left:auto;transform:translateX(105%);}
html[dir="rtl"] .aee-nav-open .aee-nav-drawer{transform:translateX(0);}
html[dir="rtl"] .aee-nav-item:hover{transform:translateX(-2px);}
html[dir="rtl"] .aee-nav-group-label,html[dir="rtl"] .aee-nav-foot{text-align:right;}
html[dir="rtl"] .aee-account-shell .aee-account-profile:before{right:0;left:auto;
  border-radius:4px 0 0 4px;}
html[dir="rtl"] .aee-account-shell .aee-account-profile{padding-right:22px;padding-left:170px;}
html[dir="rtl"] .aee-account-shell .aee-settings-photo-actions{right:auto!important;left:14px!important;}
@media(max-width:640px){
  html[dir="rtl"] .aee-account-shell .aee-account-profile{
    padding-right:15px!important;padding-left:104px!important;}
  html[dir="rtl"] .aee-account-shell .aee-settings-photo-actions{
    right:auto!important;left:9px!important;}
}
html[dir="rtl"] .aee-settings-nav button,html[dir="rtl"] .aee-settings-nav summary,
html[dir="rtl"] .aee-settings-language-menu button{text-align:right;}
html[dir="rtl"] .aee-history-metadata dt,html[dir="rtl"] .aee-history-metadata dd,
html[dir="rtl"] .aee-history-row-info,html[dir="rtl"] .aee-history-detail{
  text-align:right;}
html[dir="rtl"] .aee-update-actions,html[dir="rtl"] .aee-account-actions,
html[dir="rtl"] .aee-account-profile-actions{direction:rtl;}
html[dir="rtl"] .aee-update-progress-detail{text-align:left;}
html[dir="rtl"] .aee-account-shell input[type="email"],html[dir="rtl"] input[type="url"],
html[dir="rtl"] input[type="password"],html[dir="rtl"] input[type="search"],
html[dir="rtl"] input[type="number"],html[dir="rtl"] .aee-history-row-title,
html[dir="rtl"] .aee-history-metadata dd,html[dir="rtl"] code,html[dir="rtl"] pre,
html[dir="rtl"] .aee-version-pill{
  direction:ltr!important;unicode-bidi:plaintext;text-align:left;}
html[dir="rtl"] input[placeholder^="https://"],
html[dir="rtl"] input[placeholder^="sb_"]{direction:ltr!important;text-align:left;}
html[dir="rtl"] .aee-about-card:before,html[dir="rtl"] .aee-contact-card:before{
  right:0;left:auto;border-radius:4px 0 0 4px;}
html[dir="rtl"] .aee-progress-bar{transform-origin:right center;}
html[dir="rtl"] .aee-up-progress-detail{text-align:right;}
html[dir="rtl"] .aee-up-progress.indeterminate .aee-up-progress-bar{
  animation-name:aee-progress-indeterminate-rtl;}
@keyframes aee-progress-indeterminate-rtl{
  0%{transform:translateX(120%);}100%{transform:translateX(-320%);}}
html[dir="rtl"] .aee-heat-jump{margin-right:auto;margin-left:0;}
html[dir="rtl"] .aee-heat-model-chip{direction:rtl;}
html[dir="rtl"] .aee-settings-language-menu button[aria-pressed="true"]:after{
  right:auto;left:12px;}
html[dir="rtl"] .aee-formats-title,html[dir="rtl"] .aee-formats-note,
html[dir="rtl"] .aee-up-head,html[dir="rtl"] .aee-up-progress-hint{
  direction:rtl;}
html[dir="rtl"] .aee-account-divider:before,html[dir="rtl"] .aee-account-divider:after{
  background:linear-gradient(270deg,transparent,rgba(148,163,184,.28),transparent);}
html[dir="rtl"] .aee-account-divider{letter-spacing:0;}
html[dir="rtl"] .aee-account-eyebrow,html[dir="rtl"] .aee-settings-section-label,
html[dir="rtl"] .aee-stat-lbl,html[dir="rtl"] .aee-nav-group-label{
  letter-spacing:0;}
html[data-theme="light"] .gradio-container::before{
  background:
    radial-gradient(1100px 700px at 12% -12%,rgba(221,161,108,.2),transparent 55%),
    radial-gradient(900px 600px at 88% -4%,rgba(181,140,153,.18),transparent 55%),
    radial-gradient(1200px 800px at 50% 112%,rgba(165,141,145,.12),transparent 60%);
}
html[data-theme="light"] .gradio-container input,
html[data-theme="light"] .gradio-container textarea,
html[data-theme="light"] .gradio-container select{
  color:var(--text)!important;
  background:var(--input-background-fill)!important;
  border-color:var(--input-border-color)!important;
}
html[data-theme="light"] .gradio-container input::placeholder,
html[data-theme="light"] .gradio-container textarea::placeholder{
  color:var(--input-placeholder-color)!important;opacity:.78;
}
html[data-theme="light"] .gradio-container button.primary{
  color:var(--button-primary-text-color)!important;
  background:var(--button-primary-background-fill)!important;
  border-color:var(--button-primary-background-fill)!important;
}
html[data-theme="light"] .gradio-container button.primary:hover:not(:disabled){
  background:var(--button-primary-background-fill-hover)!important;
  border-color:var(--button-primary-background-fill-hover)!important;
}
html[data-theme="light"] .gradio-container button.secondary{
  color:var(--button-secondary-text-color)!important;
  background:var(--button-secondary-background-fill)!important;
  border-color:var(--border-strong)!important;
}
html[data-theme="light"] .gradio-container .block,
html[data-theme="light"] .gradio-container .form,
html[data-theme="light"] .gradio-container .panel,
html[data-theme="light"] .gradio-container .accordion,
html[data-theme="light"] .gradio-container .tabs,
html[data-theme="light"] .gradio-container .tabitem{
  color:var(--text);
}
html[data-theme="light"] .gradio-container .gr-dataframe thead th{
  color:var(--text)!important;background:var(--surface-2)!important;
}
html[data-theme="light"] .aee-utility-window-shell{
  background:
    radial-gradient(1000px 640px at 92% 0%,rgba(181,140,153,.2),transparent 70%),
    radial-gradient(1000px 650px at 0% 100%,rgba(221,161,108,.18),transparent 70%),
    linear-gradient(145deg,var(--shell-bg-1),var(--shell-bg-2) 76%)!important;
}
html[data-theme="light"] .aee-utility-window-header{
  background:rgba(255,248,233,.94);border-bottom-color:var(--border-strong);
}
html[data-theme="light"] .aee-nav-drawer{
  background:linear-gradient(180deg,var(--surface),var(--bg));
  border-right-color:var(--border-strong);
}
html[data-theme="light"] .aee-nav-toggle{
  color:var(--text);background:linear-gradient(145deg,var(--surface),var(--surface-2));
  border-color:var(--border-strong);
}
html[data-theme="light"] .aee-nav-brand strong,
html[data-theme="light"] .aee-nav-item,
html[data-theme="light"] .aee-nav-item:hover{
  color:var(--text);
}
html[data-theme="light"] .aee-nav-item:hover{
  background:rgba(221,161,108,.22);border-color:rgba(121,98,83,.25);
}
html[data-theme="light"] .aee-tabs [role="tablist"]{
  background:linear-gradient(180deg,rgba(248,231,200,.96),rgba(255,248,233,.98))!important;
}
html[data-theme="light"] .aee-tabs [role="tab"][aria-selected="true"]{
  color:var(--text)!important;background:var(--surface)!important;
}
html[data-theme="light"] .aee-formats{
  background:linear-gradient(135deg,rgba(248,231,200,.9),rgba(255,248,233,.96));
}
html[data-theme="light"] .aee-upload-shell{
  background:linear-gradient(180deg,var(--surface),var(--bg))!important;
  border-color:var(--border-strong)!important;
}
html[data-theme="light"] .aee-update-card,
html[data-theme="light"] .aee-update-dialog,
html[data-theme="light"] .aee-about-card,
html[data-theme="light"] .aee-contact-card{
  background:radial-gradient(520px 240px at 100% 0%,rgba(221,161,108,.16),transparent 68%),
    linear-gradient(150deg,var(--surface),var(--bg));
}
html[data-theme="light"] .aee-project-feature,
html[data-theme="light"] .aee-update-step,
html[data-theme="light"] .aee-update-status,
html[data-theme="light"] .aee-up-progress{
  background:var(--surface)!important;border-color:var(--border-strong)!important;
}
html[data-theme="light"] .aee-project-feature:hover{
  background:linear-gradient(145deg,rgba(221,161,108,.22),var(--surface))!important;
}
html[data-theme="light"] .aee-up-progress-label,
html[data-theme="light"] .aee-stat-val{
  color:var(--text)!important;
}
html[data-theme="light"] .aee-up-progress-size,
html[data-theme="light"] .aee-up-progress-hint{
  color:var(--text-muted)!important;
}
html[data-theme="light"] .aee-up-progress-track{
  background:rgba(121,98,83,.2)!important;
}
html[data-theme="light"] .aee-about-note{
  background:rgba(165,141,145,.16);
}
html[data-theme="light"] .aee-nav-toggle:hover,
html[data-theme="light"] .aee-profile-tool:hover,
html[data-theme="light"] .aee-tool:hover{
  border-color:rgba(242,140,50,.55)!important;
}
html[data-theme="light"] .aee-card,
html[data-theme="light"] .aee-settings-card,
html[data-theme="light"] .aee-update-card,
html[data-theme="light"] .aee-update-dialog,
html[data-theme="light"] .aee-about-card,
html[data-theme="light"] .aee-contact-card{
  color:var(--card-text);
  border-color:rgba(121,98,83,.48)!important;
  background:
    radial-gradient(560px 220px at 88% 100%,rgba(242,140,50,.3),transparent 72%),
    radial-gradient(460px 240px at 12% 0%,rgba(181,140,153,.32),transparent 70%),
    linear-gradient(155deg,var(--card-top) 0%,rgba(121,98,83,.94) 48%,rgba(165,141,145,.74) 100%)!important;
  box-shadow:0 24px 60px -42px rgba(121,98,83,.72),0 1px 0 rgba(255,240,204,.18) inset;
}
html[data-theme="light"] .aee-card-title,
html[data-theme="light"] .aee-update-title,
html[data-theme="light"] .aee-update-status strong,
html[data-theme="light"] .aee-settings-card-heading h2,
html[data-theme="light"] .aee-settings-card .aee-account-settings-copy strong,
html[data-theme="light"] .aee-about-title,
html[data-theme="light"] .aee-contact-title,
html[data-theme="light"] .aee-project-page-heading h1,
html[data-theme="light"] .aee-project-feature strong,
html[data-theme="light"] .aee-update-step strong{
  color:var(--card-text)!important;
}
html[data-theme="light"] .aee-card-sub,
html[data-theme="light"] .aee-update-copy,
html[data-theme="light"] .aee-update-notes,
html[data-theme="light"] .aee-settings-card-heading p,
html[data-theme="light"] .aee-settings-card .aee-account-settings-copy p,
html[data-theme="light"] .aee-about-copy,
html[data-theme="light"] .aee-contact-copy,
html[data-theme="light"] .aee-project-feature span,
html[data-theme="light"] .aee-update-step span:last-child{
  color:var(--card-text-muted)!important;
}
html[data-theme="light"] .aee-settings-fields label span[data-testid="block-info"],
html[data-theme="light"] .aee-settings-fields label span:first-child{
  color:var(--card-text-muted)!important;
}
html[data-theme="light"] .aee-project-feature,
html[data-theme="light"] .aee-update-step{
  color:var(--card-text);
  border-color:rgba(121,98,83,.42)!important;
  background:
    radial-gradient(340px 140px at 88% 100%,rgba(242,140,50,.25),transparent 72%),
    linear-gradient(155deg,var(--card-top),rgba(165,141,145,.72))!important;
}
html[data-theme="light"] .aee-project-feature:hover{
  background:
    radial-gradient(340px 140px at 88% 100%,rgba(242,140,50,.38),transparent 72%),
    linear-gradient(155deg,var(--card-top),rgba(165,141,145,.82))!important;
}
html[data-theme="light"] .aee-settings-card .aee-settings-card-heading{
  border-bottom-color:rgba(255,240,204,.22);
}
html[data-theme="light"] .aee-about-card:before,
html[data-theme="light"] .aee-contact-card:before{
  background:linear-gradient(180deg,var(--card-transition),var(--card-glow),transparent);
}
html[data-theme="light"] .aee-card,
html[data-theme="light"] .aee-settings-card,
html[data-theme="light"] .aee-update-card,
html[data-theme="light"] .aee-update-dialog,
html[data-theme="light"] .aee-about-card,
html[data-theme="light"] .aee-contact-card,
html[data-theme="light"] .aee-account-shell .column.wrap>.aee-account-auth-form{
  --surface:#796253;
  --surface-2:rgba(255,240,204,.14);
  --surface-3:rgba(255,240,204,.2);
  --border:rgba(255,240,204,.2);
  --border-strong:rgba(255,240,204,.38);
  --text:#FFF0CC;
  --text-muted:rgba(255,240,204,.86);
  --text-subtle:rgba(255,240,204,.72);
}
html[data-theme="light"] .aee-workflow{
  border-color:rgba(121,98,83,.3);
  background:
    radial-gradient(420px 100px at 88% 100%,rgba(242,140,50,.22),transparent 75%),
    linear-gradient(120deg,#796253,#8D7568 55%,#A58D91);
  box-shadow:0 16px 38px -28px rgba(85,66,55,.62),0 1px 0 rgba(255,240,204,.2) inset;
}
html[data-theme="light"] .aee-workflow-step{
  color:#FFF0CC!important;
}
html[data-theme="light"] .aee-workflow-number{
  color:#FFF0CC;background:rgba(242,140,50,.2);
  border-color:rgba(255,240,204,.32);
}
html[data-theme="light"] .aee-workflow-line{
  background:linear-gradient(90deg,rgba(255,240,204,.38),transparent);
}
html[data-theme="light"] .aee-pill{
  color:#FFF0CC!important;
  border-color:rgba(121,98,83,.35);
  background:linear-gradient(145deg,#796253,#90786C);
  box-shadow:0 5px 16px -12px rgba(85,66,55,.6);
}
html[data-theme="light"] .aee-pill:hover{
  color:#FFF0CC!important;border-color:#DDA16C;
  background:linear-gradient(145deg,#8C705F,#A58D91);
}
html[data-theme="light"] .aee-brand-name{
  color:#554237;
}
html[data-theme="light"] .aee-brand-mark{
  border-color:rgba(121,98,83,.28);
  background:linear-gradient(145deg,rgba(221,161,108,.38),rgba(181,140,153,.34));
  box-shadow:0 12px 30px -20px rgba(121,98,83,.52);
}
html[data-theme="light"] .aee-hero-badge{
  color:#796253!important;
  border-color:rgba(121,98,83,.24);
  background:linear-gradient(135deg,rgba(221,161,108,.24),rgba(181,140,153,.18));
}
html[data-theme="light"] .aee-toolbar .aee-tool{
  color:#FFF0CC!important;
  border-color:rgba(255,240,204,.24)!important;
  background:linear-gradient(145deg,#796253,#92796C)!important;
  box-shadow:0 9px 22px -16px rgba(85,66,55,.62);
}
html[data-theme="light"] .aee-toolbar .aee-tool:hover{
  color:#FFF0CC!important;border-color:#DDA16C!important;
  background:linear-gradient(145deg,#8B705F,#A58D91)!important;
}
html[data-theme="light"] .aee-profile-tool>[data-aee-profile-avatar]{
  color:#FFF0CC!important;
  background:linear-gradient(145deg,#DDA16C,#B58C99);
}
html[data-theme="light"] .aee-upload-shell{
  color:#FFF0CC;
  border-color:rgba(121,98,83,.46)!important;
  background:
    radial-gradient(480px 220px at 88% 100%,rgba(242,140,50,.28),transparent 72%),
    radial-gradient(380px 220px at 8% 0%,rgba(181,140,153,.3),transparent 72%),
    linear-gradient(155deg,#796253 0%,#887064 58%,#A58D91 100%)!important;
  box-shadow:0 24px 60px -42px rgba(85,66,55,.75),0 1px 0 rgba(255,240,204,.2) inset!important;
}
html[data-theme="light"] .aee-upload-shell::before{
  background:linear-gradient(155deg,rgba(255,240,204,.62),rgba(221,161,108,.45) 42%,
    transparent 68%,rgba(181,140,153,.58));
}
html[data-theme="light"] .aee-upload-shell::after{
  background:radial-gradient(circle,rgba(242,140,50,.3),transparent 68%);
}
html[data-theme="light"] .aee-up-title,
html[data-theme="light"] .aee-up-sub{
  color:#FFF0CC!important;
}
html[data-theme="light"] .aee-up-pill{
  color:#FFF0CC!important;
  border-color:rgba(255,240,204,.34);
  background:rgba(255,240,204,.13);
}
html[data-theme="light"] .aee-up-status{
  color:#49382D;background:#FFF0CC;border-color:rgba(255,240,204,.72);
  box-shadow:0 8px 20px -12px rgba(85,66,55,.48);
}
html[data-theme="light"] .aee-upload-trigger,
html[data-theme="light"] .aee-upload-trigger button,
html[data-theme="light"] button.aee-upload-trigger{
  color:#49382D!important;
  border:1px solid rgba(255,240,204,.48)!important;
  background:linear-gradient(115deg,#DDA16C,#F28C32 58%,#DDA16C)!important;
  box-shadow:0 14px 30px -18px rgba(85,66,55,.8),0 1px 0 rgba(255,240,204,.52) inset!important;
}
html[data-theme="light"] .aee-upload-trigger:hover,
html[data-theme="light"] .aee-upload-trigger button:hover,
html[data-theme="light"] button.aee-upload-trigger:hover{
  color:#49382D!important;
  background:linear-gradient(115deg,#E7AF79,#F28C32 58%,#E7AF79)!important;
  box-shadow:0 17px 34px -18px rgba(85,66,55,.82),0 1px 0 rgba(255,240,204,.62) inset!important;
}
html[data-theme="light"] .aee-formats{
  border-color:rgba(255,240,204,.2);
  background:rgba(255,240,204,.12);
}
html[data-theme="light"] .aee-formats-title,
html[data-theme="light"] .aee-formats-note{
  color:rgba(255,240,204,.88)!important;
}
html[data-theme="light"] .aee-format-chip{
  color:#FFF0CC;border-color:rgba(255,240,204,.24);
  background:rgba(255,240,204,.12);
}
html[data-theme="light"] .aee-format-chip:hover{
  color:#49382D;background:#DDA16C;border-color:#FFF0CC;
}
html[data-theme="light"] .aee-format-chip.more{
  color:#49382D;background:#DDA16C;border-color:rgba(255,240,204,.48);
}
html[data-theme="light"] .aee-card .aee-model{
  color:#FFF0CC!important;
  border-color:rgba(255,240,204,.22)!important;
  background:
    radial-gradient(320px 110px at 90% 100%,rgba(242,140,50,.2),transparent 75%),
    linear-gradient(145deg,rgba(255,240,204,.16),rgba(181,140,153,.2))!important;
}
html[data-theme="light"] .aee-card .aee-model span{
  color:rgba(255,240,204,.86)!important;
}
html[data-theme="light"] .aee-class-chip{
  color:#FFF0CC!important;
  border-color:rgba(255,240,204,.24);
  background:linear-gradient(135deg,rgba(181,140,153,.42),rgba(221,161,108,.22));
}
html[data-theme="light"] .aee-account-shell .column.wrap>.aee-account-auth-form{
  border-color:rgba(121,98,83,.48);
  background:
    radial-gradient(460px 240px at 88% 100%,rgba(242,140,50,.25),transparent 74%),
    radial-gradient(420px 220px at 8% 0%,rgba(181,140,153,.32),transparent 72%),
    linear-gradient(155deg,#796253,#8D7568 62%,#A58D91);
  box-shadow:0 28px 72px -48px rgba(85,66,55,.78),0 1px 0 rgba(255,240,204,.2) inset;
}
html[data-theme="light"] .aee-account-intro h2,
html[data-theme="light"] .aee-account-intro p,
html[data-theme="light"] .aee-account-auth-privacy,
html[data-theme="light"] .aee-account-auth-privacy strong,
html[data-theme="light"] .aee-account-setup-hint,
html[data-theme="light"] .aee-account-setup-hint strong,
html[data-theme="light"] .aee-account-auth-form .aee-account-divider{
  color:#FFF0CC!important;
}
html[data-theme="light"] .aee-account-auth-form input{
  color:#554237!important;background:#FFF0CC!important;
  border-color:rgba(255,240,204,.58)!important;
  box-shadow:0 1px 0 rgba(255,255,255,.52) inset!important;
}
html[data-theme="light"] .aee-account-auth-form input::placeholder{
  color:#796253!important;opacity:.82;
}
html[data-theme="light"] .aee-account-auth-form input:focus{
  border-color:#F28C32!important;box-shadow:0 0 0 3px rgba(242,140,50,.26)!important;
}
html[data-theme="light"] .aee-account-auth-form button.aee-account-submit{
  color:#49382D!important;border-color:rgba(255,240,204,.5)!important;
  background:linear-gradient(115deg,#DDA16C,#F28C32)!important;
  box-shadow:0 12px 26px -18px rgba(85,66,55,.82)!important;
}
html[data-theme="light"] .aee-account-auth-form button.aee-account-google,
html[data-theme="light"] .aee-account-auth-form button.aee-account-guest{
  color:#FFF0CC!important;border-color:rgba(255,240,204,.3)!important;
  background:rgba(255,240,204,.12)!important;
}
html[data-theme="light"] .aee-account-auth-form button.aee-account-mode-switch{
  color:#FFF0CC!important;
}
html[data-theme="light"] .aee-account-shell .aee-account-profile,
html[data-theme="light"] .aee-account-shell .aee-account-settings,
html[data-theme="light"] .aee-settings-nav,
html[data-theme="light"] .aee-settings-card{
  color:#FFF0CC!important;
  border-color:rgba(121,98,83,.38)!important;
  background:
    radial-gradient(400px 170px at 90% 100%,rgba(242,140,50,.2),transparent 74%),
    linear-gradient(145deg,#796253,#90786C)!important;
  box-shadow:0 18px 44px -34px rgba(85,66,55,.68),0 1px 0 rgba(255,240,204,.15) inset;
}
html[data-theme="light"] .aee-account-shell .aee-account-identity strong,
html[data-theme="light"] .aee-account-shell .aee-account-eyebrow,
html[data-theme="light"] .aee-settings-card-heading h2,
html[data-theme="light"] .aee-settings-card .aee-account-settings-copy strong{
  color:#FFF0CC!important;
}
html[data-theme="light"] .aee-account-shell .aee-account-identity .aee-account-detail,
html[data-theme="light"] .aee-account-shell .aee-account-identity .aee-account-joined,
html[data-theme="light"] .aee-settings-card-heading p,
html[data-theme="light"] .aee-settings-card .aee-account-settings-copy p{
  color:rgba(255,240,204,.8)!important;
}
html[data-theme="light"] .aee-settings-nav button,
html[data-theme="light"] .aee-settings-nav button:hover,
html[data-theme="light"] .aee-settings-nav button[aria-current="page"]{
  color:#FFF0CC!important;border-color:rgba(255,240,204,.2)!important;
  background:rgba(255,240,204,.1)!important;
}
html[data-theme="light"] .aee-settings-nav button[aria-current="page"]{
  color:#49382D!important;background:#DDA16C!important;border-color:#DDA16C!important;
}
html[data-theme="light"] .aee-settings-fields input,
html[data-theme="light"] .aee-settings-fields select,
html[data-theme="light"] .aee-grant-field input{
  color:#554237!important;background:#FFF0CC!important;
  border-color:rgba(255,240,204,.58)!important;
}
html[data-theme="light"] .aee-settings-photo-button button,
html[data-theme="light"] .aee-settings-theme-button{
  color:#49382D!important;background:#FFF0CC!important;border-color:rgba(255,240,204,.46)!important;
}
html[data-theme="light"] .aee-account-shell .aee-account-mode[data-provider="google"]{
  color:#FFF0CC!important;border-color:rgba(255,240,204,.3)!important;
  background:rgba(255,240,204,.12)!important;
}
html[data-theme="light"] body.aee-access-required #aee-account-window{
  background:
    radial-gradient(900px 540px at 88% 100%,rgba(242,140,50,.22),transparent 74%),
    radial-gradient(860px 520px at 8% 0%,rgba(181,140,153,.2),transparent 76%),
    linear-gradient(145deg,#FFF0CC,#F8E7C8 58%,#EFD9B7)!important;
}
html[data-theme="light"] .aee-update-status{
  color:#FFF0CC!important;
  background:linear-gradient(145deg,#796253,#90786C)!important;
}
html[data-theme="light"] .aee-update-status[data-tone="success"]{
  border-color:rgba(255,240,204,.34)!important;
  background:linear-gradient(145deg,#796253,#90786C)!important;
}
html[data-theme="light"] .aee-update-status[data-tone="available"]{
  color:#49382D!important;border-color:rgba(121,98,83,.26)!important;
  background:linear-gradient(145deg,#DDA16C,#F28C32)!important;
}
html[data-theme="light"] .aee-update-status[data-tone="error"]{
  border-color:rgba(255,240,204,.42)!important;
  background:linear-gradient(145deg,#796253,#554237)!important;
}
html[data-theme="light"] .aee-version-pill{
  color:#FFF0CC!important;border-color:rgba(255,240,204,.24);
  background:rgba(255,240,204,.12);
}
html[data-theme="light"] .aee-update-eyebrow,
html[data-theme="light"] .aee-about-eyebrow,
html[data-theme="light"] .aee-project-page-heading small,
html[data-theme="light"] .aee-project-feature-index{
  color:#FFF0CC!important;
}
html[data-theme="light"] .aee-update-dot{
  background:#F28C32;box-shadow:0 0 16px rgba(242,140,50,.72);
}
html[data-theme="light"] .aee-up-progress{
  color:#FFF0CC;background:linear-gradient(145deg,#796253,#90786C)!important;
  border-color:rgba(255,240,204,.24)!important;
}
html[data-theme="light"] .aee-up-progress-label,
html[data-theme="light"] .aee-up-progress-size,
html[data-theme="light"] .aee-up-progress-hint{
  color:#FFF0CC!important;
}
html[data-theme="light"] .aee-up-progress-track{
  border-color:rgba(255,240,204,.24)!important;background:rgba(255,240,204,.12)!important;
}
html[data-theme="light"] .aee-up-progress-bar,
html[data-theme="light"] .aee-update-progress-fill{
  background:linear-gradient(90deg,#DDA16C,#F28C32,#B58C99)!important;
  box-shadow:0 0 16px rgba(242,140,50,.28)!important;
}
html[data-theme="light"] .aee-utility-window-shell{
  background:
    radial-gradient(1000px 640px at 92% 0%,rgba(181,140,153,.2),transparent 70%),
    radial-gradient(1000px 650px at 0% 100%,rgba(221,161,108,.2),transparent 70%),
    linear-gradient(145deg,#FFF0CC,#F8E7C8 76%)!important;
}
html[data-theme="light"] .aee-utility-window-header{
  color:#796253;background:rgba(255,240,204,.94)!important;
  border-bottom-color:rgba(121,98,83,.24);
}
html[data-theme="light"] .aee-utility-window-close{
  color:#796253!important;background:rgba(121,98,83,.08)!important;
  border-color:rgba(121,98,83,.24)!important;
}
html[data-theme="light"] .aee-tabs [role="tablist"]{
  border-color:rgba(121,98,83,.22)!important;
  background:linear-gradient(145deg,#F8E7C8,#FFF0CC)!important;
  box-shadow:0 8px 22px -18px rgba(85,66,55,.55);
}
html[data-theme="light"] .aee-tabs [role="tab"]{
  color:#796253!important;border-radius:10px!important;
}
html[data-theme="light"] .aee-tabs [role="tab"][aria-selected="true"]{
  color:#49382D!important;background:#DDA16C!important;
  box-shadow:0 6px 16px -12px rgba(85,66,55,.72);
}
html[data-theme="light"] .gradio-container button{
  border-color:rgba(121,98,83,.24);
}
html[data-theme="light"] .gradio-container button.primary,
html[data-theme="light"] .aee-update-actions button.primary,
html[data-theme="light"] .aee-settings-card-heading>button:not(.aee-account-signout){
  color:#49382D!important;border-color:#DDA16C!important;
  background:linear-gradient(115deg,#DDA16C,#F28C32)!important;
  box-shadow:0 10px 24px -18px rgba(85,66,55,.72)!important;
}
html[data-theme="light"] .gradio-container button.secondary,
html[data-theme="light"] .aee-update-actions button.secondary{
  color:#554237!important;border-color:rgba(121,98,83,.24)!important;
  background:#FFF0CC!important;
}
html[data-theme="light"] .gradio-container .wrap.generating{
  color:#FFF0CC!important;border-color:rgba(121,98,83,.32)!important;
  background:linear-gradient(145deg,#796253,#90786C)!important;
  box-shadow:0 18px 50px -24px rgba(85,66,55,.38),inset 0 1px rgba(255,240,204,.14)!important;
}
html[data-theme="light"] .gradio-container .wrap.generating .loading,
html[data-theme="light"] .gradio-container .wrap.generating .progress-level-inner{
  color:#FFF0CC!important;
}
html[data-theme="light"] .gradio-container .wrap.generating::before{
  border-color:rgba(255,240,204,.24);border-top-color:#F28C32;border-right-color:#B58C99;
  box-shadow:0 0 24px rgba(242,140,50,.2);
}
html[data-theme="light"] .gradio-container .wrap.generating .progress-bar-wrap{
  border-color:rgba(255,240,204,.2)!important;background:rgba(255,240,204,.12)!important;
}
html[data-theme="light"] .gradio-container .wrap.generating .progress-bar{
  background:linear-gradient(90deg,#DDA16C,#F28C32,#B58C99)!important;
}
html[data-theme="light"] .aee-about-note{
  color:#554237;border-color:rgba(121,98,83,.2);background:rgba(221,161,108,.22);
}
html[data-theme="light"] .aee-about-note-mark{
  color:#49382D;background:#DDA16C;
}
html[data-theme="light"] .aee-contact-link{
  color:#49382D!important;border-color:rgba(255,240,204,.52)!important;
  background:linear-gradient(115deg,#DDA16C,#F28C32)!important;
  box-shadow:0 14px 30px -19px rgba(85,66,55,.76);
}
html[data-theme="light"] .aee-empty{
  color:#796253!important;border-color:rgba(121,98,83,.28)!important;
  background:linear-gradient(145deg,#FFF8E9,#F8E7C8)!important;
}
html[data-theme="light"] .aee-empty-title,
html[data-theme="light"] .aee-empty-sub{
  color:#796253!important;
}
html[data-theme="light"] .aee-nav-drawer{
  color:#FFF0CC;border-color:rgba(121,98,83,.38)!important;
  background:
    radial-gradient(420px 220px at 90% 100%,rgba(242,140,50,.18),transparent 74%),
    linear-gradient(160deg,#796253,#90786C 68%,#A58D91)!important;
  box-shadow:24px 0 70px -38px rgba(85,66,55,.62);
}
html[data-theme="light"] .aee-nav-head,
html[data-theme="light"] .aee-nav-foot{
  border-color:rgba(255,240,204,.2)!important;color:rgba(255,240,204,.76)!important;
}
html[data-theme="light"] .aee-nav-brand strong,
html[data-theme="light"] .aee-nav-brand span,
html[data-theme="light"] .aee-nav-group-label,
html[data-theme="light"] .aee-nav-item,
html[data-theme="light"] .aee-nav-item svg{
  color:#FFF0CC!important;
}
html[data-theme="light"] .aee-nav-close,
html[data-theme="light"] .aee-nav-item{
  border-color:rgba(255,240,204,.18)!important;background:rgba(255,240,204,.08);
}
html[data-theme="light"] .aee-nav-close:hover,
html[data-theme="light"] .aee-nav-item:hover,
html[data-theme="light"] .aee-nav-item[aria-current="page"]{
  color:#49382D!important;border-color:#DDA16C!important;background:#DDA16C!important;
}
html[data-theme="light"] .aee-nav-item:hover svg,
html[data-theme="light"] .aee-nav-item[aria-current="page"] svg{
  color:#49382D!important;
}
html[data-theme="light"] .aee-settings-language-control,
html[data-theme="light"] .aee-settings-language-control>summary{
  color:#FFF0CC!important;border-color:rgba(255,240,204,.24)!important;
  background:rgba(255,240,204,.1)!important;
}
html[data-theme="light"] .aee-settings-language-control>summary:hover,
html[data-theme="light"] .aee-settings-language-control[open]>summary{
  color:#49382D!important;border-color:#DDA16C!important;background:#DDA16C!important;
}
html[data-theme="light"] .aee-settings-language-menu{
  border-color:rgba(121,98,83,.24)!important;background:#FFF0CC!important;
  box-shadow:0 18px 38px -22px rgba(85,66,55,.4);
}
html[data-theme="light"] .aee-settings-language-menu button,
html[data-theme="light"] .aee-settings-language-menu button:hover,
html[data-theme="light"] .aee-settings-language-menu button[aria-pressed="true"]{
  color:#796253!important;border-color:rgba(121,98,83,.16)!important;background:transparent!important;
}
html[data-theme="light"] .aee-settings-language-menu button:hover,
html[data-theme="light"] .aee-settings-language-menu button[aria-pressed="true"]{
  color:#49382D!important;background:#DDA16C!important;
}
html[data-theme="light"] .aee-history-window .aee-history-heading,
html[data-theme="light"] .aee-history-window .aee-history-toolbar,
html[data-theme="light"] .aee-history-row,
html[data-theme="light"] .aee-history-detail,
html[data-theme="light"] .aee-history-metric{
  color:#FFF0CC!important;border-color:rgba(121,98,83,.32)!important;
  background:
    radial-gradient(360px 140px at 92% 100%,rgba(242,140,50,.18),transparent 74%),
    linear-gradient(145deg,#796253,#90786C)!important;
  box-shadow:0 14px 34px -28px rgba(85,66,55,.58),0 1px 0 rgba(255,240,204,.12) inset;
}
html[data-theme="light"] .aee-history-heading h2,
html[data-theme="light"] .aee-history-window .aee-history-section-heading h2,
html[data-theme="light"] .aee-history-row-title,
html[data-theme="light"] .aee-history-metric strong,
html[data-theme="light"] .aee-history-empty strong{
  color:#FFF0CC!important;
}
html[data-theme="light"] .aee-history-heading h2{
  color:#796253!important;
}
html[data-theme="light"] .aee-history-window .aee-history-heading h2{
  color:#FFF0CC!important;
}
html[data-theme="light"] .aee-history-heading p,
html[data-theme="light"] .aee-history-window .aee-history-section-heading p,
html[data-theme="light"] .aee-history-row-meta,
html[data-theme="light"] .aee-history-metric span,
html[data-theme="light"] .aee-history-metadata dt,
html[data-theme="light"] .aee-history-metadata dd,
html[data-theme="light"] .aee-history-notes,
html[data-theme="light"] .aee-history-window .aee-history-toolbar label span{
  color:rgba(255,240,204,.8)!important;
}
html[data-theme="light"] .aee-history-heading p,
html[data-theme="light"] .aee-history-toolbar label span{
  color:#796253!important;
}
html[data-theme="light"] .aee-history-window .aee-history-heading p,
html[data-theme="light"] .aee-history-window .aee-history-toolbar label span{
  color:rgba(255,240,204,.8)!important;
}
html[data-theme="light"] .aee-history-toolbar{
  border:1px solid rgba(121,98,83,.2);border-radius:14px;
  background:linear-gradient(145deg,#FFF8E9,#F8E7C8);
}
html[data-theme="light"] .aee-history-toolbar input,
html[data-theme="light"] .aee-history-toolbar select{
  color:#554237!important;border-color:rgba(255,240,204,.54)!important;
  background:#FFF0CC!important;
}
html[data-theme="light"] .aee-history-stat,
html[data-theme="light"] .aee-history-export-status{
  color:#49382D!important;border-color:rgba(255,240,204,.48)!important;background:#DDA16C!important;
}
html[data-theme="light"] .aee-history-stat strong,
html[data-theme="light"] .aee-history-stat span{
  color:#49382D!important;
}
html[data-theme="light"] .aee-history-empty{
  color:#796253!important;border-color:rgba(121,98,83,.28)!important;background:#FFF8E9!important;
}
html[data-theme="light"] .aee-history-empty strong{
  color:#554237!important;
}
html[data-theme="light"] .aee-history-row-actions .aee-history-details button,
html[data-theme="light"] #aee-history-details button,
html[data-theme="light"] button#aee-history-details,
html[data-theme="light"] .aee-history-download-all button,
html[data-theme="light"] #aee-history-download-all button,
html[data-theme="light"] button#aee-history-download-all{
  color:#49382D!important;border-color:#DDA16C!important;
  background:linear-gradient(115deg,#DDA16C,#F28C32)!important;
  box-shadow:0 7px 17px -12px rgba(85,66,55,.78)!important;
}
html[data-theme="light"] .aee-history-refresh button,
html[data-theme="light"] #aee-history-refresh button,
html[data-theme="light"] button#aee-history-refresh{
  color:#796253!important;border-color:rgba(255,240,204,.38)!important;
  background:rgba(255,240,204,.14)!important;
}
html[data-theme="light"] .aee-history-row-actions .aee-history-remove button,
html[data-theme="light"] #aee-history-remove button,
html[data-theme="light"] button#aee-history-remove,
html[data-theme="light"] .aee-history-clear button,
html[data-theme="light"] #aee-history-clear button,
html[data-theme="light"] button#aee-history-clear{
  color:#FFF0CC!important;border-color:rgba(255,240,204,.42)!important;
  background:linear-gradient(135deg,#796253,#554237)!important;
  box-shadow:0 6px 16px -10px rgba(85,66,55,.86)!important;
}
html[data-theme="light"] .aee-stat{
  color:#FFF0CC;border-color:rgba(255,240,204,.22);
  background:linear-gradient(145deg,#796253,#90786C);
}
html[data-theme="light"] .aee-stat-val{
  color:#FFF0CC!important;
}
html[data-theme="light"] .aee-stat-lbl{
  color:rgba(255,240,204,.78)!important;
}
html[data-theme="light"] .aee-contact-link:hover,
html[data-theme="light"] .aee-project-feature:hover{
  transform:translateY(-2px);
}
html[data-theme="light"] .gradio-container button:focus-visible,
html[data-theme="light"] .gradio-container input:focus-visible,
html[data-theme="light"] .gradio-container select:focus-visible,
html[data-theme="light"] .gradio-container textarea:focus-visible{
  outline:3px solid #F28C32!important;outline-offset:3px!important;
}
html[data-theme="light"] body .gradio-container .contain .aee-workflow{
  border-color:rgba(121,98,83,.3)!important;
  background:
    radial-gradient(420px 100px at 88% 100%,rgba(242,140,50,.22),transparent 75%),
    linear-gradient(120deg,#796253,#8D7568 55%,#A58D91)!important;
}
html[data-theme="light"] body .gradio-container .contain .aee-workflow-step{
  color:#FFF0CC!important;
}
html[data-theme="light"] body .gradio-container .contain .aee-workflow-number{
  color:#FFF0CC!important;background:rgba(242,140,50,.2)!important;
  border-color:rgba(255,240,204,.32)!important;
}
html[data-theme="light"] body .gradio-container .contain .aee-pill{
  color:#FFF0CC!important;border-color:rgba(121,98,83,.35)!important;
  background:linear-gradient(145deg,#796253,#90786C)!important;
}
html[data-theme="light"] body .gradio-container .contain .aee-toolbar .aee-tool{
  color:#FFF0CC!important;border-color:rgba(255,240,204,.24)!important;
  background:linear-gradient(145deg,#796253,#92796C)!important;
}
html[data-theme="light"] body .gradio-container .contain .aee-upload-shell{
  color:#FFF0CC!important;border-color:rgba(121,98,83,.46)!important;
  background:
    radial-gradient(480px 220px at 88% 100%,rgba(242,140,50,.28),transparent 72%),
    radial-gradient(380px 220px at 8% 0%,rgba(181,140,153,.3),transparent 72%),
    linear-gradient(155deg,#796253 0%,#887064 58%,#A58D91 100%)!important;
}
html[data-theme="light"] body .gradio-container .contain .aee-upload-shell .aee-upload-trigger,
html[data-theme="light"] body .gradio-container .contain .aee-upload-shell .aee-upload-trigger button,
html[data-theme="light"] body .gradio-container .contain .aee-upload-shell button.aee-upload-trigger{
  color:#49382D!important;border-color:rgba(255,240,204,.48)!important;
  background:linear-gradient(115deg,#DDA16C,#F28C32 58%,#DDA16C)!important;
}
html[data-theme="light"] body .gradio-container .contain .aee-formats{
  color:#FFF0CC!important;border-color:rgba(255,240,204,.2)!important;
  background:rgba(255,240,204,.12)!important;
}
html[data-theme="light"] body .gradio-container .contain .aee-formats-title,
html[data-theme="light"] body .gradio-container .contain .aee-formats-note{
  color:rgba(255,240,204,.88)!important;
}
html[data-theme="light"] body .gradio-container .contain .aee-card,
html[data-theme="light"] body .gradio-container .contain .aee-settings-card,
html[data-theme="light"] body .gradio-container .contain .aee-update-card,
html[data-theme="light"] body .gradio-container .contain .aee-update-dialog,
html[data-theme="light"] body .gradio-container .contain .aee-about-card,
html[data-theme="light"] body .gradio-container .contain .aee-contact-card{
  --surface:#796253!important;
  --surface-2:rgba(255,240,204,.14)!important;
  --surface-3:rgba(255,240,204,.2)!important;
  --border:rgba(255,240,204,.2)!important;
  --border-strong:rgba(255,240,204,.38)!important;
  --text:#FFF0CC!important;
  --text-muted:rgba(255,240,204,.86)!important;
  --text-subtle:rgba(255,240,204,.72)!important;
  color:#FFF0CC!important;border-color:rgba(121,98,83,.48)!important;
  background:
    radial-gradient(560px 220px at 88% 100%,rgba(242,140,50,.3),transparent 72%),
    radial-gradient(460px 240px at 12% 0%,rgba(181,140,153,.32),transparent 70%),
    linear-gradient(155deg,#796253 0%,rgba(121,98,83,.94) 48%,rgba(165,141,145,.74) 100%)!important;
}
html[data-theme="light"] body .gradio-container .contain .aee-card .aee-card-title,
html[data-theme="light"] body .gradio-container .contain .aee-card .aee-card-sub,
html[data-theme="light"] body .gradio-container .contain .aee-card .aee-model,
html[data-theme="light"] body .gradio-container .contain .aee-card .aee-model span,
html[data-theme="light"] body .gradio-container .contain .aee-card .aee-class-chip{
  color:#FFF0CC!important;
}
html[data-theme="light"] body .gradio-container .contain .aee-card .aee-model{
  border-color:rgba(255,240,204,.22)!important;
  background:
    radial-gradient(320px 110px at 90% 100%,rgba(242,140,50,.2),transparent 75%),
    linear-gradient(145deg,rgba(255,240,204,.16),rgba(181,140,153,.2))!important;
}
html[data-theme="light"] body .gradio-container .contain .aee-history-row,
html[data-theme="light"] body .gradio-container .contain .aee-history-detail,
html[data-theme="light"] body .gradio-container .contain .aee-history-metric{
  color:#FFF0CC!important;border-color:rgba(121,98,83,.32)!important;
  background:
    radial-gradient(360px 140px at 92% 100%,rgba(242,140,50,.18),transparent 74%),
    linear-gradient(145deg,#796253,#90786C)!important;
}
html[data-theme="light"] body .gradio-container .contain .aee-history-row-title,
html[data-theme="light"] body .gradio-container .contain .aee-history-metric strong{
  color:#FFF0CC!important;
}
html[data-theme="light"] body .gradio-container .contain .aee-history-row-meta,
html[data-theme="light"] body .gradio-container .contain .aee-history-metric span{
  color:rgba(255,240,204,.8)!important;
}
html[data-theme="light"] body .gradio-container .contain .aee-nav-drawer{
  color:#FFF0CC!important;border-color:rgba(121,98,83,.38)!important;
  background:linear-gradient(160deg,#796253,#90786C 68%,#A58D91)!important;
}
html[data-theme="light"] body .gradio-container .contain .aee-nav-item,
html[data-theme="light"] body .gradio-container .contain .aee-nav-brand strong{
  color:#FFF0CC!important;
}
html[data-theme="light"] body .gradio-container .contain .aee-nav-item:hover,
html[data-theme="light"] body .gradio-container .contain .aee-nav-item[aria-current="page"]{
  color:#49382D!important;border-color:#DDA16C!important;background:#DDA16C!important;
}
html[data-theme="light"] body .gradio-container .contain .aee-account-shell .column.wrap>.aee-account-auth-form{
  border-color:rgba(121,98,83,.48)!important;
  background:
    radial-gradient(460px 240px at 88% 100%,rgba(242,140,50,.25),transparent 74%),
    radial-gradient(420px 220px at 8% 0%,rgba(181,140,153,.32),transparent 72%),
    linear-gradient(155deg,#796253,#8D7568 62%,#A58D91)!important;
}
html[data-theme="light"] body .gradio-container .contain .aee-account-intro h2,
html[data-theme="light"] body .gradio-container .contain .aee-account-intro p,
html[data-theme="light"] body .gradio-container .contain .aee-account-auth-privacy,
html[data-theme="light"] body .gradio-container .contain .aee-account-setup-hint{
  color:#FFF0CC!important;
}
html[data-theme="light"] body .gradio-container .contain .aee-account-auth-form input,
html[data-theme="light"] body .gradio-container .contain .aee-settings-fields input,
html[data-theme="light"] body .gradio-container .contain .aee-settings-fields select{
  color:#554237!important;border-color:rgba(255,240,204,.58)!important;
  background:#FFF0CC!important;
}
html[data-theme="light"] body .gradio-container .contain .aee-account-auth-form button.aee-account-submit,
html[data-theme="light"] body .gradio-container .contain .gradio-container button.primary{
  color:#49382D!important;border-color:#DDA16C!important;
  background:linear-gradient(115deg,#DDA16C,#F28C32)!important;
}
html[data-theme="light"] body .gradio-container .contain button.aee-btn-run,
html[data-theme="light"] body .gradio-container .contain button.aee-account-submit,
html[data-theme="light"] body .gradio-container .contain button.primary{
  color:#49382D!important;border-color:#DDA16C!important;
  background:linear-gradient(115deg,#DDA16C,#F28C32)!important;
  box-shadow:0 12px 28px -18px rgba(85,66,55,.78),0 1px 0 rgba(255,240,204,.42) inset!important;
}
html[data-theme="light"] body .gradio-container .contain button.aee-btn-run:hover,
html[data-theme="light"] body .gradio-container .contain button.primary:hover{
  color:#49382D!important;border-color:#DDA16C!important;
  background:linear-gradient(115deg,#E7AF79,#F28C32)!important;
}
html[data-theme="light"] body .gradio-container .contain button.aee-btn-reset{
  color:#796253!important;border-color:rgba(255,240,204,.4)!important;
  background:#FFF0CC!important;
  box-shadow:0 8px 18px -14px rgba(85,66,55,.6)!important;
}
html[data-theme="light"] body .gradio-container .contain button.aee-btn-reset:hover{
  color:#49382D!important;border-color:#DDA16C!important;background:#F8E7C8!important;
}
html[data-theme="light"] body .gradio-container .contain .aee-account-auth-form button.aee-account-google,
html[data-theme="light"] body .gradio-container .contain .aee-account-auth-form button.aee-account-guest{
  color:#FFF0CC!important;border-color:rgba(255,240,204,.3)!important;
  background:rgba(255,240,204,.12)!important;
}
html[data-theme="light"] body .gradio-container .contain .aee-update-status{
  color:#FFF0CC!important;border-color:rgba(255,240,204,.28)!important;
  background:linear-gradient(145deg,#796253,#90786C)!important;
}
html[data-theme="light"] body .gradio-container .contain .aee-up-progress{
  color:#FFF0CC!important;border-color:rgba(255,240,204,.24)!important;
  background:linear-gradient(145deg,#796253,#90786C)!important;
}
html[data-theme="light"] body .gradio-container .contain .aee-tabs [role="tablist"]{
  border-color:rgba(121,98,83,.22)!important;
  background:linear-gradient(145deg,#F8E7C8,#FFF0CC)!important;
}
html[data-theme="light"] body .gradio-container .contain .aee-tabs [role="tab"]{
  color:#796253!important;
}
html[data-theme="light"] body .gradio-container .contain .aee-tabs [role="tab"][aria-selected="true"]{
  color:#49382D!important;background:#DDA16C!important;
}
html[data-theme="light"] body .gradio-container{
  --body-background-fill:#FFF0CC!important;
  --body-text-color:#796253!important;
  --body-text-color-subdued:#796253!important;
  --background-fill-primary:#FFF8E9!important;
  --background-fill-secondary:#F8E7C8!important;
  --background-fill-tertiary:#EFD9B7!important;
  --block-background-fill:#FFF8E9!important;
  --block-border-color:rgba(121,98,83,.24)!important;
  --border-color-primary:rgba(121,98,83,.24)!important;
  --border-color-accent:#DDA16C!important;
  --color-accent:#F28C32!important;
  --color-accent-soft:#F8E7C8!important;
  --input-background-fill:#FFF8E9!important;
  --input-background-fill-focus:#FFFDF7!important;
  --input-border-color:rgba(121,98,83,.35)!important;
  --input-border-color-focus:#DDA16C!important;
  --input-placeholder-color:#796253!important;
  --button-primary-background-fill:#F28C32!important;
  --button-primary-background-fill-hover:#DDA16C!important;
  --button-primary-text-color:#49382D!important;
  --button-secondary-background-fill:#F8E7C8!important;
  --button-secondary-background-fill-hover:#EFD9B7!important;
  --button-secondary-text-color:#796253!important;
}
html[data-theme="light"] body .gradio-container .contain .table-container{
  color:#796253!important;border-color:rgba(121,98,83,.24)!important;
  background:#FFF8E9!important;
}
html[data-theme="light"] body .gradio-container .contain .table-container table,
html[data-theme="light"] body .gradio-container .contain .table-container thead,
html[data-theme="light"] body .gradio-container .contain .table-container tbody{
  color:#796253!important;background:#FFF8E9!important;
}
html[data-theme="light"] body .gradio-container .contain .table-container th,
html[data-theme="light"] body .gradio-container .contain .table-container th.header-cell,
html[data-theme="light"] body .gradio-container .contain .table-container .row-number-header{
  color:#FFF0CC!important;border-color:rgba(255,240,204,.2)!important;
  background:linear-gradient(145deg,#796253,#90786C)!important;
}
html[data-theme="light"] body .gradio-container .contain .table-container td,
html[data-theme="light"] body .gradio-container .contain .table-container .row-number-cell{
  color:#554237!important;border-color:rgba(121,98,83,.16)!important;
  background:#FFF8E9!important;
}
html[data-theme="light"] body .gradio-container .contain .table-container tbody tr:nth-child(even) td{
  background:#F8E7C8!important;
}
html[data-theme="light"] body .gradio-container .contain .table-container tbody tr:hover td{
  background:rgba(221,161,108,.38)!important;
}
html[data-theme="light"] body .gradio-container .contain .table-container input,
html[data-theme="light"] body .gradio-container .contain .table-container button{
  color:#554237!important;border-color:rgba(121,98,83,.24)!important;
  background:#FFF0CC!important;
}
"""


# ============================================================
# HTML builders
# ============================================================
def _hero_html(info):
    fusion_state = "Learned" if FUSION.clf is not None else "Agreement"
    return f"""
    <div class="aee-hero">
      <div class="aee-toolbar">
        <button class="aee-tool aee-update-app" type="button" data-aee-nav-update
          aria-label="Update the app" title="Open app updates" hidden>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"
            stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M20 7v5h-5M4 17v-5h5"/><path d="M6.1 9A7 7 0 0 1 18 6l2 6M4 12l2 6a7 7 0 0 0 11.9-3"/>
          </svg>
          <span>Update the app</span>
        </button>
        <a class="aee-tool" id="theme-toggle" title="Toggle theme (T)">&#9681;</a>
        <a class="aee-tool" id="cmd-open" title="Command palette (Ctrl+K)">&#8981;</a>
        <button class="aee-tool aee-profile-tool" type="button"
          data-aee-open-settings aria-label="Open profile settings" title="Profile settings">
          <span data-aee-profile-avatar aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"
              stroke-linecap="round" stroke-linejoin="round">
              <circle cx="12" cy="8" r="3.5"/><path d="M5 20c.7-3.4 3.2-5.2 7-5.2s6.3 1.8 7 5.2"/>
            </svg>
          </span>
        </button>
      </div>
      <div class="aee-brand">
        <div class="aee-brand-mark" aria-hidden="true">
          <svg viewBox="0 0 48 48" fill="none">
            <path d="M24 4.5 28.4 18l13.1 6-13.1 5.2L24 43.5l-4.8-14.3L6 24l13.2-6L24 4.5Z"
              fill="url(#aeeSpark)" stroke="white" stroke-opacity=".78" stroke-width="1.4" stroke-linejoin="round"/>
            <path d="m35.8 5 .9 3.1 3.1.9-3.1.9-.9 3.1-.9-3.1-3.1-.9 3.1-.9.9-3.1Z" fill="#a5b4fc"/>
            <defs><linearGradient id="aeeSpark" x1="10" y1="10" x2="39" y2="39" gradientUnits="userSpaceOnUse">
              <stop stop-color="#67e8f9"/><stop offset=".52" stop-color="#60a5fa"/><stop offset="1" stop-color="#a78bfa"/>
            </linearGradient></defs>
          </svg>
        </div>
        <div class="aee-brand-copy"><div class="aee-brand-name">AI Evidence Extractor</div>
          <div class="aee-brand-caption">Video intelligence workspace</div></div>
      </div>
      <div class="aee-hero-badge"><span class="aee-pulse"></span> AI-assisted forensic video analysis</div>
      <p class="aee-hero-description">Review detections, follow object tracks, inspect model explanations, and export a complete evidence report from one workspace.</p>
      <div class="aee-hero-pills">
        <span class="aee-pill">{info.get('n_models',0)} models</span>
        <span class="aee-pill">Threat scoring</span>
        <span class="aee-pill">Object tracking</span>
        <span class="aee-pill">Model explainability</span>
        <span class="aee-pill">{fusion_state} fusion</span>
      </div>
    </div>"""


def _workflow_html():
    return """
    <div class="aee-workflow" aria-label="Analysis workflow">
      <div class="aee-workflow-step"><span class="aee-workflow-number">01</span> Add video</div>
      <div class="aee-workflow-line"></div>
      <div class="aee-workflow-step"><span class="aee-workflow-number">02</span> Configure</div>
      <div class="aee-workflow-line"></div>
      <div class="aee-workflow-step"><span class="aee-workflow-number">03</span> Analyze</div>
      <div class="aee-workflow-line"></div>
      <div class="aee-workflow-step"><span class="aee-workflow-number">04</span> Review &amp; export</div>
    </div>"""


def _upload_header_html():
    return """
    <div class="aee-up-head">
      <div class="aee-up-icon">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
             stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="17 8 12 3 7 8"/>
          <line x1="12" y1="3" x2="12" y2="15"/>
        </svg>
      </div>
      <div class="aee-up-text">
        <div class="aee-up-title">
          <span>Evidence intake</span>
          <span class="aee-up-pill">batch supported</span>
        </div>
        <div class="aee-up-sub">Fingerprint every byte before analysis</div>
      </div>
      <div class="aee-up-status">
        <span class="aee-up-status-dot"></span>
        READY
      </div>
    </div>"""


def _formats_html():
    primary = ["MP4","MOV","MKV","AVI","WEBM"]
    more    = ["FLV","WMV","M4V","MPG","MPEG","3GP","3G2","MTS","M2TS",
               "TS","VOB","OGV","RM","RMVB","ASF","DIVX","F4V","MXF"]
    primary_chips = "".join(
        f'<span class="aee-format-chip">{f}</span>' for f in primary)
    more_chips = "".join(
        f'<span class="aee-format-chip">{f}</span>' for f in more)
    total = len(primary) + len(more)
    return f"""
    <div class="aee-formats">
      <div class="aee-formats-title">
        <span>🎞</span>
        <span>Supported formats · {total} total</span>
      </div>
      <div class="aee-formats-list">
        {primary_chips}
        {more_chips}
        <span class="aee-format-chip more">+{total} supported</span>
      </div>
      <div class="aee-formats-note">
        <span>♾️</span>
        <span>No size limit · batch upload supported · content-hashed</span>
      </div>
    </div>"""


def _ensemble_panel_html(info):
    members = info.get("members", [])
    active = [m for m in members if m["enabled"]]
    device = _html.escape(str(info.get("device") or "Unknown"))
    precision = _html.escape(str(info.get("precision") or "Unknown"))
    cards = ""
    for m in members:
        classes = m["classes"]
        is_context = all(norm(c) in CONTEXT for c in classes) and len(classes) <= 3
        tag_lbl = "Context" if is_context else "Weapon"
        chips = "".join(f'<span class="aee-class-chip">{icon_for(norm(c))} {c}</span>'
                        for c in classes[:40])
        mo = CFG.PER_MODEL_CONF.get(m["name"])
        if isinstance(mo, (int, float)):
            gate_lbl = f" · gate {float(mo):.2f}"
        elif isinstance(mo, dict):
            parts = ", ".join(f"{k}≥{v:.2f}" for k, v in mo.items())
            gate_lbl = f" · gate {parts}"
        else:
            gate_lbl = ""
        cards += f"""
        <div class="aee-model" style="padding:14px 16px;
             border-radius:12px;background:var(--surface-2);border:1px solid var(--border);
             margin-bottom:10px;">
          <div style="display:flex;justify-content:space-between;align-items:center;
               margin-bottom:8px;">
            <span style="font-weight:700;font-size:.9rem;
                 font-family:'JetBrains Mono',monospace;">{m['name']}</span>
            <span style="font-size:.72rem;color:var(--text-subtle);
                 font-family:'JetBrains Mono',monospace;">{m['size_mb']} MB</span>
          </div>
          <div style="display:flex;gap:10px;font-size:.72rem;color:var(--text-subtle);
               font-family:'JetBrains Mono',monospace;flex-wrap:wrap;">
            <span>{tag_lbl}</span><span>🎯 {len(classes)} classes</span>
            <span>⚡ conf {m['conf']:.2f}{gate_lbl}</span>
          </div>
          <div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:8px;">{chips}</div>
        </div>"""
    vocab = sorted({c for m in active for c in m["classes"]})
    vocab_chips = "".join(f'<span class="aee-class-chip">{c}</span>'
                          for c in vocab[:32])
    n_weapon = sum(1 for m in active if not (
        all(norm(c) in CONTEXT for c in m["classes"]) and len(m["classes"]) <= 3))
    n_context = len(active) - n_weapon
    return f"""
    <div class="aee-card">
      <div class="aee-card-title">🧠 Ensemble Configuration</div>
      <div class="aee-card-sub">
        {len(active)} active · {n_weapon} weapon · {n_context} context · Object Tracking
      </div>
      <div class="aee-card-sub">Inference device: {device} · {precision}</div>
      <div style="max-height:520px;overflow-y:auto;margin-top:16px;">{cards}</div>
      <div style="margin-top:16px;padding-top:16px;border-top:1px solid var(--border);">
        <div style="font-size:.72rem;color:var(--text-subtle);font-weight:600;
             letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px;">
          Combined vocabulary ({len(vocab)})</div>
        <div>{vocab_chips}</div>
      </div>
    </div>"""


def _empty_html():
    return """
    <div class="aee-empty">
      <div class="aee-empty-icon">🛡️</div>
      <div class="aee-empty-title">Ready to analyze</div>
      <div class="aee-empty-sub">Upload footage and press <b>Run analysis</b>
        to extract weapon evidence with threat scoring, object identity,
        and EigenCAM explainability.</div>
    </div>"""


def _upload_files_html(files):
    if not files: return ""
    if not isinstance(files, list): files = [files]
    total_size, total_dur, valid_count = 0, 0.0, 0
    for f in files:
        p = f if isinstance(f, str) else getattr(f, "name", None)
        if not p or not Path(p).exists(): continue
        info = probe_video(p)
        total_size += info.get("size") or 0
        total_dur  += info.get("duration") or 0.0
        if info.get("valid"): valid_count += 1
    if total_size == 0 and total_dur == 0: return ""
    n = len(files)
    dur_txt = fdur(total_dur) if total_dur else "—"
    ok_badge = (f'<span class="aee-file-stat-ok">● {valid_count} ready</span>'
                if valid_count else
                '<span class="aee-file-stat-warn">● verify codecs</span>')
    return f"""
    <div class="aee-file-summary">
      <div class="aee-file-stats">
        <span>📊 <b>{n}</b> file{"s" if n != 1 else ""}</span>
        <span class="aee-file-stats-sep">·</span>
        <span>💾 <b>{fbytes(total_size)}</b></span>
        <span class="aee-file-stats-sep">·</span>
        <span>⏱ <b>{dur_txt}</b> footage</span>
        <span class="aee-file-stats-sep">·</span>
        {ok_badge}
      </div>
    </div>"""


_STAGES = [("meta","📖","Metadata"),("audio","🔊","Audio"),("detect","🎯","Detect"),
           ("highlight","✂️","Highlights"),("summary","📁","Summaries")]


def _progress_html(stage, stage_frac, msg, elapsed, vid, vi, vt,
                   overall_frac, ticker_dets=None, sub=None, model_times=None):
    idx = next((i for i,(s,_,_) in enumerate(_STAGES) if s == stage), 0)
    stages_html = ""
    for i, (s, ico, lbl) in enumerate(_STAGES):
        cls = "done" if i < idx else ("active" if i == idx else "")
        stages_html += f"""<div class="aee-stage {cls}">
            <span class="aee-stage-icon">{ico}</span>
            <span class="aee-stage-lbl">{lbl}</span>
        </div>"""
    pct = int(max(0, min(1, overall_frac)) * 100)
    R = 60; C = 2 * math.pi * R
    offset = C * (1 - max(0, min(1, overall_frac)))
    ring = f"""
    <div class="aee-ring-wrap">
      <svg class="aee-ring-svg" viewBox="0 0 140 140">
        <defs><linearGradient id="aee-grad" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="#3b82f6"/>
          <stop offset="50%" stop-color="#8b5cf6"/>
          <stop offset="100%" stop-color="#22d3ee"/>
        </linearGradient></defs>
        <circle class="aee-ring-bg" cx="70" cy="70" r="{R}"/>
        <circle class="aee-ring-fg" cx="70" cy="70" r="{R}"
          stroke-dasharray="{C:.1f}" stroke-dashoffset="{offset:.1f}"/>
      </svg>
      <div class="aee-ring-center">
        <div class="aee-ring-pct">{pct}%</div>
        <div class="aee-ring-lbl">Overall</div>
      </div>
    </div>"""
    eta = ""
    if 0.05 < overall_frac < 0.98 and elapsed > 0:
        rate = overall_frac / max(elapsed, 1e-6)
        eta = fdur((1.0 - overall_frac) / max(rate, 1e-6))
    sub_html = ""
    if sub:
        sub_html = f"""
        <div class="aee-substep">
          <div class="aee-sub-item"><span class="aee-sub-lbl">Frame</span>
            <span class="aee-sub-val">{sub.get('frame',0):,} / {sub.get('total_frames',0):,}</span></div>
          <div class="aee-sub-item"><span class="aee-sub-lbl">Sampled</span>
            <span class="aee-sub-val">{sub.get('sampled',0):,} / {sub.get('total_sampled',0):,}</span></div>
          <div class="aee-sub-item"><span class="aee-sub-lbl">Throughput</span>
            <span class="aee-sub-val accent">{sub.get('fps_processed',0):.1f} fps</span></div>
          <div class="aee-sub-item"><span class="aee-sub-lbl">Detections</span>
            <span class="aee-sub-val ok">{sub.get('detections',0)}</span></div>
        </div>"""
    infer_html = ""
    if model_times:
        peak = max(model_times.values()) or 1.
        rows = ""
        for nm, ms in sorted(model_times.items(), key=lambda x: -x[1]):
            w = min(100, ms / peak * 100)
            rows += f"""<div class="aee-infer-row">
                <span class="aee-infer-name">{nm[:26]}</span>
                <div class="aee-infer-bar">
                  <div class="aee-infer-fill" style="width:{w:.0f}%"></div></div>
                <span class="aee-infer-ms">{ms:.0f} ms</span>
            </div>"""
        infer_html = ('<div class="aee-infer">'
                      '<div class="aee-infer-title">Per-model inference time</div>'
                      f'{rows}</div>')
    ticker_html = ""
    if ticker_dets:
        rows = ""
        for d in ticker_dets[-14:][::-1]:
            ic = icon_for(d['label'])
            rows += f"""<div class="aee-tick wpn">
                <span class="t">{ts(d['timestamp'])}</span>
                <span class="l">{ic} {d['label']}</span>
                <span class="c">{d['confidence']:.2f}</span>
                <span class="s">{d.get('source','—')}</span>
            </div>"""
        ticker_html = f"""<div class="aee-ticker">
            <div class="aee-ticker-title"><span class="aee-ticker-dot"></span>
              Live · {len(ticker_dets)} weapon detections</div>
            <div class="aee-ticker-list">{rows}</div>
        </div>"""
    stage_lbl = next((l for s,_,l in _STAGES if s == stage), stage)
    video_pct = int(min(100, stage_frac * 100))
    return f"""
    <div class="aee-load">
      <div class="aee-load-top">{ring}
        <div>
          <div class="aee-load-video">
            <span class="aee-load-video-idx">Video {vi} / {vt}</span>
            <span>{vid}</span>
          </div>
          <div class="aee-load-stage">{stage_lbl}</div>
          <div class="aee-load-msg">{msg or 'Working…'}</div>
          <div style="font-size:.75rem;color:var(--text-subtle);margin-top:8px;
               font-family:'JetBrains Mono',monospace;">⏱ {fdur(elapsed)} elapsed
               {f'→ ETA {eta}' if eta else ''}</div>
        </div>
      </div>
      <div class="aee-bars">
        <div><div class="aee-bar-lbl"><span>Overall</span><b>{pct}%</b></div>
          <div class="aee-bar-outer tall">
            <div class="aee-bar-fill" style="width:{pct}%"></div></div></div>
        <div><div class="aee-bar-lbl"><span>{stage_lbl}</span><b>{video_pct}%</b></div>
          <div class="aee-bar-outer">
            <div class="aee-bar-fill cyan" style="width:{video_pct}%"></div></div></div>
      </div>
      <div class="aee-stages">{stages_html}</div>
      {sub_html}{infer_html}{ticker_html}
    </div>"""


_MASTER_MAP = []


def _src_to_master(video_name, ts_val):
    for rng in _MASTER_MAP:
        if rng["video"] == video_name and rng["src_start"] <= ts_val <= rng["src_end"]:
            return rng["master_start"] + (ts_val - rng["src_start"])
    return None


def _compute_heatmap_payload(results):
    global _MASTER_MAP
    if not _MASTER_MAP:
        return None
    dur = max(r["master_end"] for r in _MASTER_MAP) or 0
    if dur <= 0:
        return None
    nbins = max(180, min(2000, int(math.ceil(dur * 1.5))))
    bin_size = dur / nbins
    per_model = {}
    class_models = [defaultdict(set) for _ in range(nbins)]
    total_detections = 0
    for r in results:
        for d in r.detections:
            if d.get("kind") != "object": continue
            if d.get("label") not in WEAPONS: continue
            mt = _src_to_master(r.video_name, d["timestamp"])
            if mt is None: continue
            model = str(d.get("source") or "unknown")
            if model not in per_model:
                per_model[model] = {
                    "heat": [0.0] * nbins,
                    "counts": [0] * nbins,
                    "hits": 0,
                    "confidence_sum": 0.0,
                }
            i = min(nbins - 1, max(0, int(float(mt) / bin_size)))
            confidence = min(1.0, max(0.0, float(d.get("confidence", 0.0))))
            per_model[model]["heat"][i] += confidence
            per_model[model]["counts"][i] += 1
            per_model[model]["hits"] += 1
            per_model[model]["confidence_sum"] += confidence
            class_models[i][d["label"]].add(model)
            total_detections += 1
    if not per_model:
        return None
    ordered_models = sorted(per_model)
    models = {}
    comparative_activity = [0.0] * nbins
    for model in ordered_models:
        row = per_model[model]
        model_peak = max(row["heat"]) or 1.0
        for i, value in enumerate(row["heat"]):
            comparative_activity[i] += value / model_peak
        models[model] = {
            "heat": [round(value, 3) for value in row["heat"]],
            "counts": row["counts"],
            "hits": row["hits"],
            "mean_confidence": round(
                row["confidence_sum"] / max(1, row["hits"]), 3),
        }
    consensus = [
        max((len(names) for names in labels.values()), default=0)
        for labels in class_models
    ]
    peak_total = max(comparative_activity) or 1.0
    win = max(2, nbins // 20)
    prefix = [0.0]
    for value in comparative_activity:
        prefix.append(prefix[-1] + value)
    best_i = max(range(nbins - win + 1),
                 key=lambda i: prefix[i + win] - prefix[i])
    return {
        "version": 2,
        "duration": round(dur, 3),
        "nbins": nbins,
        "models": models,
        "consensus": consensus,
        "total_detections": total_detections,
        "peak_total": round(peak_total, 3),
        "peak_window": [best_i, best_i + win],
    }


def _threat_distribution_html(results, tracker=None):
    buckets = {"critical":[],"high":[],"medium":[],"low":[]}
    for r in results:
        for d in r.detections:
            if d.get("kind") != "object" or d.get("label") not in WEAPONS: continue
            lvl, sc, _ = threat_score(d, tracker, r.video_name)
            if lvl in buckets: buckets[lvl].append((sc, d, r.video_name))
    total = sum(len(v) for v in buckets.values())
    if total == 0: return ""
    cells = ""
    for lvl in ["critical","high","medium","low"]:
        n = len(buckets[lvl])
        avg = (sum(x[0] for x in buckets[lvl]) / n) if n else 0
        col = THREAT_COLORS[lvl]
        cells += (f'<div class="aee-threat-cell" style="--lvl-color:{col};">'
                  f'<div class="lvl">{lvl}</div>'
                  f'<div class="num">{n}</div>'
                  f'<div class="sub">avg risk {avg:.0f}</div></div>')
    top = sorted([x for v in buckets.values() for x in v], key=lambda t: -t[0])[:3]
    top_rows = ""
    for sc, d, vn in top:
        lvl, _, reasons = threat_score(d, tracker, vn)
        col = THREAT_COLORS.get(lvl, "#64748b")
        top_rows += (f'<div style="padding:10px 14px;background:rgba(0,0,0,.28);'
                     f'border-radius:8px;border-left:2px solid {col};'
                     f'display:grid;grid-template-columns:70px 1fr 60px 90px;'
                     f'gap:10px;align-items:center;font-size:.78rem;'
                     f'font-family:\'JetBrains Mono\',monospace;">'
                     f'<span style="color:var(--text-subtle);">'
                     f'{ts(d["timestamp"])}</span>'
                     f'<span style="color:var(--text);font-weight:600;">'
                     f'{icon_for(d["label"])} {d["label"]}</span>'
                     f'<span style="color:{col};font-weight:700;">R{sc}</span>'
                     f'<span style="color:var(--text-subtle);font-size:.68rem;'
                     f'text-align:right;overflow:hidden;text-overflow:ellipsis;'
                     f'white-space:nowrap;">{vn[:16]}</span>'
                     f'</div>')
    return f"""
    <div class="aee-threat-card">
      <div class="aee-threat-title">
        🎚 Threat Distribution
        <span>automated scoring · {total} weapon hits</span>
      </div>
      <div class="aee-threat-grid">{cells}</div>
      <div style="font-size:.72rem;color:var(--text-subtle);font-weight:700;
           letter-spacing:.1em;text-transform:uppercase;margin:18px 0 10px;">
        Top-risk detections</div>
      <div style="display:flex;flex-direction:column;gap:6px;">{top_rows}</div>
    </div>"""


def _overview_html(results, errs, sums, master, pdf, elapsed, tracker=None):
    if not results: return _empty_html()
    total = len(results)
    payload = None
    heat_html = ""
    if master is not None:
        payload = _compute_heatmap_payload(results)
        if payload:
            json_str = json.dumps(payload, separators=(",", ":"))
            escaped = _html.escape(json_str, quote=False)
            heat_html = (f'<div class="aee-heat-data" '
                         f'style="display:none">{escaped}</div>')
    peak_start = peak_end = None
    if payload and payload.get("peak_window"):
        master_dur = payload["duration"]
        nb = payload["nbins"]
        bin_size = master_dur / max(1, nb)
        peak_start = payload["peak_window"][0] * bin_size
        peak_end = payload["peak_window"][1] * bin_size
    weapon_pairs = [(r.video_name, d) for r in results for d in r.detections
                    if d.get("kind") == "object" and d.get("label") in WEAPONS]
    weapon_dets = [d for _, d in weapon_pairs]
    context_dets = [d for r in results for d in r.detections
                    if d.get("label") in CONTEXT]
    nc = sum(len(r.captures) for r in results)
    times = [d["timestamp"] for d in weapon_dets]
    tl = ""
    if times:
        tmin, tmax = min(times), max(times); span = max(tmax - tmin, 1e-3)
        nb_b = min(80, max(24, int(span))); bs = span / nb_b
        bins = [{"c":0,"cls":set(),"first":None} for _ in range(nb_b)]
        for vname, d in weapon_pairs:
            i = min(nb_b-1, int((d["timestamp"]-tmin)/bs))
            bins[i]["c"] += 1; bins[i]["cls"].add(d["label"])
            if bins[i]["first"] is None:
                bins[i]["first"] = (vname, d["timestamp"])
        peak = max(b["c"] for b in bins) or 1
        bars = ""
        for i, b in enumerate(bins):
            t0 = tmin + i*bs; t1 = tmin + (i+1)*bs
            h = max(3, int(b["c"]/peak*100))
            cs = ", ".join(sorted(b["cls"])[:3]) or "no detections"
            seek_id = ""
            title_text = f'{ts(t0)} → {ts(t1)}&#10;{b["c"]} dets · {cs}'
            cursor = "default"
            in_peak = False
            if b["first"] is not None:
                mtime = _src_to_master(b["first"][0], b["first"][1])
                if mtime is not None:
                    ms = int(round(mtime * 1000))
                    seek_id = f'id="aee-seek-{i}-{ms}"'
                    title_text = f'▶ click to seek {mtime:.2f}s&#10;' + title_text
                    cursor = "pointer"
                    if (peak_start is not None and peak_end is not None
                            and peak_start <= mtime <= peak_end):
                        in_peak = True
                        title_text = '★ peak activity · ' + title_text
            else:
                seek_id = f'id="aee-seek-{i}-none"'
            bg = ("linear-gradient(180deg,#22d3ee,#06b6d4)" if in_peak
                  else "linear-gradient(180deg,#3b82f6,#8b5cf6)")
            shadow = ("box-shadow:0 0 8px rgba(34,211,238,.75);" if in_peak else "")
            bars += (f'<div class="aee-tl-bar" {seek_id} '
                     f'style="flex:1;height:{h}%;min-height:3px;'
                     f'border-radius:3px 3px 0 0;'
                     f'background:{bg};{shadow}'
                     f'cursor:{cursor};" '
                     f'title="{title_text}"></div>')
        tl = (f'<div style="padding:22px;border-radius:14px;background:var(--surface-2);'
              f'border:1px solid var(--border);margin:14px 0;">'
              f'<h4 style="font-size:.9rem;margin-bottom:14px;display:flex;'
              f'justify-content:space-between;align-items:baseline;color:var(--text);">'
              f'Weapon Detection Timeline '
              f'<span style="font-size:.7rem;color:var(--text-subtle);font-weight:400;'
              f'text-transform:uppercase;letter-spacing:.1em;">'
               f'click any bar to seek · <span style="color:#93c5fd;">'
               f'blue = peak activity</span></span></h4>'
              f'<div style="display:flex;align-items:flex-end;gap:2px;height:110px;">'
              f'{bars}</div>'
              f'<div style="display:flex;justify-content:space-between;font-size:.7rem;'
              f'color:var(--text-subtle);margin-top:10px;'
              f"font-family:'JetBrains Mono',monospace;\">"
              f'<span>{ts(tmin)}</span><span>{ts((tmin+tmax)/2)}</span>'
              f'<span>{ts(tmax)}</span></div></div>')
    return f"""
    {heat_html}
    <div class="aee-stats">
      <div class="aee-stat"><span class="aee-stat-icon">🎬</span>
        <div class="aee-stat-val">{total}</div>
        <div class="aee-stat-lbl">Videos</div></div>
      <div class="aee-stat"><span class="aee-stat-icon">🔫</span>
        <div class="aee-stat-val">{len(weapon_dets)}</div>
        <div class="aee-stat-lbl">Weapon dets</div></div>
      <div class="aee-stat"><span class="aee-stat-icon">👤</span>
        <div class="aee-stat-val">{len(context_dets)}</div>
        <div class="aee-stat-lbl">Context hits</div></div>
      <div class="aee-stat"><span class="aee-stat-icon">📸</span>
        <div class="aee-stat-val">{nc}</div>
        <div class="aee-stat-lbl">Captures</div></div>
      <div class="aee-stat"><span class="aee-stat-icon">⏱</span>
        <div class="aee-stat-val" style="font-size:1.15rem;">{fdur(elapsed)}</div>
        <div class="aee-stat-lbl">Elapsed</div></div>
    </div>
    {_threat_distribution_html(results, tracker)}
    {tl}"""


# ============================================================
# JS head.
#
# Gradio 6 parses head= as an HTML document and re-appends only its head
# children. Keep executable scripts and stylesheet links here; create any
# body-level overlay elements from JavaScript after this script runs.
# ============================================================
_JS_HEAD = r"""
<script>
(function(){
  try {
    var gradioVersion = __AEE_GRADIO_VERSION__;
    var hrefDescriptor = window.HTMLLinkElement
      ? Object.getOwnPropertyDescriptor(HTMLLinkElement.prototype, 'href')
      : null;
    if (gradioVersion && String(gradioVersion).startsWith('6.') &&
        !window.__aeeGradioCssPathFix && hrefDescriptor && hrefDescriptor.set){
      Object.defineProperty(HTMLLinkElement.prototype, 'href', {
        configurable: true,
        enumerable: hrefDescriptor.enumerable,
        get: hrefDescriptor.get,
        set: function(value){
          var target = value;
          try {
            var asset = new URL(String(value), document.baseURI);
            if (asset.origin === 'https://gradio.s3-us-west-2.amazonaws.com' &&
                /^\/assets\/[^/]+\.css$/i.test(asset.pathname)){
              asset.pathname = '/' + gradioVersion + asset.pathname;
              target = asset.href;
            }
          } catch(_){}
          return hrefDescriptor.set.call(this, target);
        }
      });
      window.__aeeGradioCssPathFix = true;
      console.info('[AEE] Gradio CDN CSS path compatibility enabled', gradioVersion);
    }
  } catch(err) {
    try { console.warn('[AEE] Gradio CSS path workaround unavailable', err); } catch(_){}
  }
})();
</script>

<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet"
      href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap">

<script>
(function(){
  try {
  const LOG = (...a) => { try { console.log('[AEE]', ...a); } catch(_){} };
  try {
    const _gradioVersion = __AEE_GRADIO_VERSION__;
    const _linkHref = window.HTMLLinkElement
      ? Object.getOwnPropertyDescriptor(HTMLLinkElement.prototype, 'href')
      : null;
    if (_gradioVersion && String(_gradioVersion).startsWith('6.') &&
        !window.__aeeGradioCssPathFix && _linkHref && _linkHref.set){
      Object.defineProperty(HTMLLinkElement.prototype, 'href', {
        configurable: true,
        enumerable: _linkHref.enumerable,
        get: _linkHref.get,
        set: function(value){
          let target = value;
          try {
            const asset = new URL(String(value), document.baseURI);
            if (asset.origin === 'https://gradio.s3-us-west-2.amazonaws.com' &&
                /^\/assets\/[^/]+\.css$/i.test(asset.pathname)){
              asset.pathname = '/' + _gradioVersion + asset.pathname;
              target = asset.href;
            }
          } catch(_){}
          return _linkHref.set.call(this, target);
        }
      });
      window.__aeeGradioCssPathFix = true;
      LOG('Gradio CDN CSS path compatibility enabled', _gradioVersion);
    }
  } catch(err) {
    try { console.warn('[AEE] Gradio CSS path workaround unavailable', err); } catch(_){}
  }
  LOG('bootstrap start');

  const overlayRoot = document.body || document.documentElement;
  if (!document.getElementById('aee-toast-wrap')){
    const toastWrap = document.createElement('div');
    toastWrap.className = 'aee-toast-wrap';
    toastWrap.id = 'aee-toast-wrap';
    overlayRoot.appendChild(toastWrap);
  }
  if (!document.getElementById('aee-cmd')){
    const cmd = document.createElement('div');
    cmd.className = 'aee-cmd';
    cmd.id = 'aee-cmd';
    cmd.setAttribute('aria-hidden', 'true');
    cmd.innerHTML =
      '<div class="aee-cmd-backdrop"></div>' +
      '<div class="aee-cmd-panel">' +
        '<input class="aee-cmd-input" id="aee-cmd-input" ' +
          'placeholder="Type a command or jump to a tab…" autocomplete="off"/>' +
        '<div class="aee-cmd-list" id="aee-cmd-list"></div>' +
        '<div class="aee-cmd-foot">' +
          '<span><span class="kbd">↑↓</span> navigate</span>' +
          '<span><span class="kbd">↵</span> run</span>' +
          '<span><span class="kbd">esc</span> close</span>' +
        '</div>' +
      '</div>';
    overlayRoot.appendChild(cmd);
  }

  const root = document.documentElement;
  const saved = localStorage.getItem('aee-theme') || 'dark';
  root.setAttribute('data-theme', saved);

  function toggleTheme(){
    const cur = root.getAttribute('data-theme');
    const nxt = cur === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', nxt);
    localStorage.setItem('aee-theme', nxt);
    toast(nxt === 'dark' ? '🌙 Dark theme' : '☀️ Light theme', 'success');
  }
  window.aeeToggleTheme = toggleTheme;

  function clickTab(i){
    const tabs = document.querySelectorAll('.aee-tabs [role="tab"]');
    if (tabs[i]) tabs[i].click();
  }

  function setAeeNavigation(open){
    if (!document.body) return;
    document.body.classList.toggle('aee-nav-open', !!open);
    const toggle = document.querySelector('[data-aee-nav-toggle]');
    if (toggle) toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
  }
  function syncAeeProfileTool(){
    const target = document.querySelector('[data-aee-profile-avatar]');
    const source = document.querySelector('#aee-account-profile-source .aee-account-profile');
    if (!target || !source) return;
    const avatar = source.querySelector('.aee-account-avatar');
    const identity = source.querySelector('.aee-account-identity strong');
    if (!avatar) return;
    const image = avatar.querySelector('img');
    const name = identity ? identity.textContent.trim() : 'Profile';
    const signature = name + '|' + (image ? image.getAttribute('src') : avatar.textContent.trim());
    if (target.dataset.profileSignature === signature) return;
    target.dataset.profileSignature = signature;
    target.replaceChildren(image ? image.cloneNode(true) : document.createTextNode(
      avatar.textContent.trim() || 'A'
    ));
    const button = target.closest('[data-aee-open-settings]');
    if (button){
      button.setAttribute('aria-label', 'Open profile settings for ' + name);
      button.title = name + ' — Profile settings';
    }
  }
  function syncAeeTopbarLayout(){
    const mobile = window.innerWidth <= 640;
    const toggle = document.querySelector('[data-aee-nav-toggle]');
    if (toggle){
      toggle.style.left = mobile ? '12px' : '';
      toggle.style.top = mobile ? '12px' : '';
      toggle.style.width = mobile ? '40px' : '';
      toggle.style.height = mobile ? '40px' : '';
      toggle.style.borderRadius = mobile ? '12px' : '';
    }
  }
  function syncAeeUpdateShortcut(){
    const button = document.querySelector('[data-aee-nav-update]');
    const status = document.querySelector('#aee-update-status [data-update-available]');
    if (button) button.hidden = !status || status.dataset.updateAvailable !== 'true';
  }
  window.addEventListener('resize', syncAeeTopbarLayout, {passive:true});
  function aeeAccessRequired(){
    const marker = document.querySelector('[data-aee-access-status]');
    if (marker) return marker.dataset.aeeAccessStatus !== 'ready';
    const startup = document.querySelector('[data-aee-account-startup]');
    return !!startup && startup.dataset.aeeAccountStartup === 'required';
  }
  function applyAeeAccessGate(){
    if (!document.body) return;
    const required = aeeAccessRequired();
    const wasRequired = window.__aeeAccessWasRequired;
    window.__aeeAccessWasRequired = required;
    document.body.classList.toggle('aee-access-required', required);
    const close = document.querySelector('#aee-account-close button');
    if (close && close.disabled !== required) close.disabled = required;
    if (wasRequired === true && !required){
      window.setTimeout(() => {
        const accountWindow = document.querySelector('#aee-account-window');
        if (accountWindow && accountWindow.getAttribute('aria-modal') === 'true'){
          closeAeeUtilityWindow(accountWindow);
        }
      }, 0);
    }
  }
  function isAeeAccountTarget(target){
    return target instanceof Element && !!target.closest('#aee-account-window');
  }
  document.addEventListener('click', event => {
    if (!aeeAccessRequired()) return;
    if (isAeeAccountTarget(event.target) &&
        !event.target.closest('.aee-utility-window-close')) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    window.setTimeout(() => activateAeeUtilityWindow('account'), 0);
  }, true);
  document.addEventListener('keydown', event => {
    if (!aeeAccessRequired()) return;
    if (event.key === 'Escape'){
      event.preventDefault();
      event.stopImmediatePropagation();
      return;
    }
    if (!isAeeAccountTarget(event.target)){
      event.preventDefault();
      event.stopImmediatePropagation();
      const first = document.querySelector(
        '#aee-account-window input:not([disabled]), #aee-account-window button:not([disabled])');
      if (first) first.focus({preventScroll:true});
      return;
    }
    if (event.key === 'Tab'){
      const controls = Array.from(document.querySelectorAll(
        '#aee-account-window input:not([disabled]), #aee-account-window button:not([disabled]), ' +
        '#aee-account-window textarea:not([disabled]), #aee-account-window select:not([disabled])'
      )).filter(control => control.offsetParent !== null);
      if (!controls.length) return;
      const first = controls[0];
      const last = controls[controls.length - 1];
      if (event.shiftKey && (document.activeElement === first ||
          !isAeeAccountTarget(document.activeElement))){
        event.preventDefault();
        last.focus({preventScroll:true});
      } else if (!event.shiftKey && document.activeElement === last){
        event.preventDefault();
        first.focus({preventScroll:true});
      }
    }
  }, true);
  document.addEventListener('focusin', event => {
    if (!aeeAccessRequired() || isAeeAccountTarget(event.target)) return;
    const first = document.querySelector(
      '#aee-account-window input:not([disabled]), #aee-account-window button:not([disabled])');
    if (first) first.focus({preventScroll:true});
  }, true);
  function activateAeeUtilityWindow(name){
    setAeeNavigation(false);
    const shellId = 'aee-' + name + '-window';
    const shells = Array.from(document.querySelectorAll('[id="' + shellId + '"]'));
    if (!shells.length) return;
    document.querySelectorAll('.aee-utility-window-shell').forEach(panel => {
      panel.style.display = 'none';
      panel.classList.remove('aee-utility-window-open');
      panel.setAttribute('aria-hidden', 'true');
      panel.removeAttribute('aria-modal');
      panel.removeAttribute('role');
    });
    document.body.classList.add('aee-utility-window-active');
    shells.forEach(panel => {
      panel.style.display = 'block';
      panel.scrollTop = 0;
      panel.setAttribute('aria-hidden', 'false');
      requestAnimationFrame(() => panel.classList.add('aee-utility-window-open'));
    });
    const shell = shells[0];
    shell.setAttribute('aria-modal', 'true');
    shell.setAttribute('role', 'dialog');
    const close = shell.querySelector('.aee-utility-window-close button');
    const initialFocus = name === 'account' && aeeAccessRequired()
      ? shell.querySelector('.aee-account-auth-form input:not([disabled])')
      : close;
    if (initialFocus) initialFocus.focus({preventScroll:true});
    document.dispatchEvent(new CustomEvent('aee:utility-window', {detail:{open:true,name}}));
  }
  function closeAeeUtilityWindow(shell){
    shell = shell || document.querySelector('.aee-utility-window-shell[aria-modal="true"]');
    if (!shell) return;
    const settingsForm = shell.querySelector(
      '#aee-settings-profile-section[data-aee-dirty="true"]');
    if (settingsForm && settingsForm.dataset.aeeDirty === 'true' &&
        !window.confirm('Discard your unsaved profile changes?')){
      return;
    }
    const shellId = shell.id;
    document.querySelectorAll('[id="' + shellId + '"]').forEach(panel => {
      panel.classList.remove('aee-utility-window-open');
      panel.setAttribute('aria-hidden', 'true');
      panel.removeAttribute('aria-modal');
      panel.removeAttribute('role');
    });
    shell.removeAttribute('aria-modal');
    shell.removeAttribute('role');
    window.setTimeout(() => {
      if (document.querySelector('.aee-utility-window-shell[aria-modal="true"]')) return;
      document.querySelectorAll('.aee-utility-window-shell:not(.aee-utility-window-open)')
        .forEach(panel => { panel.style.display = 'none'; });
      document.body.classList.remove('aee-utility-window-active');
      const toggle = document.querySelector('[data-aee-nav-toggle]');
      if (toggle) toggle.focus({preventScroll:true});
    }, 350);
    document.dispatchEvent(new CustomEvent('aee:utility-window', {detail:{open:false}}));
  }
  function markAeeProfileFieldChanges(event){
    const field = event.target.closest(
      '.aee-settings-fields input, .aee-settings-fields select') ||
      (event.target.matches('#aee-photo-file-input')
        ? event.target : null);
    if (!field) return;
    const form = field.closest('#aee-settings-profile-section') ||
      document.querySelector('#aee-settings-profile-section');
    if (!form) return;
    if (field.type !== 'file'){
      if (!Object.prototype.hasOwnProperty.call(field.dataset, 'aeeInitialValue')){
        field.dataset.aeeInitialValue = field.value;
      }
      field.dataset.aeeModified = field.value !== field.dataset.aeeInitialValue
        ? 'true' : 'false';
    } else {
      form.dataset.aeePhotoDirty = field.files && field.files.length ? 'true' : 'false';
    }
    const changed = form.querySelector('[data-aee-modified="true"]') ||
      form.dataset.aeePhotoDirty === 'true';
    form.dataset.aeeDirty = changed ? 'true' : 'false';
    const notice = form.querySelector('[data-aee-settings-dirty]');
    if (notice) notice.hidden = !changed;
  }
  function initializeAeePhotoEditor(){
    const openButton = document.getElementById('aee-photo-change-button');
    const fileInput = document.getElementById('aee-photo-file-input');
    const modal = document.getElementById('aee-photo-editor-modal');
    const canvas = document.getElementById('aee-photo-crop-canvas');
    const zoomRange = document.getElementById('aee-photo-zoom-range');
    const errorMessage = document.getElementById('aee-photo-editor-error');
    if (!openButton || !fileInput || !modal || !canvas || !zoomRange) return;
    if (openButton.dataset.aeePhotoEditorBound) return;
    openButton.dataset.aeePhotoEditorBound = 'true';
    document.body.append(modal, fileInput);
    const context = canvas.getContext('2d');
    const picker = fileInput;
    let image = null;
    let objectUrl = '';
    let zoom = 1;
    let panX = 0;
    let panY = 0;
    let drag = null;
    const maximumPhotoBytes = 5 * 1024 * 1024;

    function componentControl(id, selector){
      const component = document.getElementById(id);
      if (!component) return null;
      return component.matches(selector) ? component : component.querySelector(selector);
    }

    function geometry(){
      const bounds = canvas.getBoundingClientRect();
      const width = bounds.width;
      const height = bounds.height;
      const side = Math.min(width, height) * .82;
      const x = (width - side) / 2;
      const y = (height - side) / 2;
      const scale = image
        ? Math.max(side / image.naturalWidth, side / image.naturalHeight) * zoom
        : 1;
      const left = width / 2 - image.naturalWidth * scale / 2 + panX;
      const top = height / 2 - image.naturalHeight * scale / 2 + panY;
      return {width, height, side, x, y, scale, left, top};
    }
    function constrainPan(layout){
      const leftAtCenter = layout.width / 2 - image.naturalWidth * layout.scale / 2;
      const topAtCenter = layout.height / 2 - image.naturalHeight * layout.scale / 2;
      panX = Math.max(layout.x + layout.side - leftAtCenter - image.naturalWidth * layout.scale,
        Math.min(layout.x - leftAtCenter, panX));
      panY = Math.max(layout.y + layout.side - topAtCenter - image.naturalHeight * layout.scale,
        Math.min(layout.y - topAtCenter, panY));
    }
    function draw(){
      if (!image || !image.complete || !image.naturalWidth) return;
      const bounds = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const pixelWidth = Math.round(bounds.width * dpr);
      const pixelHeight = Math.round(bounds.height * dpr);
      if (canvas.width !== pixelWidth || canvas.height !== pixelHeight){
        canvas.width = pixelWidth;
        canvas.height = pixelHeight;
      }
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      const layout = geometry();
      constrainPan(layout);
      layout.left = bounds.width / 2 -
        image.naturalWidth * layout.scale / 2 + panX;
      layout.top = bounds.height / 2 -
        image.naturalHeight * layout.scale / 2 + panY;
      context.clearRect(0, 0, layout.width, layout.height);
      context.drawImage(image, layout.left, layout.top,
        image.naturalWidth * layout.scale, image.naturalHeight * layout.scale);
      context.fillStyle = 'rgba(7,10,13,.62)';
      context.fillRect(0, 0, layout.width, layout.y);
      context.fillRect(0, layout.y + layout.side, layout.width,
        layout.height - layout.y - layout.side);
      context.fillRect(0, layout.y, layout.x, layout.side);
      context.fillRect(layout.x + layout.side, layout.y,
        layout.width - layout.x - layout.side, layout.side);
      context.strokeStyle = '#ff5d55';
      context.lineWidth = 3;
      context.strokeRect(layout.x + 1.5, layout.y + 1.5,
        layout.side - 3, layout.side - 3);
    }
    function showError(message){
      errorMessage.textContent = message;
      errorMessage.hidden = !message;
    }
    function closeEditor(){
      modal.hidden = true;
      document.body.classList.remove('aee-photo-editor-open');
      image = null;
      drag = null;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      objectUrl = '';
      picker.value = '';
      showError('');
      const section = document.getElementById('aee-settings-profile-section');
      if (section){
        section.dataset.aeePhotoDirty = 'false';
        const changed = section.querySelector('[data-aee-modified="true"]');
        section.dataset.aeeDirty = changed ? 'true' : 'false';
        const notice = section.querySelector('[data-aee-settings-dirty]');
        if (notice) notice.hidden = !changed;
      }
      componentControl('aee-photo-change-button', 'button')
        ?.focus({preventScroll:true});
    }
    function openEditor(file){
      if (!file) return;
      const validType = ['image/jpeg', 'image/png', 'image/webp'].includes(file.type) ||
        (!file.type && /\.(jpe?g|png|webp)$/i.test(file.name));
      if (!validType || file.size <= 0 || file.size > maximumPhotoBytes){
        showError('Choose a JPG, PNG, or WebP photo smaller than 5 MB.');
        modal.hidden = false;
        document.body.classList.add('aee-photo-editor-open');
        return;
      }
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      objectUrl = URL.createObjectURL(file);
      modal.hidden = false;
      document.body.classList.add('aee-photo-editor-open');
      showError('');
      const nextImage = new Image();
      nextImage.onload = () => {
        if (nextImage.naturalWidth * nextImage.naturalHeight > 40000000){
          showError('This photo is too large to edit. Choose an image with fewer pixels.');
          return;
        }
        image = nextImage;
        zoom = 1;
        panX = 0;
        panY = 0;
        zoomRange.value = '1';
        document.querySelector('#aee-photo-apply-result')?.replaceChildren();
        showError('');
        requestAnimationFrame(draw);
        canvas.focus({preventScroll:true});
      };
      nextImage.onerror = () => showError('This image could not be opened. Choose another photo.');
      nextImage.src = objectUrl;
    }
    function applyCrop(){
      if (!image) return;
      const layout = geometry();
      constrainPan(layout);
      const left = layout.width / 2 -
        image.naturalWidth * layout.scale / 2 + panX;
      const top = layout.height / 2 -
        image.naturalHeight * layout.scale / 2 + panY;
      const sourceX = (layout.x - left) / layout.scale;
      const sourceY = (layout.y - top) / layout.scale;
      const sourceSize = layout.side / layout.scale;
      const output = document.createElement('canvas');
      output.width = 512;
      output.height = 512;
      output.getContext('2d').drawImage(image, sourceX, sourceY, sourceSize,
        sourceSize, 0, 0, output.width, output.height);
      output.toBlob(blob => {
        if (!blob || blob.size > maximumPhotoBytes){
          showError('The edited photo could not be prepared. Try a different crop.');
          return;
        }
        const reader = new FileReader();
        reader.onload = () => {
          const transfer = componentControl('aee-photo-payload', 'textarea, input');
          const submit = componentControl('aee-photo-submit', 'button');
          if (!transfer || !submit){
            showError('The photo editor controls are unavailable. Close the editor and reopen Settings.');
            return;
          }
          const valuePrototype = transfer instanceof HTMLTextAreaElement
            ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
          const setter = Object.getOwnPropertyDescriptor(valuePrototype, 'value').set;
          setter.call(transfer, reader.result);
          transfer.dispatchEvent(new Event('input', {bubbles:true}));
          transfer.dispatchEvent(new Event('change', {bubbles:true}));
          showError('');
          submit.click();
          closeEditor();
        };
        reader.onerror = () => showError('The edited photo could not be read. Please try again.');
        reader.readAsDataURL(blob);
      }, 'image/jpeg', .92);
    }
    openButton.addEventListener('click', event => {
      event.preventDefault();
      picker.click();
    });
    picker.addEventListener('change', () => openEditor(picker.files[0]));
    modal.addEventListener('click', event => {
      if (event.target.closest('[data-aee-photo-cancel]')){
        event.preventDefault();
        closeEditor();
      } else if (event.target.closest('[data-aee-photo-apply]')){
        event.preventDefault();
        applyCrop();
      } else {
        const zoomButton = event.target.closest('[data-aee-photo-zoom]');
        if (zoomButton){
          zoom = Math.max(1, Math.min(3, zoom + Number(zoomButton.dataset.aeePhotoZoom)));
          zoomRange.value = String(zoom);
          draw();
        }
      }
    });
    zoomRange.addEventListener('input', () => {
      zoom = Number(zoomRange.value);
      draw();
    });
    canvas.addEventListener('pointerdown', event => {
      if (!image) return;
      drag = {x:event.clientX, y:event.clientY, panX, panY};
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener('pointermove', event => {
      if (!drag) return;
      panX = drag.panX + event.clientX - drag.x;
      panY = drag.panY + event.clientY - drag.y;
      draw();
    });
    canvas.addEventListener('pointerup', () => { drag = null; });
    canvas.addEventListener('pointercancel', () => { drag = null; });
    canvas.addEventListener('keydown', event => {
      const step = event.shiftKey ? 24 : 8;
      if (event.key === 'ArrowLeft') panX -= step;
      else if (event.key === 'ArrowRight') panX += step;
      else if (event.key === 'ArrowUp') panY -= step;
      else if (event.key === 'ArrowDown') panY += step;
      else return;
      event.preventDefault();
      draw();
    });
    new ResizeObserver(draw).observe(canvas);
  }
  function refreshAeeProfileFieldBaseline(event){
    const field = event.target.closest(
      '.aee-settings-fields input, .aee-settings-fields select');
    const form = field && field.closest('#aee-settings-profile-section');
    if (!field || !form || form.dataset.aeeDirty === 'true') return;
    field.dataset.aeeInitialValue = field.value;
    delete field.dataset.aeeModified;
  }
  const AEE_ARABIC_TRANSLATIONS = Object.freeze({
    'English':'الإنجليزية','Arabic':'العربية',
    'Open navigation':'فتح قائمة التنقل','Close navigation':'إغلاق قائمة التنقل',
    'App navigation':'قائمة التطبيق','Project menu':'قائمة المشروع',
    'Workspace':'مساحة العمل','Back to workspace':'العودة إلى مساحة العمل',
    'Project':'المشروع','Account':'الحساب','account':'الحساب',
    'Detailed History':'السجل التفصيلي',
    'Updates':'التحديثات','About Us':'من نحن','Contact Us':'اتصل بنا',
    'AI-assisted video evidence review':'مراجعة أدلة الفيديو بمساعدة الذكاء الاصطناعي',
    'Human review remains essential.':'تظل المراجعة البشرية ضرورية.',
    'Video intelligence workspace':'منصة ذكية لتحليل الفيديو',
    'AI-assisted forensic video analysis':'تحليل جنائي للفيديو بمساعدة الذكاء الاصطناعي',
    'Review detections, follow object tracks, inspect model explanations, and export a complete evidence report from one workspace.':'راجع نتائج الاكتشاف، وتتبّع الأجسام، وافحص تفسيرات النماذج، وصدّر تقرير أدلة متكاملاً من مساحة عمل واحدة.',
    'Threat scoring':'تقييم مستوى التهديد','Object tracking':'تتبّع الأجسام',
    'Model explainability':'تفسير نتائج النماذج','Learned':'تعلّم آلي',
    'Agreement':'توافق النماذج','models':'نماذج','fusion':'دمج',
    'Analysis workflow':'خطوات التحليل','Add video':'إضافة فيديو',
    'Configure':'الإعدادات','Analyze':'تحليل','Review & export':'مراجعة وتصدير',
    'Evidence intake':'استيراد الأدلة','batch supported':'يدعم المعالجة الدفعية',
    'Fingerprint every byte before analysis':'تحقق من سلامة البيانات قبل التحليل',
    'READY':'جاهز','Supported video formats':'تنسيقات الفيديو المدعومة',
    'Supported formats':'التنسيقات المدعومة','total':'إجمالاً',
    'supported':'مدعوم','weapon':'سلاح',
    'Weapon':'سلاح','Firearm':'سلاح ناري','Gun':'سلاح ناري',
    'Blunt_Weapon':'سلاح غير ناري','Explosive':'مادة متفجرة',
    'Fire_Smoke':'حريق أو دخان','Melee_Weapon':'سلاح أبيض',
    'Person':'شخص','person':'شخص','face':'وجه','knife':'سكين',
    'pistol':'مسدس','Tool':'أداة','Violence':'عنف','Criminal':'مجرم',
    'Context':'سياق','classes':'فئات','conf':'الثقة','gate':'الحد',
    'Object Tracking':'تتبّع الأجسام','History':'السجل',
    'Analyzed':'تم التحليل','analysed':'تم التحليل','clips':'مقاطع',
    'sec':'ثانية','avg':'المتوسط',
    'No size limit · batch upload supported · content-hashed':'بلا حد للحجم · يدعم التحميل الدفعي · مع التحقق من سلامة الملفات',
    'Preparing upload…':'جارٍ تجهيز التحميل…','Connecting…':'جارٍ الاتصال…',
    'Preparing a secure upload.':'جارٍ إعداد تحميل آمن.',
    'Use test-time augmentation on supported models for difficult or small objects.':'استخدم تحسين وقت الاختبار مع النماذج المدعومة لاكتشاف الأجسام الصغيرة أو صعبة الرصد.',
    'Ensemble Configuration':'إعداد مجموعة النماذج',
    'active':'نشط','context':'سياق',
    'Inference device:':'جهاز الاستدلال:','Combined vocabulary':'الفئات المجمّعة',
    'Supported formats ·':'التنسيقات المدعومة ·',
    'download any clip below':'نزّل أي مقطع أدناه',
    'Download any clip below':'نزّل أي مقطع أدناه',
    'More tabs':'علامات تبويب إضافية','Empty value':'قيمة فارغة',
    'Thumbnail':'صورة مصغّرة','detections':'اكتشافات',
    'Duration':'المدة','Mode':'الوضع','Standard':'قياسي',
    'Case':'القضية','Not specified':'غير محدد','Original video':'الفيديو الأصلي',
    'Resolution':'الدقة','Frame rate':'معدل الإطارات',
    'Downloading':'جارٍ التنزيل','Preparing':'جارٍ التجهيز',
    'Display name':'الاسم المعروض','Personal information':'المعلومات الشخصية',
    'Update the profile details associated with your account.':'حدّث بيانات الملف الشخصي المرتبطة بحسابك.',
    'Unsaved changes · Save to keep your updates.':'توجد تغييرات غير محفوظة · احفظها للاحتفاظ بالتعديلات.',
    'Guest profile details stay on this computer. Signed-in profile details are saved to your Supabase account. Email changes require confirmation from your inbox.':'تُحفظ بيانات ملف الضيف على هذا الكمبيوتر. أما بيانات الملف المرتبط بحساب، فتُحفظ في حسابك على Supabase. يتطلب تغيير البريد الإلكتروني تأكيداً عبر صندوق الوارد.',
    'Security & account':'الأمان والحساب','security':'الأمان',
    'Manage how you sign in and connect your account.':'أدر طرق تسجيل الدخول وربط حسابك.',
    'Saved account session restored.':'تمت استعادة جلسة الحساب المحفوظة.',
    'Email changes require confirmation from your inbox.':'يتطلب تغيير البريد الإلكتروني تأكيداً عبر صندوق الوارد.',
    'Enhanced accuracy (slower)':'دقة محسّنة (أبطأ)',
    'Ready to analyze':'جاهز للتحليل',
    'Upload footage and press':'حمّل مقطع الفيديو واضغط على',
    'to extract weapon evidence with threat scoring, object identity, and EigenCAM explainability.':'لاستخراج أدلة الأسلحة وتقييم التهديدات وتتبع الأجسام وتفسير النتائج باستخدام EigenCAM.',
    'Metadata':'البيانات الوصفية','Audio':'الصوت','Detect':'اكتشاف',
    'Highlights':'المقاطع المهمة','Summaries':'الملخصات','Overall':'الإجمالي',
    'Frame':'الإطار','Sampled':'العينات','Throughput':'معدل المعالجة',
    'Detections':'الاكتشافات','detections':'اكتشافات',
    'Per-model inference time':'زمن الاستدلال لكل نموذج',
    'Recent analyzed videos':'مقاطع الفيديو التي جرى تحليلها مؤخراً',
    'A simple overview of completed analyses saved on this computer.':'عرض موجز للتحليلات المكتملة المحفوظة على هذا الجهاز.',
    'Preview summary':'معاينة الملخص','Filter by object, source, or video':'تصفية حسب الجسم أو المصدر أو الفيديو',
    'Video summaries':'ملخصات الفيديو',
    'Explainability':'تفسير النتائج','Evidence log':'سجل الأدلة',
    'Reports & exports':'التقارير والتصدير','Captures':'اللقطات',
    'Object track analytics':'تحليلات تتبّع الأجسام',
    'Per-track lifecycle, persistence, threat level, and re-identification events aggregated across all analysed videos.':'ملخص دورة حياة كل مسار واستمراريته ومستوى تهديده وأحداث إعادة التعرّف عبر جميع مقاطع الفيديو.',
    'Class-agnostic activation heatmaps per weapon detection. Heatmap-only and blended views are cached for every detection.':'خرائط تنشيط حرارية لكل اكتشاف سلاح، بصورتها الحرارية أو المدمجة، مع حفظ نسخة لكل اكتشاف.',
    'Master summary video':'الفيديو الموجز الرئيسي',
    'Union of every detection.':'يجمع جميع الاكتشافات.',
    'Forensic PDF report':'تقرير الأدلة الجنائية بصيغة PDF',
    'Evidence log + object-track threat analytics.':'سجل الأدلة وتحليلات تهديد مسارات الأجسام.',
    'ANALYSIS ARCHIVE':'أرشيف التحليلات',
    'Browse saved analyses, evidence files, detection records, and object tracks.':'استعرض التحليلات المحفوظة وملفات الأدلة وسجلات الاكتشاف ومسارات الأجسام.',
    'Download all history ZIP':'تنزيل السجل كاملاً بصيغة ZIP',
    'Save your profile':'حفظ الملف الشخصي',
    'Connect your Supabase project':'ربط مشروع Supabase',
    'Copy the project URL and public publishable key (or legacy anon key) from your Supabase project API settings. Never use a service-role or secret key. Run the setup SQL below in the Supabase SQL Editor. The connection is tested before these settings are saved, and the SQL is safe to rerun.':'انسخ رابط المشروع ومفتاح النشر العام (أو مفتاح anon القديم) من إعدادات واجهة API في مشروع Supabase. لا تستخدم مفتاح الخدمة أو أي مفتاح سري. نفّذ استعلام SQL أدناه في محرر SQL لدى Supabase. سيُختبر الاتصال قبل حفظ الإعدادات، ويمكن إعادة تشغيل الاستعلام بأمان.',
    'View setup instructions':'عرض تعليمات الإعداد',
    'Supabase project URL':'رابط مشروع Supabase',
    'Supabase public API key':'مفتاح Supabase العام لواجهة API',
    'Save and test connection':'حفظ الإعدادات واختبار الاتصال',
    'Run in Supabase SQL Editor':'التنفيذ في محرر SQL لدى Supabase',
    'Connection setup':'إعداد الاتصال','Account security':'أمان الحساب',
    'Profile details':'تفاصيل الملف الشخصي',
    'Your work stays private. Videos, analysis history, and reports remain on this device, whether you create an account or continue as a guest.':'تبقى بياناتك خاصة. تُحفظ مقاطع الفيديو وسجل التحليلات والتقارير على هذا الجهاز، سواء أنشأت حساباً أو تابعت كضيف.',
    'Online sign-in is not configured on this device yet. Continue as a guest, or ask your administrator to configure online accounts.':'لم يُعدّ تسجيل الدخول عبر الإنترنت على هذا الجهاز بعد. يمكنك المتابعة كضيف أو طلب إعداد الحسابات عبر الإنترنت من المسؤول.',
    'Built for careful review':'مصمم لمراجعة دقيقة',
    'Evidence deserves a clearer picture.':'تستحق الأدلة عرضاً أوضح.',
    'AI Evidence Extractor brings video analysis and evidence review together in one desktop workspace. Explore object detections, follow tracks, inspect model explanations, and prepare reports without losing sight of the source footage.':'يجمع AI Evidence Extractor تحليل الفيديو ومراجعة الأدلة في مساحة عمل واحدة. استعرض اكتشافات الأجسام ومساراتها وتفسيرات النماذج، وأعدّ التقارير مع الحفاظ على إمكانية الرجوع إلى الفيديو الأصلي.',
    'Project capabilities':'إمكانات المشروع','Video analysis':'تحليل الفيديو',
    'Review footage with an ensemble of detection models.':'راجع مقاطع الفيديو باستخدام مجموعة من نماذج الاكتشاف.',
    'Evidence review':'مراجعة الأدلة',
    'Explore captures, object tracks, and model explanations.':'استعرض اللقطات ومسارات الأجسام وتفسيرات النماذج.',
    'Case context':'سياق القضية',
    'Keep analysis history and case details close at hand.':'احتفظ بسجل التحليلات وتفاصيل القضية في متناولك.',
    'Report exports':'تصدير التقارير',
    'Prepare evidence summaries and downloadable reports.':'أعدّ ملخصات الأدلة والتقارير القابلة للتنزيل.',
    'Human review stays in control':'تظل المراجعة البشرية هي الأساس',
    'Detections, threat scores, and explanations are decision-support tools. Always verify findings against the original video.':'نتائج الاكتشاف وتقييمات التهديد والتفسيرات أدوات لدعم القرار. تحقّق دائماً من النتائج بالرجوع إلى الفيديو الأصلي.',
    'Contact the project team':'تواصل مع فريق المشروع',
    'Share questions, feature ideas, or bug reports through the project issue tracker. For a useful report, include the app version and steps to reproduce the issue. Do not post private videos, case details, or other sensitive evidence.':'أرسل أسئلتك واقتراحاتك أو بلاغات الأخطاء عبر متعقّب مشكلات المشروع. لتسهيل معالجة البلاغ، أرفق إصدار التطبيق وخطوات إعادة إظهار المشكلة. لا تنشر مقاطع خاصة أو تفاصيل القضايا أو أي أدلة حساسة.',
    'Open GitHub Issues':'فتح مشكلات GitHub',
    'Send us a message':'أرسل إلينا رسالة',
    'Name':'الاسم','Message':'الرسالة','Send message':'إرسال الرسالة',
    'Download Master MP4':'تنزيل الفيديو الرئيسي بصيغة MP4',
    'Download PDF Report':'تنزيل التقرير بصيغة PDF',
    'Track detail':'تفاصيل التتبع','Activation heatmaps':'خرائط التنشيط الحرارية',
    'Open profile settings':'فتح إعدادات الملف الشخصي',
    'Profile settings':'إعدادات الملف الشخصي','Update the app':'تحديث التطبيق',
    'Open app updates':'فتح تحديثات التطبيق','Toggle theme (T)':'تبديل السمة (T)',
    'Command palette (Ctrl+K)':'لوحة الأوامر (Ctrl+K)',
    'Type a command or jump to a tab…':'اكتب أمراً أو انتقل إلى علامة تبويب…',
    'Video upload progress':'تقدم تحميل الفيديو',
    'Edit profile photo':'تعديل صورة الملف الشخصي',
    'Cancel photo edit':'إلغاء تعديل الصورة',
    'Drag to position the photo inside the square crop':'اسحب الصورة لضبط موضعها داخل إطار القص المربع',
    'Photo zoom':'تكبير الصورة','Zoom out':'تصغير','Zoom photo':'تكبير الصورة',
    'Zoom in':'تكبير','Choose a profile photo':'اختيار صورة للملف الشخصي',
    'Settings sections':'أقسام الإعدادات',
    'Overview':'نظرة عامة','Evidence':'الأدلة','Analysis':'التحليل',
    'Upload video':'تحميل فيديو','Upload videos':'تحميل مقاطع الفيديو',
    'Reset':'إعادة ضبط','← Reset':'إعادة ضبط ←',
    'Run analysis':'بدء التحليل','⚡ Run analysis':'⚡ بدء التحليل',
    'High accuracy':'دقة عالية',
    'Search':'بحث','Class':'الفئة','All':'الكل',
    'Export tracks CSV':'تصدير مسارات التتبع بصيغة CSV',
    'Export detections CSV':'تصدير النتائج بصيغة CSV',
    'Export detections JSON':'تصدير النتائج بصيغة JSON',
    'Refresh history':'تحديث السجل',
    'Clear all history':'مسح السجل بالكامل','Details':'التفاصيل',
    'Download ZIP':'تنزيل ZIP','Remove':'إزالة','Captures and CAM evidence':'اللقطات وأدلة خرائط التنشيط',
    'Reports and generated files':'التقارير والملفات المُنشأة',
    'Detection records':'سجلات الاكتشاف','Complete saved detections for the selected analysis.':'كافة الاكتشافات المحفوظة للتحليل المحدد.',
    'Object tracks':'مسارات الأجسام','Saved per-object tracking and threat analytics.':'بيانات التتبع وتحليل المخاطر المحفوظة لكل جسم.',
    'Your analysis archive':'أرشيف التحليلات','Browse your analyzed videos in order. Open details, download an analysis ZIP, or remove an item from this computer.':'استعرض مقاطع الفيديو التي جرى تحليلها. افتح التفاصيل، أو نزّل التحليل بصيغة ZIP، أو أزل عنصراً من هذا الجهاز.',
    'No saved analyses yet':'لا توجد تحليلات محفوظة بعد',
    'Completed video analyses will appear here.':'ستظهر هنا تحليلات الفيديو المكتملة.',
    'Select an analysis':'اختر تحليلاً','Analysis not found':'لم يتم العثور على التحليل',
    'Case notes:':'ملاحظات القضية:','ZIP export failed':'تعذر تصدير ملف ZIP',
    'ZIP ready:':'ملف ZIP جاهز:','Analysis removed':'تمت إزالة التحليل',
    'Could not remove analysis':'تعذرت إزالة التحليل',
    'Some generated files need cleanup':'تحتاج بعض الملفات المُنشأة إلى التنظيف',
    'Could not clear history':'تعذر مسح السجل','History cleared':'تم مسح السجل',
    'Account center':'مركز الحساب','ACCOUNT CENTER':'مركز الحساب',
    'Manage your profile, sign-in, and privacy.':'أدر ملفك الشخصي وتسجيل الدخول وإعدادات الخصوصية.',
    'Back to app':'العودة إلى التطبيق','Your name':'اسمك',
    'Enter your name':'أدخل اسمك','Email address':'عنوان البريد الإلكتروني',
    'Password':'كلمة المرور','Enter your password':'أدخل كلمة المرور',
    'you@example.com':'you@example.com',
    'Create my account':'إنشاء حسابي','Log in':'تسجيل الدخول',
    'Already have an account? Log in':'لديك حساب بالفعل؟ سجّل الدخول',
    'or continue with':'أو المتابعة باستخدام','Continue with Google':'المتابعة باستخدام Google',
    'or':'أو','Continue as a guest':'المتابعة كضيف',
    'Your work stays private.':'تبقى بيانات عملك خاصة.',
    'Local profile':'ملف شخصي محلي','Your profile stays on this device':'يبقى ملفك الشخصي على هذا الجهاز',
    'GUEST · LOCAL ONLY':'ضيف · محلي فقط','Account profile':'الملف الشخصي للحساب',
    'Member since':'عضو منذ','SIGNED IN WITH GOOGLE':'تم تسجيل الدخول باستخدام Google',
    'Save changes':'حفظ التغييرات','Save profile':'حفظ الملف الشخصي',
    'Refresh profile':'تحديث الملف الشخصي','Change photo':'تغيير الصورة',
    'Apply photo':'تطبيق الصورة','Sign out':'تسجيل الخروج',
    'Continue with another account':'المتابعة بحساب آخر',
    'Security':'الأمان','Language':'اللغة','Choose language':'اختر اللغة',
    'Settings':'الإعدادات','Profile':'الملف الشخصي',
    'Update center':'مركز التحديثات','APP CENTER':'مركز التطبيق',
    'Check for the latest improvements and install updates.':'تحقق من أحدث التحسينات وثبّت التحديثات.',
    'Check for updates':'التحقق من وجود تحديثات','Check for Updates':'التحقق من وجود تحديثات',
    'Download update':'تنزيل التحديث','Download Update':'تنزيل التحديث',
    'Install update':'تثبيت التحديث','Install Update':'تثبيت التحديث',
    'Restart app':'إعادة تشغيل التطبيق','Restart App':'إعادة تشغيل التطبيق',
    'The update is downloaded, checksum-verified, and ready to apply. Select Install Update to see the in-app installation progress.':'تم تنزيل التحديث والتحقق من سلامته وأصبح جاهزاً للتثبيت. اختر «تثبيت التحديث» لمتابعة التقدم داخل التطبيق.',
    'Could not reach the update service. Check your internet connection and try again.':'تعذر الاتصال بخدمة التحديث. تحقق من اتصال الإنترنت وحاول مجدداً.',
    'A smoother way to stay current.':'ابقَ على اطلاع بأحدث إصدار بسهولة.',
    'Check for a release, review it, and update without opening a setup file. The download and each installation step show live progress in this window.':'تحقق من إصدار جديد وراجعه وثبّته دون فتح ملف إعداد. ستظهر هنا مباشرةً مراحل التنزيل والتثبيت.',
    'Check for a release, review it, and update without opening a setup file.':'تحقق من إصدار جديد وراجعه وثبّته دون فتح ملف إعداد.',
    'The download and each installation step show live progress in this window.':'ستظهر هنا مباشرةً مراحل التنزيل والتثبيت.',
    'Installed version':'الإصدار المثبّت','How app updates work':'كيفية تحديث التطبيق',
    'Check':'تحقق','Look for the latest release securely.':'ابحث بأمان عن أحدث إصدار.',
    'Download':'تنزيل','Track download and checksum verification in the app.':'تابع التنزيل والتحقق من سلامة الملف داخل التطبيق.',
    'Apply':'تطبيق','Watch file-by-file progress, then restart to load the update.':'تابع تقدم تحديث الملفات، ثم أعد التشغيل لتفعيل التحديث.',
    'THE PROJECT':'عن المشروع','Learn about the workspace and how it supports evidence review.':'تعرّف على مساحة العمل وكيفية دعمها لمراجعة الأدلة.',
    'SUPPORT':'الدعم','Get help, share feedback, or report an issue.':'احصل على المساعدة أو أرسل ملاحظاتك أو أبلغ عن مشكلة.',
    'Video evidence analysis':'تحليل أدلة الفيديو','Videos':'مقاطع الفيديو',
    'Weapon dets':'اكتشافات الأسلحة','Context hits':'المؤشرات السياقية',
    'Elapsed':'الوقت المنقضي',
    'Threat Distribution':'توزيع مستوى التهديد','Top-risk detections':'الاكتشافات الأعلى خطورة',
    'critical':'حرج','high':'مرتفع','medium':'متوسط','low':'منخفض',
    'Case number':'رقم القضية','Evidence ID':'معرّف الدليل',
    'Examiner':'الفاحص','Agency':'الجهة','Case notes':'ملاحظات القضية',
    'Export detections':'تصدير الاكتشافات','Export tracks':'تصدير المسارات',
    'Refresh':'تحديث','Cancel':'إلغاء',
    'Threat analysis':'تحليل التهديدات',
    'Download report':'تنزيل التقرير','Download PDF':'تنزيل PDF',
    'No results yet':'لا توجد نتائج بعد','Processing':'جارٍ المعالجة',
    'Analysis completed':'اكتمل التحليل','Analysis stopped.':'تم إيقاف التحليل.',
    'Apply the update':'تطبيق التحديث',
    'Installing update':'جارٍ تثبيت التحديث',
    'Signed in with Google.':'تم تسجيل الدخول باستخدام Google.',
    'Language preferences':'تفضيلات اللغة','Choose a language':'اختر لغة',
    'Select video files':'اختر ملفات الفيديو','Reports':'التقارير',
    'Status':'الحالة','Date':'التاريخ','Type':'النوع','Object':'الجسم',
    'Timestamp':'الطابع الزمني','Confidence':'مستوى الثقة','Risk':'مستوى الخطورة',
    'Select another saved video to inspect its evidence and reports.':'اختر فيديو محفوظاً آخر لمراجعة أدلته وتقاريره.',
    'Some files were not saved':'لم يتم حفظ بعض الملفات',
    'analysis included.':'تحليل مُضمّن.',
    'Analysis removed.':'تمت إزالة التحليل.','saved analysis(es) cleared.':'تم مسح التحليلات المحفوظة.',
    'Could not start the update:':'تعذر بدء التحديث:',
    'Applying update files…':'جارٍ تطبيق ملفات التحديث…',
    'Update':'تحديث','installed successfully.':'تم التثبيت بنجاح.',
    'Restarting AI Evidence Extractor…':'جارٍ إعادة تشغيل AI Evidence Extractor…',
    'Signed-in account':'الحساب المسجّل',
    'Guest account':'حساب ضيف',
    'This profile is stored on this device only.':'يُحفظ هذا الملف الشخصي على هذا الجهاز فقط.',
    'Email':'البريد الإلكتروني','Username':'اسم المستخدم',
    'Save':'حفظ','Save profile changes':'حفظ تغييرات الملف الشخصي',
    'Update profile':'تحديث الملف الشخصي','Change account':'تغيير الحساب',
    'Sign in':'تسجيل الدخول','Sign up':'إنشاء حساب',
    'New password':'كلمة مرور جديدة','Confirm password':'تأكيد كلمة المرور',
    'Project URL':'رابط المشروع','Anonymous key':'المفتاح العام',
    'Save connection settings':'حفظ إعدادات الاتصال',
    'Account settings':'إعدادات الحساب',
    'Profile photo':'صورة الملف الشخصي','Local only':'محلي فقط',
    'Connection settings':'إعدادات الاتصال','Google sign-in':'تسجيل الدخول عبر Google',
    'Sign in with Google':'تسجيل الدخول باستخدام Google',
    'Don’t have an account? Create one':'ليس لديك حساب؟ أنشئ حساباً',
    'weapon hits':'حالات اكتشاف أسلحة','avg risk':'متوسط الخطورة',
    'click any bar to seek':'انقر على أي شريط للانتقال',
    'blue = peak activity':'الأزرق = ذروة النشاط',
    'Weapon Detection Timeline':'الجدول الزمني لاكتشاف الأسلحة',
    'Detection details':'تفاصيل الاكتشاف','Threat level':'مستوى التهديد',
    'No detections found':'لم يتم العثور على اكتشافات',
    'Analysis complete':'اكتمل التحليل','No video selected':'لم يتم اختيار فيديو',
    'Choose files':'اختر الملفات','Export':'تصدير',
    'Photo':'الصورة','Apply crop':'تطبيق القص','Reset crop':'إعادة ضبط القص',
    'Jan':'يناير','January':'يناير','Feb':'فبراير','February':'فبراير',
    'Mar':'مارس','March':'مارس','Apr':'أبريل','April':'أبريل',
    'May':'مايو','Jun':'يونيو','June':'يونيو','Jul':'يوليو','July':'يوليو',
    'Aug':'أغسطس','August':'أغسطس','Sep':'سبتمبر','September':'سبتمبر',
    'Oct':'أكتوبر','October':'أكتوبر','Nov':'نوفمبر','November':'نوفمبر',
    'Dec':'ديسمبر','December':'ديسمبر'
  });
  const AEE_ARABIC_TRANSLATION_ENTRIES = Object.entries(AEE_ARABIC_TRANSLATIONS)
    .filter(([source]) => source.includes(' ') || source.length >= 3)
    .sort(([left], [right]) => right.length - left.length);
  const aeeTranslatedText = new WeakMap();
  const aeeTranslatedAttributes = new WeakMap();
  let aeeArabicObserver;
  function aeeTranslateValue(value, arabic){
    if (!arabic) return value;
    const leading = value.match(/^\s*/)[0];
    const trailing = value.match(/\s*$/)[0];
    const trimmed = value.trim().replace(/\s+/g, ' ');
    const exact = AEE_ARABIC_TRANSLATIONS[trimmed];
    if (exact) return leading + exact + trailing;
    const brandMarker = '\uE000';
    let translated = trimmed.split('AI Evidence Extractor').join(brandMarker);
    AEE_ARABIC_TRANSLATION_ENTRIES.forEach(([source, target]) => {
      if (source.includes(' ')){
        translated = translated.split(source).join(target);
      } else {
        const escaped = source.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        translated = translated.replace(
          new RegExp(`\\b${escaped}\\b`, 'g'), target);
      }
    });
    return leading + translated.split(brandMarker).join('AI Evidence Extractor') + trailing;
  }
  function aeeTranslateSubtree(root, arabic){
    if (!root) return;
    const skipSelector = 'script,style,code,pre,textarea,[data-aee-no-translate],.aee-history-row-title';
    const translateText = node => {
      if (!node || !node.nodeValue || !node.nodeValue.trim() ||
          /(?:[A-Za-z]:\\|https?:\/\/)/.test(node.nodeValue) ||
          (node.parentElement && node.parentElement.closest(skipSelector))) return;
      const previous = aeeTranslatedText.get(node);
      if (previous && previous.value === node.nodeValue && previous.arabic === arabic) return;
      const original = previous && previous.value === node.nodeValue ?
        previous.original : node.nodeValue;
      const translated = aeeTranslateValue(original, arabic);
      if (translated !== node.nodeValue){
        node.nodeValue = translated;
      }
      aeeTranslatedText.set(node, {value:translated, arabic, original});
    };
    const translateAttributes = element => {
      if (!element || element.nodeType !== Node.ELEMENT_NODE ||
          element.matches(skipSelector)) return;
      const names = ['aria-label','placeholder','title','alt'];
      let previous = aeeTranslatedAttributes.get(element);
      if (!previous){ previous = {}; aeeTranslatedAttributes.set(element, previous); }
      names.forEach(name => {
        if (!element.hasAttribute(name)) return;
        const value = element.getAttribute(name);
        if (previous[name] && previous[name].value === value &&
            previous[name].arabic === arabic) return;
        const original = previous[name] && previous[name].value === value ?
          previous[name].original : value;
        const translated = aeeTranslateValue(original, arabic);
        if (translated !== value){
          element.setAttribute(name, translated);
        }
        previous[name] = {value:translated, arabic, original};
      });
    };
    if (root.nodeType === Node.TEXT_NODE) translateText(root);
    else if (root.nodeType === Node.ELEMENT_NODE) translateAttributes(root);
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    let node;
    while ((node = walker.nextNode())) translateText(node);
    if (root.querySelectorAll){
      root.querySelectorAll('*').forEach(translateAttributes);
    }
  }
  function applyAeeLanguage(language, persist){
    const arabic = language === 'Arabic';
    document.documentElement.lang = arabic ? 'ar' : 'en';
    document.documentElement.dir = arabic ? 'rtl' : 'ltr';
    aeeTranslateSubtree(document.documentElement, arabic);
    if (!aeeArabicObserver){
      aeeArabicObserver = new MutationObserver(records => {
        const activeArabic = document.documentElement.lang === 'ar';
        records.forEach(record => {
          if (record.type === 'characterData'){
            aeeTranslateSubtree(record.target, activeArabic);
          } else if (record.type === 'attributes'){
            aeeTranslateSubtree(record.target, activeArabic);
          } else {
            record.addedNodes.forEach(node => aeeTranslateSubtree(node, activeArabic));
          }
        });
      });
      aeeArabicObserver.observe(document.documentElement, {
        subtree:true,childList:true,characterData:true,attributes:true,
        attributeFilter:['aria-label','placeholder','title','alt']
      });
    }
    if (persist){
      try { localStorage.setItem('aee-language', arabic ? 'Arabic' : 'English'); }
      catch (error) { console.warn('Could not save language preference.', error); }
    }
  }
  function initializeAeeLanguageSelector(){
    const languageSelector = document.getElementById('aee-language-selector');
    if (!languageSelector) return;
    const languageDropdown = languageSelector.closest('.aee-settings-language-control');
    const languageOptions = languageDropdown
      ? Array.from(languageDropdown.querySelectorAll('[data-aee-language-option]')) : [];
    function syncLanguageOptions(){
      languageOptions.forEach(option => {
        option.setAttribute('aria-pressed',
          option.dataset.aeeLanguageOption === languageSelector.value ? 'true' : 'false');
      });
    }
    let savedLanguage = 'English';
    try { savedLanguage = localStorage.getItem('aee-language') || 'English'; }
    catch (error) { console.warn('Could not read language preference.', error); }
    if (!languageSelector.dataset.aeeLanguageInitialized){
      languageSelector.value = savedLanguage === 'Arabic' ? 'Arabic' : 'English';
      languageSelector.dataset.aeeLanguageInitialized = 'true';
      languageSelector.addEventListener('change', () => {
        syncLanguageOptions();
        applyAeeLanguage(languageSelector.value, true);
      });
      languageOptions.forEach(option => option.addEventListener('click', () => {
        if (!['English', 'Arabic'].includes(option.dataset.aeeLanguageOption)) return;
        languageSelector.value = option.dataset.aeeLanguageOption;
        languageSelector.dispatchEvent(new Event('change', {bubbles: true}));
        if (languageDropdown) languageDropdown.open = false;
      }));
      if (languageDropdown) languageDropdown.addEventListener('keydown', event => {
        if (event.key === 'Escape' && languageDropdown.open){
          languageDropdown.open = false;
          languageDropdown.querySelector('summary').focus();
        }
      });
    }
    syncLanguageOptions();
    applyAeeLanguage(languageSelector.value, false);
  }
  function initializeAeeProfileFieldValues(){
      document.querySelectorAll(
        '.aee-settings-fields input, .aee-settings-fields select')
        .forEach(field => {
          if (!Object.prototype.hasOwnProperty.call(field.dataset, 'aeeInitialValue'))
            field.dataset.aeeInitialValue = field.value;
        });
  }
  initializeAeeProfileFieldValues();
  document.addEventListener('focusin', refreshAeeProfileFieldBaseline, true);
  document.addEventListener('input', markAeeProfileFieldChanges, true);
  document.addEventListener('change', markAeeProfileFieldChanges, true);
  function clearAeeProfileFieldChanges(){
    const statuses = document.querySelectorAll(
      '[id="aee-account-status"] [data-profile-saved="true"]');
    if (!statuses.length) return;
    document.querySelectorAll('[id="aee-settings-profile-section"]').forEach(form => {
      delete form.dataset.aeeDirty;
      delete form.dataset.aeePhotoDirty;
    });
    document.querySelectorAll('#aee-settings-profile-section [data-aee-modified]').forEach(field => {
      delete field.dataset.aeeModified;
      delete field.dataset.aeeInitialValue;
    });
    document.querySelectorAll('[data-aee-settings-dirty]').forEach(notice => {
      notice.hidden = true;
    });
    statuses.forEach(status => status.removeAttribute('data-profile-saved'));
  }
  new MutationObserver(clearAeeProfileFieldChanges).observe(document.documentElement,
    {childList:true,subtree:true});
  function liftAeeNavigation(){
    applyAeeAccessGate();
    initializeAeeLanguageSelector();
    initializeAeePhotoEditor();
    syncAeeProfileTool();
    syncAeeTopbarLayout();
    syncAeeUpdateShortcut();
    initializeAeeProfileFieldValues();
    const host = document.getElementById('aee-nav-host');
    const navigation = host && host.querySelector('.aee-navigation');
    if (navigation && navigation.parentElement !== document.body){
      document.body.appendChild(navigation);
      host.style.display = 'none';
    }
    const startupMarker = document.querySelector('[data-aee-account-startup]');
    if (startupMarker && startupMarker.dataset.aeeAccountStartup === 'required' &&
        !window.__aeeInitialAccountOpened){
      window.__aeeInitialAccountOpened = true;
      window.setTimeout(() => activateAeeUtilityWindow('account'), 0);
    }
  }
  document.addEventListener('click', event => {
    if (event.target.closest('[data-aee-nav-toggle]')){
      const isOpen = document.body && document.body.classList.contains('aee-nav-open');
      setAeeNavigation(!isOpen); return;
    }
    if (event.target.closest('[data-aee-open-settings]')){
      activateAeeUtilityWindow('account'); return;
    }
    if (event.target.closest('[data-aee-nav-update]')){
      activateAeeUtilityWindow('updates'); return;
    }
    if (event.target.closest('[data-aee-nav-close]') || event.target.closest('[data-aee-nav-backdrop]')){
      setAeeNavigation(false); return;
    }
    if (event.target.classList.contains('aee-utility-window-shell')){
      closeAeeUtilityWindow(event.target); return;
    }
    const closeButton = event.target.closest('.aee-utility-window-close');
    if (closeButton){
      closeAeeUtilityWindow(closeButton.closest('.aee-utility-window-shell')); return;
    }
    const item = event.target.closest('[data-aee-nav-tab]');
    if (item){
      const name = item.getAttribute('data-aee-nav-tab');
      if (['Account','Settings','Updates','About Us','Contact Us','Detailed History'].includes(name)){
        const windowName = ['Account', 'Settings'].includes(name) ? 'account' :
          name.toLowerCase().replace(/\s+/g, '-');
        activateAeeUtilityWindow(windowName); return;
      }
      const tab = Array.from(document.querySelectorAll('.aee-tabs [role="tab"]'))
        .find(candidate => candidate.textContent.trim() === name);
      if (tab) tab.click();
      setAeeNavigation(false);
    }
    const settingsItem = event.target.closest('[data-aee-settings-section]');
    if (settingsItem){
      const destination = settingsItem.dataset.aeeSettingsSection;
      const section = document.getElementById('aee-settings-' + destination + '-section');
      if (section) section.scrollIntoView({behavior:'smooth', block:'start'});
      document.querySelectorAll('.aee-settings-nav [data-aee-settings-section]')
        .forEach(button => button.removeAttribute('aria-current'));
      settingsItem.setAttribute('aria-current', 'page');
    }
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape'){
      const photoEditor = document.getElementById('aee-photo-editor-modal');
      if (photoEditor && !photoEditor.hidden){
        photoEditor.querySelector('[data-aee-photo-cancel]')?.click();
        return;
      }
      const windowShell = document.querySelector('.aee-utility-window-shell[aria-modal="true"]');
      if (windowShell) closeAeeUtilityWindow(windowShell);
      else setAeeNavigation(false);
    }
  });
  liftAeeNavigation();
  new MutationObserver(liftAeeNavigation).observe(document.documentElement,
    {childList:true,subtree:true});

  function toast(msg, type, ms){
    type = type || 'info'; ms = ms || 2800;
    const wrap = document.getElementById('aee-toast-wrap');
    if (!wrap) return;
    const icons = {info:'ℹ️',success:'✅',warn:'⚠️',error:'⛔'};
    const el = document.createElement('div');
    el.className = 'aee-toast ' + type;
    el.innerHTML = '<span class="ico">' + (icons[type]||'ℹ️') + '</span>' +
                   '<span>' + msg + '</span>';
    wrap.appendChild(el);
    requestAnimationFrame(() => el.classList.add('show'));
    setTimeout(() => {
      el.classList.remove('show');
      setTimeout(() => el.remove(), 400);
    }, ms);
  }
  window.aeeToast = toast;

  function fmtBytes(n){
    if (!n || n < 0) return '—';
    const u = ['B','KB','MB','GB','TB'];
    let i = 0;
    while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
    return n.toFixed(n < 10 ? 1 : 0) + ' ' + u[i];
  }

  /* ============ UPLOAD PROGRESS ============ */
  const _PANEL_HTML =
    '<div class="aee-up-progress-head">' +
      '<div class="aee-up-progress-label">' +
        '<div class="aee-up-progress-spinner"></div>' +
        '<span class="aee-up-progress-text">Preparing upload…</span>' +
      '</div>' +
      '<div class="aee-up-progress-size">—</div>' +
    '</div>' +
    '<div class="aee-up-progress-track" role="progressbar" ' +
      'aria-label="Video upload progress" aria-valuemin="0" aria-valuemax="100">' +
      '<div class="aee-up-progress-bar" style="width:0%"></div>' +
    '</div>' +
    '<div class="aee-up-progress-hint">' +
      '<span class="aee-up-progress-pct">Connecting…</span>' +
      '<span class="aee-up-progress-detail">Preparing a secure upload.</span>' +
    '</div>';

  let _upTotal = 0, _upLoaded = 0, _upPending = false, _upInflight = 0;
  let _upBaseNode = null, _upBaseText = '', _upBaseStatus = '';
  let _upBaseStatusMarker = '';
  let _upSignature = '', _upFileLabel = '', _upRun = 0, _upStartedAt = 0;
  let _upSafety = null, _upHide = null, _upCancelTimer = null;

  function _createUploadPanel(){
    let el = document.getElementById('aee-up-progress');
    if (el && el.isConnected) return el;
    el = document.createElement('div');
    el.id = 'aee-up-progress';
    el.className = 'aee-up-progress';
    el.setAttribute('role', 'status');
    el.setAttribute('aria-live', 'polite');
    el.setAttribute('aria-atomic', 'true');
    el.innerHTML = _PANEL_HTML;
    const host = document.getElementById('aee-upload-progress-host');
    const shell = document.querySelector('.aee-upload-shell');
    if (host){
      host.appendChild(el);
    } else if (shell){
      const fmt = shell.querySelector('.aee-formats');
      if (fmt) fmt.insertAdjacentElement('afterend', el);
      else shell.appendChild(el);
    } else {
      document.body.appendChild(el);
    }
    LOG('panel created in', host ? 'formats-adjacent slot' : (shell ? 'upload shell fallback' : 'body fallback'));
    return el;
  }

  function _initCreate(){
    try {
      if (!document.querySelector('.aee-upload-shell')){
        setTimeout(_initCreate, 400);
        return;
      }
      _createUploadPanel();
      LOG('panel pre-created OK');
    } catch(e){ LOG('pre-create err', e); }
  }
  if (document.readyState === 'loading')
    document.addEventListener('DOMContentLoaded', _initCreate);
  else _initCreate();

  function _panel(){
    let el = document.getElementById('aee-up-progress');
    if (!el || !el.isConnected) el = _createUploadPanel();
    return el;
  }
  const _q = (el, s) => el.querySelector(s);

  function _show(){
    const el = _panel();
    if (_upHide){ clearTimeout(_upHide); _upHide = null; }
    el.classList.add('active', 'aee-visible');
    el.style.display = 'block';
    return el;
  }
  function _setMode(mode){
    const el = _panel();
    el.classList.remove('indeterminate', 'complete', 'error');
    if (mode) el.classList.add(mode);
    return el;
  }
  function _formatTime(seconds){
    if (!isFinite(seconds) || seconds < 0) return '';
    if (seconds < 60) return Math.ceil(seconds) + ' sec';
    const mins = Math.floor(seconds / 60);
    const secs = Math.ceil(seconds % 60);
    return mins + ' min ' + String(secs).padStart(2, '0') + ' sec';
  }
  function _setIndeterminate(label, detail){
    const el = _show();
    _setMode('indeterminate');
    _q(el,'.aee-up-progress-text').textContent = label || 'Uploading video…';
    _q(el,'.aee-up-progress-pct').textContent = 'In progress';
    _q(el,'.aee-up-progress-detail').textContent =
      detail || _upFileLabel || 'Sending video to the server.';
    _q(el,'.aee-up-progress-track').removeAttribute('aria-valuenow');
    _q(el,'.aee-up-progress-bar').style.width = '34%';
    _q(el,'.aee-up-progress-size').textContent =
      _upTotal > 0 ? (fmtBytes(_upTotal) + ' total') : 'Starting…';
  }
  function _paint(pct, loaded, total){
    const el = _show();
    _setMode('');
    const c = Math.max(0, Math.min(100, pct));
    _q(el,'.aee-up-progress-bar').style.width = c.toFixed(2) + '%';
    _q(el,'.aee-up-progress-pct').textContent = Math.round(c) + '%';
    _q(el,'.aee-up-progress-track').setAttribute('aria-valuenow',
      String(Math.round(c)));
    const s = _q(el,'.aee-up-progress-size');
    if (total > 0) s.textContent = fmtBytes(loaded) + ' / ' + fmtBytes(total);
    const elapsed = Math.max(0.25, (Date.now() - _upStartedAt) / 1000);
    const rate = loaded > 0 ? loaded / elapsed : 0;
    let detail = rate > 0 ? (fmtBytes(rate) + '/s') : 'Transfer progress received';
    if (rate > 0 && total > loaded)
      detail += ' · about ' + _formatTime((total - loaded) / rate) + ' left';
    _q(el,'.aee-up-progress-detail').textContent = detail;
  }
  function _hideUploadPanel(){
    const el = document.getElementById('aee-up-progress');
    if (!el) return;
    el.classList.remove('active', 'aee-visible', 'indeterminate',
                        'complete', 'error');
    el.style.display = 'none';
    LOG('panel hidden');
  }
  window.aeeHideUploadProgress = _hideUploadPanel;

  function _clearTimers(){
    if (_upSafety){ clearTimeout(_upSafety); _upSafety = null; }
    if (_upCancelTimer){ clearTimeout(_upCancelTimer); _upCancelTimer = null; }
    if (_upHide){ clearTimeout(_upHide); _upHide = null; }
  }

  function _begin(files, fromClick){
    files = files ? Array.from(files) : [];
    const signature = files.map(f =>
      [f.name || '', f.size || 0, f.lastModified || 0].join(':')).join('|');
    if (_upPending && signature && signature === _upSignature) return;
    if (_upPending && !signature && fromClick && !_upSignature) return;
    _clearTimers();
    _upRun++;
    _upTotal = files.reduce((sum, f) => sum + (Number(f.size) || 0), 0);
    _upLoaded = 0;
    _upInflight = 0;
    _upPending = true;
    _upSignature = signature;
    _upStartedAt = Date.now();
    _upFileLabel = files.length === 1
      ? (files[0].name || 'Selected video')
      : (files.length > 1
          ? ((files[0].name || 'Video') + ' + ' + (files.length - 1) + ' more')
          : '');
    _upBaseNode = document.querySelector('.aee-file-stats');
    _upBaseText = _upBaseNode ? _upBaseNode.textContent : '';
    const statusNode = document.getElementById('aee-analysis-progress');
    _upBaseStatus = statusNode ? statusNode.textContent : '';
    const statusMarker = statusNode
      ? statusNode.querySelector('[data-aee-upload-ready]') : null;
    _upBaseStatusMarker = statusMarker
      ? statusMarker.getAttribute('data-aee-upload-ready') : '';
    const label = fromClick && !files.length ? 'Choose a video to upload…'
      : (files.length > 1 ? ('Uploading ' + files.length + ' videos…')
          : (files.length === 1 ? 'Uploading video…' : 'Connecting to upload…'));
    _setIndeterminate(label,
      fromClick && !files.length
        ? 'Select a video, or close the picker to cancel.'
        : (_upFileLabel || 'Sending video to the server.'));
    if (fromClick && files.length === 0){
      _upCancelTimer = setTimeout(() => {
        if (_upPending && !_upSignature){
          _upPending = false;
          _clearTimers();
          _hideUploadPanel();
        }
      }, 60000);
    }
    LOG('begin', {nFiles: files.length, total: _upTotal, fromClick: fromClick});
  }

  function _complete(){
    if (!_upPending) return;
    _upPending = false;
    _clearTimers();
    const el = _show();
    _setMode('complete');
    _q(el,'.aee-up-progress-text').textContent = 'Video ready';
    _q(el,'.aee-up-progress-pct').textContent = 'Complete';
    _q(el,'.aee-up-progress-detail').textContent =
      'Upload confirmed · ready for analysis.';
    _q(el,'.aee-up-progress-bar').style.width = '100%';
    _q(el,'.aee-up-progress-track').setAttribute('aria-valuenow', '100');
    _q(el,'.aee-up-progress-size').textContent =
      _upTotal > 0 ? fmtBytes(_upTotal) + ' uploaded' : 'Ready';
    _upHide = setTimeout(_hideUploadPanel, 3200);
    LOG('complete');
  }

  function _fail(message, detail){
    if (!_upPending) return;
    _upPending = false;
    _clearTimers();
    const el = _show();
    _setMode('error');
    _q(el,'.aee-up-progress-text').textContent = message || 'Upload failed';
    _q(el,'.aee-up-progress-pct').textContent = 'Try again';
    _q(el,'.aee-up-progress-detail').textContent =
      detail || 'Check your connection, then choose the video again.';
    _q(el,'.aee-up-progress-track').removeAttribute('aria-valuenow');
    _q(el,'.aee-up-progress-size').textContent =
      _upTotal > 0 ? fmtBytes(_upTotal) + ' not confirmed' : 'Not confirmed';
    LOG('failed', message);
  }

  function _finalizing(run){
    if (!_upPending || run !== _upRun) return;
    const el = _show();
    _setMode('');
    _q(el,'.aee-up-progress-text').textContent = 'Checking uploaded video…';
    _q(el,'.aee-up-progress-pct').textContent = 'Transfer complete';
    _q(el,'.aee-up-progress-detail').textContent =
      'Waiting for the app to confirm the video details.';
    _q(el,'.aee-up-progress-bar').style.width = '100%';
    _q(el,'.aee-up-progress-track').setAttribute('aria-valuenow', '100');
    _q(el,'.aee-up-progress-size').textContent =
      _upTotal > 0 ? fmtBytes(_upTotal) + ' received' : 'Received';
    if (_upSafety) clearTimeout(_upSafety);
    _upSafety = setTimeout(() => {
      if (!_upPending || run !== _upRun) return;
      _fail('Upload confirmation timed out',
        'Check the file summary below before retrying to avoid uploading it twice.');
    }, 60000);
  }

  function _settleRequest(run, status){
    if (run !== _upRun || !_upPending) return;
    _upInflight = Math.max(0, _upInflight - 1);
    if (status === 0)
      return _fail('Connection interrupted',
        'The server did not confirm this upload. Check your connection and retry.');
    if (status >= 400)
      return _fail('Upload rejected (HTTP ' + status + ')',
        'Check the server or file, then choose the video again.');
    if (_upInflight === 0) _finalizing(run);
  }

  function _checkStats(){
    if (!_upPending) return;
    const n = document.querySelector('.aee-file-stats');
    if (n && (n !== _upBaseNode || (n.textContent || '') !== _upBaseText)){
      _complete();
      return;
    }
    const statusNode = document.getElementById('aee-analysis-progress');
    const statusText = statusNode ? statusNode.textContent : '';
    const statusMarker = statusNode
      ? statusNode.querySelector('[data-aee-upload-ready]') : null;
    const markerValue = statusMarker
      ? statusMarker.getAttribute('data-aee-upload-ready') : '';
    const statusChanged = markerValue
      ? markerValue !== _upBaseStatusMarker
      : statusText && statusText !== _upBaseStatus;
    if (statusChanged && /\b\d+\s+files?\s+ready\b/i.test(statusText)){
      _complete();
    }
  }
  setInterval(_checkStats, 250);

  function _onFileChange(e){
    const t = e.target;
    if (t && t.type === 'file' && t.files && t.files.length){
      LOG('change captured, files:', t.files.length);
      _begin(t.files, false);
    }
  }
  document.addEventListener('change', _onFileChange, true);
  document.addEventListener('change', _onFileChange, false);

  document.addEventListener('click', function(e){
    if (e.target.closest('.aee-upload-trigger')){
      LOG('upload trigger clicked');
      setTimeout(() => { if (!_upPending) _begin([], true); }, 40);
    }
  }, true);

  const _observedInputs = new WeakSet();
  function _attachToInputs(){
    document.querySelectorAll('input[type="file"]').forEach(inp => {
      if (_observedInputs.has(inp)) return;
      _observedInputs.add(inp);
      inp.addEventListener('change', _onFileChange, false);
    });
  }
  _attachToInputs();
  try {
    new MutationObserver(_attachToInputs).observe(document.documentElement,
      { childList: true, subtree: true });
  } catch(e){}

  (function(){
    const _open = XMLHttpRequest.prototype.open;
    const _send = XMLHttpRequest.prototype.send;
    XMLHttpRequest.prototype.open = function(m, url){
      try { this.__aee_url = String(url || ''); } catch(_){}
      return _open.apply(this, arguments);
    };
    XMLHttpRequest.prototype.send = function(){
      const u = this.__aee_url || '';
      if (/upload/i.test(u)){
        const xhr = this;
        if (!_upPending) _begin([], false);
        const run = _upRun;
        _upInflight++;
        let prev = 0;
        try {
          if (xhr.upload && xhr.upload.addEventListener){
            xhr.upload.addEventListener('progress', function(e){
              if (run !== _upRun || !_upPending || !e.lengthComputable) return;
              _upLoaded += Math.max(0, e.loaded - prev); prev = e.loaded;
              if (_upTotal <= 0) _upTotal = e.total;
              const denom = Math.max(_upTotal, e.total || 0, _upLoaded);
              _paint(Math.min(99, (_upLoaded / denom) * 100), _upLoaded, denom);
            });
          }
          xhr.addEventListener('loadend', function(){
            _settleRequest(run, xhr.status || 0);
          });
          xhr.addEventListener('error', function(){
            if (run === _upRun) _fail('Connection interrupted',
              'The upload stopped before the server confirmed it. Retry when connected.');
          });
          xhr.addEventListener('abort', function(){
            if (run === _upRun) _fail('Upload cancelled',
              'Choose the video again when you are ready to retry.');
          });
          xhr.addEventListener('timeout', function(){
            if (run === _upRun) _fail('Upload timed out',
              'Check your connection, then choose the video again.');
          });
        } catch(_){}
      }
      return _send.apply(this, arguments);
    };
  })();

  if (window.fetch){
    const _origFetch = window.fetch;
    window.fetch = function(input){
      const url = (typeof input === 'string') ? input : (input && input.url) || '';
      const isUp = /upload/i.test(url);
      if (isUp && !_upPending) _begin([], false);
      const run = _upRun;
      const p = _origFetch.apply(this, arguments);
      if (isUp){
        _upInflight++;
        p.then(response => {
          _settleRequest(run, response && response.status
            ? response.status : (response && response.ok ? 200 : 0));
        }, () => {
          if (run === _upRun)
            _fail('Connection interrupted',
              'The upload stopped before the server confirmed it. Check your connection and retry.');
        });
      }
      return p;
    };
  }

  /* ---------- Drag & drop ---------- */
  let dragDepth = 0;
  function findShell(){ return document.querySelector('.aee-upload-shell'); }
  ['dragenter','dragover'].forEach(ev => {
    document.addEventListener(ev, e => {
      const s = findShell();
      if (!s) return;
      if (!e.dataTransfer) return;
      const types = Array.from(e.dataTransfer.types || []);
      if (types.indexOf('Files') < 0) return;
      e.preventDefault();
      dragDepth++;
      s.classList.add('dragover');
    });
  });
  document.addEventListener('dragleave', e => {
    const s = findShell();
    if (!s) return;
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) s.classList.remove('dragover');
  });
  document.addEventListener('drop', e => {
    const s = findShell();
    if (!s) return;
    dragDepth = 0;
    s.classList.remove('dragover');
  });

  /* ---------- Paste ---------- */
  document.addEventListener('paste', e => {
    const items = (e.clipboardData && e.clipboardData.items) || [];
    const files = [];
    for (let i = 0; i < items.length; i++){
      const it = items[i];
      if (it.kind !== 'file') continue;
      const f = it.getAsFile();
      if (f && f.type && f.type.indexOf('video/') === 0) files.push(f);
    }
    if (!files.length) return;
    const inp = document.querySelector('input[type="file"]');
    if (!inp) return;
    e.preventDefault();
    try {
      const dt = new DataTransfer();
      files.forEach(f => dt.items.add(f));
      inp.files = dt.files;
      _begin(files, false);
      inp.dispatchEvent(new Event('change', { bubbles: true }));
    } catch(err) {
      toast('⚠️ Paste not supported', 'warn', 2400);
    }
  });

  /* ---------- Timeline seek ---------- */
  function _findMasterVideo(){
    const wrap = document.getElementById('aee-master-wrap')
              || document.querySelector('.aee-master-wrap');
    if (wrap){
      const v = wrap.querySelector('video');
      if (v) return v;
    }
    const testid = document.querySelector('[data-testid="video"]');
    if (testid){
      const v = testid.querySelector('video');
      if (v) return v;
    }
    const all = Array.from(document.querySelectorAll('video'));
    for (let i = 0; i < all.length; i++){
      const v = all[i];
      const r = v.getBoundingClientRect();
      if (v.offsetParent !== null && r.height > 60 && r.width > 60) return v;
    }
    return all[0] || null;
  }

  function _parseSeekFromBar(bar){
    if (!bar) return null;
    const id = bar.getAttribute('id') || '';
    let m = id.match(/^aee-seek-\d+-(\d+)$/);
    if (m) return parseInt(m[1], 10) / 1000;
    const title = bar.getAttribute('title') || '';
    m = title.match(/seek\s+([\d.]+)s/i);
    if (m) return parseFloat(m[1]);
    return null;
  }

  function seekMasterVideo(t){
    const video = _findMasterVideo();
    if (!video){ toast('⚠️ Master video not found', 'warn', 2000); return false; }
    try { video.scrollIntoView({behavior:'smooth', block:'center'}); } catch(_){}
    const doSeek = () => {
      try {
        video.currentTime = t;
        const p = video.play();
        if (p && p.catch) p.catch(() => {});
      } catch(err){}
    };
    if (video.readyState === 0){ try { video.load(); } catch(_){} }
    if (video.readyState >= 1 && isFinite(video.duration)){ doSeek(); return true; }
    let tries = 0;
    const tick = () => {
      tries++;
      if (video.readyState >= 1 && isFinite(video.duration)){ doSeek(); return; }
      if (tries < 60) setTimeout(tick, 100);
      else doSeek();
    };
    setTimeout(tick, 100);
    return true;
  }
  window.aeeSeek = seekMasterVideo;

  document.addEventListener('click', e => {
    if (e.target.closest('#theme-toggle')) { toggleTheme(); return; }
    if (e.target.closest('#cmd-open')) { openCmd(); return; }
    if (e.target.closest('.aee-cmd-backdrop')) { closeCmd(); return; }
    const bar = e.target.closest('.aee-tl-bar');
    if (bar){
      const t = _parseSeekFromBar(bar);
      if (t !== null && isFinite(t)){
        e.preventDefault(); e.stopPropagation();
        const ok = seekMasterVideo(t);
        if (ok){
          bar.style.outline = '2px solid #22d3ee';
          setTimeout(() => { bar.style.outline = ''; }, 500);
          toast('⏱ Seek → ' + t.toFixed(2) + 's', 'info', 1200);
        }
      }
    }
  }, true);

  document.addEventListener('keydown', e => {
    const t = e.target.tagName;
    const isTyping = (t === 'INPUT' || t === 'TEXTAREA');
    if ((e.metaKey || e.ctrlKey) && (e.key === 'k' || e.key === 'K')) {
      e.preventDefault();
      const el = document.getElementById('aee-cmd');
      if (el && el.classList.contains('open')) closeCmd(); else openCmd();
      return;
    }
    if (e.key === 'Escape') { closeCmd(); return; }
    const cmdOpen = document.getElementById('aee-cmd');
    if (cmdOpen && cmdOpen.classList.contains('open')) {
      if (e.key === 'ArrowDown') {
        cmdIdx = Math.min(cmdFiltered.length-1, cmdIdx+1);
        renderCmds(); e.preventDefault();
      } else if (e.key === 'ArrowUp') {
        cmdIdx = Math.max(0, cmdIdx-1); renderCmds(); e.preventDefault();
      } else if (e.key === 'Enter') {
        if (cmdFiltered[cmdIdx]) runCmd(cmdFiltered[cmdIdx]);
        e.preventDefault();
      }
      return;
    }
    if (isTyping) return;
    if (e.key === 'a' || e.key === 'A') {
      const b = document.querySelector('button.primary');
      if (b) { b.click(); e.preventDefault(); }
    }
    if (['1','2','3','4','5','6','7','8'].indexOf(e.key) >= 0) clickTab(parseInt(e.key)-1);
    if (e.key === 't' || e.key === 'T') toggleTheme();
  });

  const cmds = [
    {label:'📊  Go to Overview',   keys:['1'], run:()=>clickTab(0)},
    {label:'📸  Go to Captures',   keys:['2'], run:()=>clickTab(1)},
    {label:'🎞  Go to Summaries',  keys:['3'], run:()=>clickTab(2)},
    {label:'🧭  Go to Object Tracking',  keys:['4'], run:()=>clickTab(3)},
    {label:'🔍  Go to EigenCAM',   keys:['5'], run:()=>clickTab(4)},
    {label:'🔬  Go to Evidence',   keys:['6'], run:()=>clickTab(5)},
    {label:'📄  Go to Report',     keys:['7'], run:()=>clickTab(6)},
    {label:'History',               keys:['8'], run:()=>clickTab(7)},
    {label:'⚡  Run analysis',      keys:['A'], run:()=>{
       const b = document.querySelector('button.primary'); if (b) b.click(); }},
    {label:'🌓  Toggle theme',     keys:['T'], run: toggleTheme},
  ];
  let cmdIdx = 0, cmdFiltered = cmds.slice();

  function renderCmds(){
    const list = document.getElementById('aee-cmd-list');
    if (!list) return;
    list.innerHTML = '';
    cmdFiltered.forEach((c, i) => {
      const el = document.createElement('div');
      el.className = 'aee-cmd-item' + (i === cmdIdx ? ' active' : '');
      el.innerHTML = '<span>' + c.label + '</span>' +
        (c.keys ? '<span class="kbd">' + c.keys[0] + '</span>' : '');
      el.addEventListener('click', () => { runCmd(c); });
      list.appendChild(el);
    });
  }
  function openCmd(){
    const el = document.getElementById('aee-cmd');
    if (!el) return;
    el.classList.add('open');
    const inp = document.getElementById('aee-cmd-input');
    if (inp) { inp.value=''; inp.focus(); }
    cmdIdx = 0; cmdFiltered = cmds.slice(); renderCmds();
  }
  function closeCmd(){
    const el = document.getElementById('aee-cmd');
    if (el) el.classList.remove('open');
  }
  function runCmd(c){
    closeCmd();
    try { c.run(); } catch(e){}
  }

  document.addEventListener('input', e => {
    if (e.target.id === 'aee-cmd-input') {
      const q = e.target.value.toLowerCase();
      cmdFiltered = cmds.filter(c => c.label.toLowerCase().indexOf(q) >= 0);
      cmdIdx = 0; renderCmds();
    }
  });

  /* ============ MODEL HEATMAP ============ */
  const _PALETTE = [
    '#ef4444','#f59e0b','#8b5cf6','#22d3ee','#10b981',
    '#eab308','#ec4899','#06b6d4','#a3e635','#f97316',
    '#3b82f6','#d946ef',
  ];

  function _findVideoEl(){
    const wrap = document.getElementById('aee-master-wrap')
              || document.querySelector('.aee-master-wrap');
    if (wrap){ const v = wrap.querySelector('video'); if (v) return v; }
    const all = document.querySelectorAll('video');
    for (let i = 0; i < all.length; i++){
      const v = all[i];
      const r = v.getBoundingClientRect();
      if (v.offsetParent !== null && r.height > 80 && r.width > 120) return v;
    }
    return null;
  }

  function _rgba(hex, a){
    const h = hex.replace('#','');
    const r = parseInt(h.substring(0,2), 16);
    const g = parseInt(h.substring(2,4), 16);
    const b = parseInt(h.substring(4,6), 16);
    return 'rgba(' + r + ',' + g + ',' + b + ',' + a + ')';
  }

  function _smooth(arr, sigma){
    const r = Math.ceil(sigma * 3), k = [];
    let ks = 0;
    for (let i = -r; i <= r; i++){
      const w = Math.exp(-(i*i)/(2*sigma*sigma));
      k.push(w); ks += w;
    }
    const out = new Array(arr.length).fill(0);
    for (let i = 0; i < arr.length; i++){
      if (!arr[i]) continue;
      for (let j = -r; j <= r; j++){
        const idx = i + j;
        if (idx >= 0 && idx < arr.length) out[idx] += arr[i] * k[j + r] / ks;
      }
    }
    return out;
  }

  function _drawHeatmap(canvas, data, hiddenModels){
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const dpr = W / Math.max(1, canvas.clientWidth || W);
    ctx.clearRect(0, 0, W, H);
    const nb = data.nbins;
    const models = Object.keys(data.models).sort();
    const visible = models.filter(m => !hiddenModels || !hiddenModels.has(m));
    const sigma = Math.max(0.55, nb / 900);
    const smoothed = {};
    models.forEach(m => {
      smoothed[m] = _smooth(data.models[m].heat || [], sigma);
    });
    const padTop = 5 * dpr;
    const padBottom = 5 * dpr;
    const usableH = Math.max(1, H - padTop - padBottom);
    const colW = W / nb;
    const showConsensus = models.length > 1 && visible.length === models.length;
    const consensusH = showConsensus ? Math.min(7 * dpr, usableH * .18) : 0;
    const laneGap = Math.max(1, 1.25 * dpr);
    const rowH = visible.length
      ? Math.max(1, (usableH - consensusH - laneGap * Math.max(0, visible.length - 1)) / visible.length)
      : 0;

    if (data.peak_window && data.peak_window.length === 2){
      const a = data.peak_window[0], b = data.peak_window[1];
      const x0 = (a / nb) * W, x1 = (b / nb) * W;
      const band = ctx.createLinearGradient(x0, 0, x1, 0);
      band.addColorStop(0, 'rgba(96,165,250,0)');
      band.addColorStop(.5, 'rgba(96,165,250,.12)');
      band.addColorStop(1, 'rgba(96,165,250,0)');
      ctx.fillStyle = band;
      ctx.fillRect(x0, padTop, x1 - x0, usableH);
      ctx.save();
      ctx.strokeStyle = 'rgba(147,197,253,.62)';
      ctx.lineWidth = 1 * dpr;
      ctx.setLineDash([2 * dpr, 3 * dpr]);
      ctx.beginPath();
      ctx.moveTo(x0, padTop); ctx.lineTo(x0, H - padBottom);
      ctx.moveTo(x1, padTop); ctx.lineTo(x1, H - padBottom);
      ctx.stroke();
      ctx.restore();
    }

    for (let mark = 0; mark <= 4; mark++){
      const x = W * mark / 4;
      ctx.beginPath();
      ctx.moveTo(x, padTop); ctx.lineTo(x, H - padBottom);
      ctx.strokeStyle = 'rgba(226,232,240,.10)';
      ctx.lineWidth = 1 * dpr;
      ctx.stroke();
    }

    let laneY = padTop;
    if (showConsensus && data.consensus){
      ctx.fillStyle = 'rgba(34,211,238,.07)';
      ctx.fillRect(0, laneY, W, consensusH);
      for (let i = 0; i < nb; i++){
        const agreeing = data.consensus[i] || 0;
        if (agreeing < 2) continue;
        const intensity = Math.min(1, (agreeing - 1) / Math.max(1, models.length - 1));
        ctx.fillStyle = _rgba('#22d3ee', .3 + .68 * intensity);
        ctx.fillRect(i * colW, laneY, Math.max(1, colW), consensusH);
      }
      laneY += consensusH + laneGap;
    }

    visible.forEach(model => {
      const index = models.indexOf(model);
      const color = _PALETTE[index % _PALETTE.length];
      const values = smoothed[model];
      const peak = Math.max(1e-6, Math.max.apply(null, values));
      const laneBottom = laneY + rowH;
      ctx.fillStyle = 'rgba(15,23,42,.58)';
      ctx.fillRect(0, laneY, W, rowH);
      for (let i = 0; i < nb; i++){
        const value = values[i];
        if (value <= 0) continue;
        const strength = Math.min(1, value / peak);
        const alpha = .12 + .86 * Math.pow(strength, .72);
        ctx.fillStyle = _rgba(color, alpha);
        ctx.fillRect(i * colW, laneY, Math.max(1, colW), rowH);
        const glowH = Math.max(1, rowH * (.12 + .2 * strength));
        ctx.fillStyle = _rgba(color, Math.min(1, alpha + .1));
        ctx.fillRect(i * colW, laneBottom - glowH, Math.max(1, colW), glowH);
      }
      ctx.beginPath();
      ctx.moveTo(0, laneBottom);
      ctx.lineTo(W, laneBottom);
      ctx.strokeStyle = 'rgba(226,232,240,.17)';
      ctx.lineWidth = 1 * dpr;
      ctx.stroke();
      laneY = laneBottom + laneGap;
    });

    if (!visible.length){
      ctx.fillStyle = 'rgba(226,232,240,.72)';
      ctx.font = `${Math.max(9, 10 * dpr)}px system-ui`;
      ctx.textAlign = 'center';
      ctx.fillText('Choose a model to show its activity', W / 2, H / 2);
    }
    ctx.beginPath();
    ctx.moveTo(0, H - padBottom);
    ctx.lineTo(W, H - padBottom);
    ctx.strokeStyle = 'rgba(255,255,255,.18)';
    ctx.lineWidth = 1 * dpr;
    ctx.stroke();
  }

  function _sizeAndDraw(overlay){
    const canvas = overlay.querySelector('canvas');
    if (!canvas || !overlay.__aeeData) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(50, Math.round(overlay.clientWidth * dpr));
    const h = Math.max(30, Math.round(overlay.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h){ canvas.width = w; canvas.height = h; }
    _drawHeatmap(canvas, overlay.__aeeData, overlay.__aeeHiddenModels);
  }

  function _buildHeatOverlay(video, data, key){
    const parent = video.parentElement;
    if (!parent) return null;
    if (getComputedStyle(parent).position === 'static') parent.style.position = 'relative';
    parent.querySelectorAll(':scope > .aee-heat-overlay').forEach(el => el.remove());
    const overlay = document.createElement('div');
    overlay.className = 'aee-heat-overlay';
    overlay.__aeeData = data; overlay.__aeeKey = key; overlay.__aeeVideo = video;
    overlay.__aeeHiddenModels = new Set();
    const canvas = document.createElement('canvas');
    overlay.appendChild(canvas);
    const playhead = document.createElement('div');
    playhead.className = 'aee-heat-playhead';
    overlay.appendChild(playhead);
    const legend = document.createElement('div');
    legend.className = 'aee-heat-legend';
    const title = document.createElement('span');
    title.className = 'aee-heat-title';
    title.textContent = 'Model activity';
    legend.appendChild(title);
    const agree = document.createElement('span');
    agree.className = 'aee-heat-agreement';
    const dotA = document.createElement('i'); dotA.style.background = '#22d3ee';
    agree.appendChild(dotA);
    agree.appendChild(document.createTextNode('same-class co-detection'));
    legend.appendChild(agree);
    Object.keys(data.models).sort().forEach((m, i) => {
      const color = _PALETTE[i % _PALETTE.length];
      const stats = data.models[m];
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'aee-heat-model-chip';
      chip.setAttribute('aria-pressed', 'true');
      chip.title = `${stats.hits || 0} detections · mean confidence ${Number(stats.mean_confidence || 0).toFixed(2)} · click to toggle`;
      const dot = document.createElement('i');
      dot.style.background = color;
      chip.appendChild(dot);
      chip.appendChild(document.createTextNode(m));
      const count = document.createElement('small');
      count.textContent = String(stats.hits || 0);
      chip.appendChild(count);
      chip.addEventListener('click', e => {
        e.stopPropagation();
        if (overlay.__aeeHiddenModels.has(m)){
          overlay.__aeeHiddenModels.delete(m);
          chip.setAttribute('aria-pressed', 'true');
        } else {
          overlay.__aeeHiddenModels.add(m);
          chip.setAttribute('aria-pressed', 'false');
        }
        _sizeAndDraw(overlay);
      });
      legend.appendChild(chip);
    });
    if (data.peak_window && data.duration > 0 && data.nbins > 0){
      const a = data.peak_window[0], b = data.peak_window[1];
      const mid = ((a + b) / 2) / data.nbins * data.duration;
      const jump = document.createElement('button');
      jump.type = 'button';
      jump.className = 'aee-heat-jump';
      jump.textContent = 'Peak activity';
      jump.title = 'Seek to the peak window · ' + mid.toFixed(2) + 's';
      jump.addEventListener('click', e => {
        e.stopPropagation();
        if (window.aeeSeek) window.aeeSeek(mid);
      });
      legend.appendChild(jump);
    }
    overlay.appendChild(legend);
    parent.appendChild(overlay);
    _sizeAndDraw(overlay);
    requestAnimationFrame(() => overlay.classList.add('active'));
    if (window.ResizeObserver){
      const ro = new ResizeObserver(() => _sizeAndDraw(overlay));
      ro.observe(overlay);
      overlay.__aeeRO = ro;
    }
    const dur = data.duration || 0;
    const tick = () => {
      if (!overlay.isConnected || !video.isConnected){
        if (overlay.__aeeRO) overlay.__aeeRO.disconnect();
        return;
      }
      if (dur > 0){
        const pct = Math.max(0, Math.min(100, ((video.currentTime || 0) / dur) * 100));
        playhead.style.left = pct + '%';
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
    return overlay;
  }

  function _tryInitHeatmap(){
    const dataEl = document.querySelector('.aee-heat-data');
    const text = dataEl ? (dataEl.textContent || '').trim() : '';
    const video = _findVideoEl();
    if (!text){
      document.querySelectorAll('.aee-heat-overlay').forEach(el => el.remove());
      return false;
    }
    if (!video) return false;
    const existing = video.parentElement &&
      video.parentElement.querySelector(':scope > .aee-heat-overlay');
    if (existing && existing.__aeeKey === text && existing.__aeeVideo === video){
      existing.classList.add('active');
      return true;
    }
    let data;
    try { data = JSON.parse(text); } catch(e){ return false; }
    if (!data || !data.nbins || !data.models) return false;
    const r = video.getBoundingClientRect();
    if (r.width < 80 || r.height < 40) return false;
    return !!_buildHeatOverlay(video, data, text);
  }

  function _observeHeatmap(){
    let pending = false;
    const run = () => {
      if (pending) return;
      pending = true;
      setTimeout(() => { pending = false; _tryInitHeatmap(); }, 150);
    };
    new MutationObserver(run).observe(document.documentElement,
      { childList: true, subtree: true });
    setInterval(_tryInitHeatmap, 1000);
    run();
  }

  if (document.readyState === 'loading')
    document.addEventListener('DOMContentLoaded', _observeHeatmap);
  else _observeHeatmap();

  LOG('bootstrap complete');
  } catch(err){
    try { console.error('[AEE] FATAL', err); } catch(_){}
  }
})();
</script>
""".replace(
    "__AEE_GRADIO_VERSION__",
    json.dumps(str(getattr(gr, "__version__", ""))),
)


def _install_gradio_share_css_path_fix():
    """Run the Gradio CDN CSS URL rewrite before its remote bundle loads."""
    try:
        import gradio.routes as gradio_routes

        templates = gradio_routes.templates
        if getattr(templates, "_aee_share_css_path_fix", False):
            return

        original_template_response = templates.TemplateResponse
        early_script = _JS_HEAD.split('<link rel="preconnect"', 1)[0].strip()
        marker = b"<!-- AEE_GRADIO_SHARE_CSS_PATH_FIX -->"

        def patched_template_response(*args, **kwargs):
            response = original_template_response(*args, **kwargs)
            template_name = kwargs.get("name")
            if template_name is None and len(args) > 1:
                template_name = args[1]
            if template_name != "frontend/share.html":
                return response

            body = getattr(response, "body", None)
            if not isinstance(body, bytes) or marker in body or b"<head>" not in body:
                return response

            insertion = marker + early_script.encode("utf-8") + b"\n"
            response.body = body.replace(b"<head>", b"<head>" + insertion, 1)
            response.headers["content-length"] = str(len(response.body))
            if "etag" in response.headers:
                del response.headers["etag"]
            return response

        templates.TemplateResponse = patched_template_response
        templates._aee_share_css_path_fix = True
        log.info("Installed early Gradio share-page CSS path fix")
    except Exception as exc:
        log.warning("Could not install early Gradio share-page CSS path fix: %s", exc)


def _det_df(rows):
    if not rows:
        return pd.DataFrame([{"Video":"","Timestamp":"","Type":"info",
            "Object":"no detections yet","Confidence":"","Fusion":"",
            "Source":"","Track":"","Threat":"","BBox":""}])
    return pd.DataFrame(rows)


def _build_rows(results, tracker=None):
    rows = []
    for r in results:
        for d in r.detections:
            lvl, sc, _ = threat_score(d, tracker, r.video_name)
            threat = f"{lvl.title()} ({sc})" if sc else "—"
            rows.append({"Video":r.video_name,"Timestamp":ts(d["timestamp"]),
                "Type":d.get("kind","object"),"Object":f"{icon_for(d['label'])} {d['label']}",
                "Confidence":round(float(d["confidence"]),4),
                "Fusion":round(float(d.get("fusion_prob",1.0)),4),
                "Source":d.get("source",""),"Track":d.get("track_id",-1),
                "Threat":threat,"BBox":str(d.get("bbox",""))})
    return rows


def _build_master(results, out, work):
    global _MASTER_MAP
    _MASTER_MAP = []
    work.mkdir(parents=True, exist_ok=True)
    clips, offset = [], 0.0
    for r in results:
        dur = float(r.metadata.duration or 0)
        if dur <= 0 or not Path(r.source_path).exists():
            continue
        clips.append(str(r.source_path))
        _MASTER_MAP.append({
            "video": r.video_name,
            "src_start": 0.0,
            "src_end": dur,
            "master_start": offset,
            "master_end": offset + dur,
        })
        offset += dur
    if not clips:
        return None
    return merge_clips(clips, out, work / "_mg_full")


def process_batch(videos, case_number, evidence_id, examiner, agency, case_notes,
                  high_accuracy=False, progress=gr.Progress()):
    EMPTY_DF = _det_df([])
    EMPTY_TRACK_HTML = Tracker().panel_html()
    EMPTY_CAM_HTML = CAMCache().panel_html()

    EMPTY_OUT = (None, None, [], [], None,
                 gr.update(choices=["All"], value="All"),
                 gr.update(choices=["All"], value="All"),
                 EMPTY_DF, _empty_html(), "",
                 [], [], [], None,
                 [], [], [], [],
                 EMPTY_TRACK_HTML, EMPTY_CAM_HTML, EMPTY_DF)

    if not _BATCH_LOCK.acquire(blocking=False):
        busy = (f'<div class="aee-empty"><div class="aee-empty-icon">⏳</div>'
                f'<div class="aee-empty-title">Another analysis is already running</div>'
                f'<div class="aee-empty-sub">Wait for the current full-video analysis to finish.</div></div>')
        busy_out = list(EMPTY_OUT)
        busy_out[8] = busy
        busy_out[9] = "⏳ Another analysis is still running."
        yield tuple(busy_out)
        return

    with _ACTIVE_LOCK: _ACTIVE_CANCEL.clear()
    cancel_event = _ACTIVE_CANCEL

    case_meta = {"case_number":(case_number or "").strip(),
        "evidence_id":(evidence_id or "").strip(),
        "examiner":(examiner or "").strip(),"agency":(agency or "").strip()}
    case_notes = (case_notes or "").strip()

    try:
        if not videos:
            out = list(EMPTY_OUT); out[9] = "No video is selected."
            yield tuple(out); return
        if not isinstance(videos, list): videos = [videos]
        valid = []
        for v in videos:
            p = v if isinstance(v, str) else getattr(v, "name", None)
            if not p: continue
            ext = Path(p).suffix.lower()
            if ext in CFG.SUPPORTED or ext == "":
                valid.append(p)
            else:
                info = probe_video(p)
                if info["valid"]: valid.append(p)
        if not valid:
            out = list(EMPTY_OUT); out[9] = "No usable video was found."
            yield tuple(out); return
        if not Ensemble.get().info["loaded"]:
            out = list(EMPTY_OUT); out[9] = "Detection models are not ready."
            yield tuple(out); return

        _init_html = ('<div class="aee-load" style="padding:32px;text-align:center;">'
                      '<div style="font-size:1rem;color:#e6edf7;font-weight:700;'
                      'margin-bottom:8px;">Preparing evidence…</div>'
                      '<div style="font-size:.85rem;color:#94a3b8;">'
                      'Fingerprinting and staging uploaded files</div></div>')
        prep_out = list(EMPTY_OUT); prep_out[9] = _init_html
        yield tuple(prep_out)

        stable = [_stable_copy(p) for p in valid]
        stable = [p for p in stable if p]
        if not stable:
            nofile_out = list(EMPTY_OUT)
            nofile_out[8] = _empty_html()
            nofile_out[9] = "⚠️ Files no longer accessible."
            yield tuple(nofile_out)
            return
        valid = stable

        total = len(valid)
        root = Path(tempfile.mkdtemp(prefix="aee_")); t0 = time.time()
        results, errs = [], []
        q = queue.Queue(); stop = threading.Event()
        proc = Processor()

        def worker():
            try:
                for i, vp in enumerate(valid, 1):
                    q.put(("video_start", i, Path(vp).name))
                    try:
                        for ev in proc.process_stream(
                                vp, root, cancel_event=cancel_event,
                                high_accuracy=high_accuracy):
                            if stop.is_set(): return
                            if ev[0] == "stage":
                                q.put(("stage", ev[1], ev[2], ev[3], i, Path(vp).name))
                            elif ev[0] == "substep": q.put(("substep", ev[1]))
                            elif ev[0] == "detection": q.put(("detection", ev[1]))
                            elif ev[0] == "cam": q.put(("cam", ev[1]))
                            elif ev[0] == "result": q.put(("result", ev[1]))
                            elif ev[0] == "error":
                                q.put(("error", ev[1], Path(vp).name))
                            elif ev[0] == "cancelled":
                                q.put(("cancelled", None)); return
                    except Exception as e:
                        log.exception("worker fail on %s", vp)
                        q.put(("error", str(e)[:200], Path(vp).name))
            finally:
                q.put(("done", None))

        t = threading.Thread(target=worker, daemon=True); t.start()

        ticker_dets = []
        was_cancelled = False
        current_stage = "meta"; stage_frac = 0.; stage_msg = ""
        cur_vid = valid[0]; cur_vi = 1
        last_yield = 0.; master = pdf = None; substep_data = None

        STAGE_WEIGHTS = {"meta":0.02,"audio":0.03,"detect":0.75,"highlight":0.10,"summary":0.10}
        ORDER = ["meta","audio","detect","highlight","summary"]

        def overall_frac():
            idx = ORDER.index(current_stage) if current_stage in ORDER else 0
            before = sum(STAGE_WEIGHTS[s] for s in ORDER[:idx])
            within = STAGE_WEIGHTS[current_stage] * min(1., max(0., stage_frac))
            return ((cur_vi - 1) + before + within) / max(1, total)

        def model_times():
            return {m["name"]: m["infer_ms"] for m in Ensemble.get().models
                    if m.get("infer_ms", 0) > 0}

        def snapshot():
            rows = _build_rows(results, proc.tracker)
            prog = _progress_html(current_stage, stage_frac, stage_msg,
                                  time.time()-t0, cur_vid, cur_vi, total,
                                  overall_frac(), ticker_dets, substep_data,
                                  model_times())
            track_html_s = proc.tracker.panel_html()
            cam_html_s = proc.camcache.panel_html()
            track_rows = proc.tracker.summaries()
            cam_heat = [c["heat"] for c in proc.camcache.items]
            cam_blend = [c["blend"] for c in proc.camcache.items]
            cam_display = []
            for c in proc.camcache.items:
                cap = f"{c['label']} · {c['source']} @ {ts(c['t'])}"
                cam_display.append((c["heat"], "🔥 heat · " + cap))
                cam_display.append((c["blend"], "🎯 blend · " + cap))
            return (None, None,
                    [c for r in results for c in r.captures],
                    [s for r in results for s in r.object_summaries.values()],
                    None,
                    gr.update(choices=["All"], value="All"),
                    gr.update(choices=["All"], value="All"),
                    _det_df(rows), _empty_html(), prog,
                    rows, [], [], None,
                    cam_heat, cam_blend, cam_display, track_rows,
                    track_html_s, cam_html_s, _det_df(track_rows))

        yield snapshot()

        while True:
            try: ev = q.get(timeout=0.35)
            except queue.Empty:
                if cancel_event.is_set(): stop.set()
                elif time.time() - last_yield > 1.2:
                    yield snapshot(); last_yield = time.time()
                continue

            kind = ev[0]
            if kind == "done": break
            if kind == "cancelled":
                was_cancelled = True; stop.set(); break
            if kind == "video_start":
                cur_vi, cur_vid = ev[1], ev[2]; substep_data = None
            elif kind == "stage":
                _, name, frac, msg, vi, vid = ev
                current_stage, stage_frac, stage_msg = name, frac, msg
                cur_vi, cur_vid = vi, vid
                progress(overall_frac(), desc=f"[{vi}/{total}] {vid}: {name}")
            elif kind == "substep": substep_data = ev[1]
            elif kind == "detection":
                ticker_dets.append(ev[1])
                if len(ticker_dets) > 60: ticker_dets = ticker_dets[-60:]
            elif kind == "result": results.append(ev[1])
            elif kind == "error":
                errs.append({"video": ev[2], "error": ev[1]})

            if time.time() - last_yield > 0.35:
                yield snapshot(); last_yield = time.time()

        if was_cancelled or cancel_event.is_set():
            stop.set(); t.join(timeout=5)
            cancelled_out = list(EMPTY_OUT)
            cancelled_out[9] = "Analysis stopped."
            yield tuple(cancelled_out); return

        _stamp = safe_slug(case_meta.get("case_number") or "evidence")[:60] or "evidence"

        if results:
            try:
                master = _build_master(results, root/f"{_stamp}_evidence_summary.mp4",
                                        root/"_m")
            except Exception as e: log.warning("master build: %s", e)
            try:
                pdf = build_pdf(results, Ensemble.get().info,
                                root/f"{_stamp}_evidence_report.pdf",
                                tracker=proc.tracker,
                                case_meta=case_meta, case_notes=case_notes)
            except Exception as e: log.warning("pdf build: %s", e)

        master_s = _persist_download(
            master, f"{_stamp}_evidence_summary.mp4")
        pdf_s = _persist_download(
            pdf, f"{_stamp}_evidence_report.pdf")
        if results and master_s is None:
            log.warning("Master Summary download unavailable for this analysis")
        if results and pdf_s is None:
            log.warning("Forensic PDF download unavailable for this analysis")

        all_rows = _build_rows(results, proc.tracker)
        caps = [c for r in results for c in r.captures if Path(c).exists()]
        sums = [s for r in results for s in r.object_summaries.values()
                if Path(s).exists()]
        classes = ["All"] + sorted({r["Object"] for r in all_rows}) if all_rows else ["All"]
        track_rows = proc.tracker.summaries()
        cam_heat = [c["heat"] for c in proc.camcache.items]
        cam_blend = [c["blend"] for c in proc.camcache.items]
        cam_display = []
        for c in proc.camcache.items:
            cap = f"{c['label']} · {c['source']} @ {ts(c['t'])}"
            cam_display.append((c["heat"], "🔥 heat · " + cap))
            cam_display.append((c["blend"], "🎯 blend · " + cap))

        history_warnings = _record_analysis_history(
            results, master_s, pdf_s, case_number,
            high_accuracy=high_accuracy, case_meta=case_meta,
            case_notes=case_notes, tracker_rows=track_rows,
            cam_items=proc.camcache.items, work_root=root)
        history_status = ""
        if history_warnings:
            history_status = (
                "Analysis completed, but some History files could not be saved. "
                "Open History to review the storage warnings. "
                + history_warnings[0][:300])
        yield (master_s, pdf_s, caps, sums, caps[0] if caps else None,
               gr.update(choices=classes, value="All"),
               gr.update(choices=classes, value="All"),
               _det_df(all_rows),
               _overview_html(results, errs, sums, master_s, pdf_s,
                              time.time()-t0, proc.tracker),
               history_status,
               all_rows, caps, sums, master_s,
               cam_heat, cam_blend, cam_display, track_rows,
               proc.tracker.panel_html(), proc.camcache.panel_html(),
               _det_df(track_rows))
    except Exception:
        log.exception("Gradio analysis callback failed")
        raise
    finally:
        _BATCH_LOCK.release()


def filter_gallery(class_filter, caps_state, rows_state):
    if not caps_state: return []
    if not rows_state or class_filter == "All": return caps_state
    df = pd.DataFrame(rows_state); df = df[df["Type"] == "object"]
    df = df[df["Object"].astype(str).str.contains(class_filter.split()[-1], na=False)]
    keep = set(df["Timestamp"].astype(str))
    out = []
    for c in caps_state:
        m = re.search(r"_(\d+)ms_", Path(c).stem)
        if not m: out.append(c); continue
        t = int(m.group(1)) / 1000.0
        for ts_str in keep:
            try:
                h, mi, s = ts_str.split(":")
                sec = int(h)*3600 + int(mi)*60 + float(s)
                if abs(sec - t) < 0.6: out.append(c); break
            except Exception: pass
    return out


def filter_evidence(search, class_filter, rows_state):
    if not rows_state: return pd.DataFrame()
    df = pd.DataFrame(rows_state)
    if class_filter and class_filter != "All":
        key = class_filter.split()[-1] if " " in class_filter else class_filter
        df = df[df["Object"].astype(str).str.contains(key, regex=False, na=False)]
    if search:
        q = search.lower()
        df = df[df.astype(str).apply(
            lambda r: r.str.lower().str.contains(q).any(), axis=1)]
    return df


def export_csv(rows_state, case_number):
    if not rows_state: return None
    df = pd.DataFrame(rows_state)
    stamp = safe_slug(case_number or "evidence")[:60] or "evidence"
    p = Path(tempfile.gettempdir()) / f"{stamp}_detections_{int(time.time())}.csv"
    df.to_csv(p, index=False); return str(p)


def export_json(rows_state, case_number, evidence_id, examiner, agency, case_notes):
    if not rows_state: return None
    payload = {"schema":"aee.evidence.v1",
        "generated_utc":datetime.utcnow().isoformat() + "Z",
        "case":{"case_number":case_number or "","evidence_id":evidence_id or "",
                "examiner":examiner or "","agency":agency or ""},
        "case_notes":case_notes or "","detections":rows_state}
    stamp = safe_slug(case_number or "evidence")[:60] or "evidence"
    p = Path(tempfile.gettempdir()) / f"{stamp}_detections_{int(time.time())}.json"
    p.write_text(json.dumps(payload, indent=2, default=str)); return str(p)


def export_tracks_csv(track_rows, case_number):
    if not track_rows: return None
    df = pd.DataFrame(track_rows)
    stamp = safe_slug(case_number or "evidence")[:60] or "evidence"
    p = Path(tempfile.gettempdir()) / f"{stamp}_tracks_{int(time.time())}.csv"
    df.to_csv(p, index=False); return str(p)


def pick_summary(name, sums):
    if not name or not sums: return None
    for s in sums:
        if Path(s).stem == name: return s
    return None


def cancel_active_analysis():
    _ACTIVE_CANCEL.set()
    return "Analysis stopped."


def _update_progress_html(fraction, description, indeterminate=False):
    percent = max(0, min(100, round(float(fraction) * 100)))
    label = _html.escape(str(description))
    progress_state = " aee-update-progress-indeterminate" if indeterminate else ""
    return (
        f'<div class="aee-update-progress{progress_state}" role="status" aria-live="polite">'
        f'<div class="aee-update-progress-label"><span>{label}</span><span>{percent}%</span></div>'
        '<div class="aee-update-progress-track" role="progressbar" '
        f'aria-valuemin="0" aria-valuemax="100" aria-valuenow="{percent}">'
        f'<div class="aee-update-progress-fill" style="width:{percent}%"></div></div></div>'
    )


def _run_update_task(operation, initial_description):
    events = queue.Queue()
    finished = object()

    def report(fraction, desc=None):
        events.put(("progress", fraction, desc or initial_description))

    def run():
        try:
            result = operation(report)
            events.put(("result", result, None))
        except Exception as exc:
            events.put(("result", None, exc))
        finally:
            events.put((finished, None, None))

    worker = threading.Thread(target=run, name="aee-update-task", daemon=True)
    worker.start()
    yield 0, initial_description, None, None, False
    while True:
        kind, value, detail = events.get()
        if kind is finished:
            break
        if kind == "progress":
            yield value, detail, None, None, False
        else:
            yield 1, initial_description, value, detail, True


def _update_status_html(message, tone="neutral", update=None, update_available=None,
                        progress=None):
    safe_message = _html.escape(str(message))
    available = (bool(update and update.get("available"))
                 if update_available is None else bool(update_available))
    availability = "true" if available else "false"
    if not update or not update.get("available"):
        return (f'<div class="aee-update-status" data-tone="{_html.escape(tone)}">'
                f'<span data-update-available="{availability}" hidden></span>'
                f'{safe_message}{progress or ""}</div>')
    version = _html.escape(str(update.get("version", "")))
    published = _html.escape(str(update.get("published_at", "")))
    notes = _html.escape(str(update.get("release_notes", "")))
    details = f'<div class="aee-update-notes">{notes}</div>' if notes else ""
    date = f'<span>Published {published}</span>' if published else ""
    return (f'<div class="aee-update-status" data-tone="available">'
            '<span data-update-available="true" hidden></span>'
            f'<strong>Version {version} is ready.</strong> {date}'
            f'{details}{progress or ""}</div>')


_ACCOUNT_SETUP_SQL = """create table if not exists public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  display_name text not null default '',
  avatar_path text,
  updated_at timestamptz not null default now()
);

alter table public.profiles enable row level security;
grant select, insert, update on public.profiles to authenticated;

drop policy if exists "Users can read their own profile" on public.profiles;
create policy "Users can read their own profile"
  on public.profiles for select to authenticated
  using ((select auth.uid()) = id);
drop policy if exists "Users can create their own profile" on public.profiles;
create policy "Users can create their own profile"
  on public.profiles for insert to authenticated
  with check ((select auth.uid()) = id);
drop policy if exists "Users can update their own profile" on public.profiles;
create policy "Users can update their own profile"
  on public.profiles for update to authenticated
  using ((select auth.uid()) = id)
  with check ((select auth.uid()) = id);

insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values ('profile-photos', 'profile-photos', false, 5242880,
        array['image/png', 'image/jpeg', 'image/webp'])
on conflict (id) do update set public = false, file_size_limit = 5242880,
  allowed_mime_types = excluded.allowed_mime_types;

drop policy if exists "Users can read their own profile photos" on storage.objects;
create policy "Users can read their own profile photos"
  on storage.objects for select to authenticated
  using (bucket_id = 'profile-photos' and (storage.foldername(name))[1] = (select auth.uid())::text);
drop policy if exists "Users can upload their own profile photos" on storage.objects;
create policy "Users can upload their own profile photos"
  on storage.objects for insert to authenticated
  with check (bucket_id = 'profile-photos' and (storage.foldername(name))[1] = (select auth.uid())::text);
drop policy if exists "Users can update their own profile photos" on storage.objects;
create policy "Users can update their own profile photos"
  on storage.objects for update to authenticated
  using (bucket_id = 'profile-photos' and (storage.foldername(name))[1] = (select auth.uid())::text)
  with check (bucket_id = 'profile-photos' and (storage.foldername(name))[1] = (select auth.uid())::text);
"""


def _account_status_html(message, tone="neutral", profile_saved=False):
    safe_message = _html.escape(str(message))
    hidden = " hidden" if not safe_message else ""
    saved = ' data-profile-saved="true"' if profile_saved else ""
    return (f'<div class="aee-account-status" data-kind="{_html.escape(tone)}" '
            f'role="status"{saved}{hidden}>{safe_message}</div>')


def _account_profile_html(session=None, profile=None, avatar_url=""):
    if not session:
        profile = profile if isinstance(profile, dict) else {}
        name = str(profile.get("display_name") or "")
        avatar_url = str(profile.get("avatar_url") or "")
        initials = "".join(part[0] for part in name.split()[:2] if part) or "G"
        avatar = (
            f'<img src="{_html.escape(avatar_url, quote=True)}" alt="" '
            'referrerpolicy="no-referrer">'
            if avatar_url else _html.escape(initials[:2].upper())
        )
        return ('''<section class="aee-account-profile" data-account-type="guest"
          aria-label="Guest account">
          <div class="aee-account-avatar">''' + avatar + '''</div>
          <div class="aee-account-identity">
            <span class="aee-account-eyebrow">Local profile</span><strong>''' +
                _html.escape(name or "Guest mode") + '''</strong>
            <span class="aee-account-detail">Your profile stays on this device</span>
            <span class="aee-account-mode">GUEST · LOCAL ONLY</span></div>
        </section>''')
    profile = profile if isinstance(profile, dict) else {}
    name = str(profile.get("display_name") or session.get("display_name") or "Your account")
    email = str(session.get("email") or "")
    initials = "".join(part[0] for part in name.split()[:2] if part) or "A"
    avatar = (f'<img src="{_html.escape(avatar_url, quote=True)}" alt="" '
              'referrerpolicy="no-referrer">') if avatar_url else _html.escape(initials[:2].upper())
    auth_provider = _account_client.session_auth_provider(session)
    if auth_provider == "google":
        signed_in_badge = ('''<span class="aee-account-mode" data-provider="google"
          aria-label="Signed in with Google">
          <svg viewBox="0 0 48 48" aria-hidden="true" focusable="false">
            <path fill="#4285F4" d="M43.6 24.5c0-1.4-.1-2.8-.4-4.1H24v7.8h11.1c-.5 2.5-1.9 4.6-4 6v5h6.5c3.8-3.5 6-8.6 6-14.7z"/>
            <path fill="#34A853" d="M24 44c5.4 0 10-1.8 13.4-4.8l-6.5-5c-1.8 1.2-4.1 2-6.9 2-5.3 0-9.8-3.6-11.4-8.4H6v5.2C9.4 39.6 16.1 44 24 44z"/>
            <path fill="#FBBC05" d="M12.6 27.8a12 12 0 0 1 0-7.6V15H6a20 20 0 0 0 0 18l6.6-5.2z"/>
            <path fill="#EA4335" d="M24 11.8c3 0 5.7 1 7.8 3.1l5.8-5.8C34.1 5.8 29.4 4 24 4 16.1 4 9.4 8.4 6 15l6.6 5.2c1.6-4.8 6.1-8.4 11.4-8.4z"/>
          </svg><span>SIGNED IN WITH GOOGLE</span></span>''')
    else:
        signed_in_badge = '<span class="aee-account-mode">SIGNED IN</span>'
    created_at = str(session.get("created_at") or "")
    joined = ""
    if created_at:
        try:
            joined_date = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            joined = joined_date.strftime("%b %Y")
        except ValueError:
            log.warning("The saved account creation date is invalid.")
    joined_html = (
        f'<span class="aee-account-joined">Member since {_html.escape(joined)}</span>'
        if joined else ""
    )
    return (f'<section class="aee-account-profile" data-account-type="online" '
            f'aria-label="Signed-in account">'
            f'<div class="aee-account-avatar">{avatar}</div>'
            f'<div class="aee-account-identity">'
            f'<span class="aee-account-eyebrow">Account profile</span>'
            f'<strong>{_html.escape(name)}</strong>'
            f'<span class="aee-account-detail">{_html.escape(email)}</span>'
            f'{joined_html}'
            f'{signed_in_badge}</div>'
            f'</section>')


def _account_profile_name_value(session):
    return str(session.get("display_name", "")) if isinstance(session, dict) else ""


def _account_access_html(ready):
    status = "ready" if ready else "required"
    return f'<span data-aee-access-status="{status}" hidden></span>'


def _account_auth_intro_html(mode="signup"):
    signing_in = mode == "signin"
    title = "Log in" if signing_in else "Create your account"
    copy = ("Log in to your account and continue managing your evidence projects, "
            "analysis, and progress right where you left off."
            if signing_in else
            "Create an account to start reviewing video evidence and keep your work organized.")
    return ('''<section class="aee-account-intro">
      <h2>''' + _html.escape(title) + '''</h2>
      <p>''' + _html.escape(copy) + '''</p>
    </section>''')


def _account_auth_mode_ui(mode):
    next_mode = "signin" if mode != "signin" else "signup"
    signing_in = next_mode == "signin"
    toggle_label = ("Don't have an account? Sign up" if signing_in else
                    "Already have an account? Log in")
    return (next_mode, _account_auth_intro_html(next_mode),
            gr.update(visible=not signing_in),
            gr.update(visible=not signing_in),
            gr.update(visible=signing_in),
            gr.update(value=toggle_label))


def _account_initial_session():
    try:
        session = _account_client.load_session()
        guest_mode = not session and _account_client.guest_mode_selected()
        guest_profile = {}
        status = ("Saved account session restored." if session else
                  "Guest mode restored. Your app data stays on this computer." if guest_mode else
                  "")
        tone = "success" if session or guest_mode else "neutral"
        if guest_mode:
            try:
                guest_profile = _account_client.load_guest_profile()
            except _account_client.AccountError as exc:
                log.warning("Could not restore guest profile: %s", exc)
                status = str(exc)
                tone = "error"
        settings_initial = session if session else guest_profile
        return (session, _account_status_html(status, tone),
                _account_profile_html(session, guest_profile), bool(session or guest_mode),
                guest_mode, guest_profile.get("display_name", ""),
                _account_profile_settings_values(settings_initial, session))
    except _account_client.AccountError as exc:
        log.warning("Could not restore account session: %s", exc)
        return (None, _account_status_html(str(exc), "error"),
                _account_profile_html(), False, False, "",
                _account_profile_settings_values({}, None))


def _account_profile_settings_values(profile, session=None):
    profile = profile if isinstance(profile, dict) else {}
    email = str((session or {}).get("email") or "") if session else ""
    return {
        "display_name": str(profile.get("display_name") or ""),
        "username": str(profile.get("username") or ""),
        "email": email,
    }


def _account_profile_settings_values_ui(session):
    if session:
        profile = session
    elif _account_client.guest_mode_selected():
        profile = _account_client.load_guest_profile()
    else:
        profile = {}
    return tuple(_account_profile_settings_values(profile, session).values())


def _account_profile_email_visibility_ui(session):
    return gr.update(visible=bool(session))


def _account_connection_setup_visibility_ui():
    try:
        return gr.update(visible=not _account_client.config_is_ready())
    except _account_client.AccountError as exc:
        log.warning("Could not verify the saved account connection: %s", exc)
        return gr.update(visible=True)


def _account_google_sign_in_visibility_ui():
    try:
        return gr.update(visible=_account_client.config_is_ready())
    except _account_client.AccountError as exc:
        log.warning("Google sign-in is unavailable without a valid connection: %s", exc)
        return gr.update(visible=False)


def _account_profile_settings_refresh_ui(session):
    if not session:
        return tuple(gr.skip() for _ in range(3))
    return _account_profile_settings_values_ui(session)


def _account_render_signed_in(session):
    try:
        profile, avatar_url, refreshed = _account_client.current_profile(session)
        return _account_profile_html(refreshed, profile, avatar_url), refreshed, None
    except _account_client.AccountError as exc:
        log.warning("Could not load account profile: %s", exc)
        return _account_profile_html(session), session, str(exc)


def _account_sign_in_ui(email, password, session, access_ready):
    try:
        session = _account_client.sign_in(email, password)
        profile_html, session, profile_error = _account_render_signed_in(session)
        message = "Welcome back."
        if profile_error:
            message += f" Sign-in succeeded, but the profile could not be loaded: {profile_error}"
        return (_account_status_html(message, "success" if not profile_error else "error"),
                profile_html, session, _account_profile_name_value(session),
                gr.update(visible=False), gr.update(visible=True),
                _account_access_html(True), True, gr.update(visible=True),
                gr.update(visible=True), gr.update(visible=False))
    except _account_client.AccountError as exc:
        return (_account_status_html(str(exc), "error"), _account_profile_html(session),
                session, _account_profile_name_value(session), gr.update(visible=not session),
                gr.update(visible=bool(session)), _account_access_html(access_ready),
                access_ready, gr.update(visible=bool(session)),
                gr.update(visible=bool(session)), gr.update(visible=not bool(session)))
    except Exception as exc:
        log.exception("Email sign-in failed")
        return (_account_status_html(f"Sign-in failed: {exc}", "error"),
                _account_profile_html(session), session, _account_profile_name_value(session),
                gr.update(visible=not session), gr.update(visible=bool(session)),
                _account_access_html(access_ready), access_ready,
                gr.update(visible=bool(session)), gr.update(visible=bool(session)),
                gr.update(visible=not bool(session)))


def _account_google_start_ui():
    try:
        transaction_id = _account_client.start_google_sign_in()
        return (transaction_id,
                _account_status_html(
                    "Google sign-in opened in your browser. Complete it there to continue.",
                    "neutral"),
                gr.update(interactive=False), gr.update(interactive=False),
                gr.update(active=True))
    except _account_client.AccountError as exc:
        return (None, _account_status_html(str(exc), "error"),
                gr.update(interactive=True), gr.update(interactive=True),
                gr.update(active=False))
    except Exception as exc:
        log.exception("Could not start Google sign-in")
        return (None, _account_status_html(f"Could not start Google sign-in: {exc}", "error"),
                gr.update(interactive=True), gr.update(interactive=True),
                gr.update(active=False))


def _account_google_poll_ui(transaction_id, session, access_ready):
    if not transaction_id:
        return (None, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                gr.skip(), gr.skip(), gr.skip(), gr.update(active=False),
                gr.update(interactive=True), gr.skip(), gr.skip(), gr.skip())
    try:
        google_session = _account_client.finish_google_sign_in(transaction_id)
        if google_session is None:
            return (transaction_id, gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                    gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                    gr.update(active=True), gr.skip(), gr.skip(), gr.skip(),
                    gr.skip())
        profile_html, google_session, profile_error = _account_render_signed_in(
            google_session)
        message = "Signed in with Google."
        if profile_error:
            message += f" Sign-in succeeded, but the profile could not be loaded: {profile_error}"
        return (None, _account_status_html(
                    message, "success" if not profile_error else "error"),
                profile_html, google_session, _account_profile_name_value(google_session),
                gr.update(visible=False), gr.update(visible=True),
                _account_access_html(True), True, gr.update(active=False),
                gr.update(interactive=True), gr.update(visible=True),
                gr.update(visible=True), gr.update(visible=False))
    except _account_client.AccountError as exc:
        return (None, _account_status_html(str(exc), "error"), _account_profile_html(session),
                session, _account_profile_name_value(session), gr.update(visible=not session),
                gr.update(visible=bool(session)), _account_access_html(access_ready),
                access_ready, gr.update(active=False), gr.update(interactive=True),
                gr.update(visible=bool(session)), gr.update(visible=bool(session)),
                gr.update(visible=not bool(session)))
    except Exception as exc:
        log.exception("Google sign-in did not complete")
        return (None, _account_status_html(f"Google sign-in failed: {exc}", "error"),
                _account_profile_html(session), session, _account_profile_name_value(session),
                gr.update(visible=not session), gr.update(visible=bool(session)),
                _account_access_html(access_ready), access_ready,
                gr.update(active=False), gr.update(interactive=True),
                gr.update(visible=bool(session)), gr.update(visible=bool(session)),
                gr.update(visible=not bool(session)))


def _account_sign_up_ui(display_name, email, password, session, access_ready):
    try:
        new_session, message = _account_client.sign_up(display_name, email, password)
        if new_session:
            profile_html, new_session, profile_error = _account_render_signed_in(new_session)
            if profile_error:
                message += f" Profile setup could not be completed: {profile_error}"
            return (_account_status_html(message, "success" if not profile_error else "error"),
                    profile_html, new_session, _account_profile_name_value(new_session),
                    gr.update(visible=False), gr.update(visible=True),
                    _account_access_html(True), True, gr.update(visible=True),
                    gr.update(visible=True), gr.update(visible=False))
        return (_account_status_html(message, "success"), _account_profile_html(session),
                session, _account_profile_name_value(session), gr.update(visible=not session),
                gr.update(visible=bool(session)), _account_access_html(access_ready),
                access_ready, gr.update(visible=bool(session)),
                gr.update(visible=bool(session)), gr.update(visible=not bool(session)))
    except _account_client.AccountError as exc:
        return (_account_status_html(str(exc), "error"), _account_profile_html(session),
                session, _account_profile_name_value(session), gr.update(visible=not session),
                gr.update(visible=bool(session)), _account_access_html(access_ready),
                access_ready, gr.update(visible=bool(session)),
                gr.update(visible=bool(session)), gr.update(visible=not bool(session)))
    except Exception as exc:
        log.exception("Email account creation failed")
        return (_account_status_html(f"Account creation failed: {exc}", "error"),
                _account_profile_html(session), session, _account_profile_name_value(session),
                gr.update(visible=not session), gr.update(visible=bool(session)),
                _account_access_html(access_ready), access_ready,
                gr.update(visible=bool(session)), gr.update(visible=bool(session)),
                gr.update(visible=not bool(session)))


def _account_guest_ui(session, access_ready=True):
    try:
        message = _account_client.sign_out(session)
        profile = _account_client.load_guest_profile()
        google_available = _account_google_sign_in_visibility_ui()["visible"]
        return (_account_status_html(message, "neutral"),
                _account_profile_html(profile=profile), None,
                profile.get("display_name", ""), gr.update(visible=False), gr.update(visible=True),
                _account_access_html(True), True, gr.update(visible=True),
                gr.update(visible=False), gr.update(visible=google_available))
    except _account_client.AccountError as exc:
        return (_account_status_html(str(exc), "error"), _account_profile_html(session),
                session, _account_profile_name_value(session), gr.update(visible=not session),
                gr.update(visible=bool(session)), _account_access_html(access_ready),
                access_ready, gr.update(visible=bool(session)),
                gr.update(visible=bool(session)), gr.update(visible=not bool(session)))
    except Exception as exc:
        log.exception("Could not switch to guest mode")
        return (_account_status_html(f"Could not switch to guest mode: {exc}", "error"),
                _account_profile_html(session), session, _account_profile_name_value(session),
                gr.update(visible=not session), gr.update(visible=bool(session)),
                _account_access_html(access_ready), access_ready,
                gr.update(visible=bool(session)), gr.update(visible=bool(session)),
                gr.update(visible=not bool(session)))


def _account_change_account_ui(session):
    try:
        message = _account_client.change_account(session)
        return (_account_status_html(message, "neutral"), _account_profile_html(), None,
                "", gr.update(visible=True), gr.update(visible=False),
                _account_access_html(False), False, gr.update(visible=False),
                gr.update(visible=False), gr.update(visible=False))
    except _account_client.AccountError as exc:
        return (_account_status_html(str(exc), "error"), _account_profile_html(session),
                session, _account_profile_name_value(session), gr.update(visible=not session),
                gr.update(visible=bool(session)), _account_access_html(bool(session)), bool(session),
                gr.update(visible=bool(session)), gr.update(visible=bool(session)),
                gr.update(visible=not bool(session)))
    except Exception as exc:
        log.exception("Could not change account")
        return (_account_status_html(f"Could not change account: {exc}", "error"),
                _account_profile_html(session), session, _account_profile_name_value(session),
                gr.update(visible=not session), gr.update(visible=bool(session)),
                _account_access_html(bool(session)), bool(session),
                gr.update(visible=bool(session)), gr.update(visible=bool(session)),
                gr.update(visible=not bool(session)))


def _account_save_name_ui(display_name, session):
    try:
        if not session:
            profile = _account_client.save_guest_display_name(display_name)
            return (_account_status_html(
                        "Guest profile saved on this computer.", "success"),
                    _account_profile_html(profile=profile), None,
                    profile.get("display_name", ""))
        profile, session = _account_client.save_display_name(session, display_name)
        profile_html, session, profile_error = _account_render_signed_in(session)
        message = "Profile name saved."
        if profile_error:
            message += f" The profile photo could not be refreshed: {profile_error}"
        return (_account_status_html(message, "success" if not profile_error else "error"),
                profile_html, session, str(profile.get("display_name", "")))
    except _account_client.AccountError as exc:
        return (_account_status_html(str(exc), "error"), _account_profile_html(session),
                session, str(display_name or ""))
    except Exception as exc:
        log.exception("Could not save the account profile")
        return (_account_status_html(f"Could not save profile: {exc}", "error"),
                _account_profile_html(session), session, str(display_name or ""))


def _account_save_profile_settings_ui(display_name, username, email, session):
    try:
        if not session:
            profile = _account_client.save_guest_profile_settings(
                display_name, username, email,
            )
            message = "Profile details saved on this computer."
            return (_account_status_html(message, "success", profile_saved=True),
                    _account_profile_html(profile=profile), None)

        profile, session, email_pending = _account_client.save_profile_settings(
            session, display_name, username, email,
        )
        profile_html, session, refresh_error = _account_render_signed_in(session)
        message = "Profile details saved to your account."
        if email_pending:
            message += (
                f" Confirm the email change using the link sent to {email.strip()}."
            )
        if refresh_error:
            message += f" The updated profile could not be refreshed: {refresh_error}"
        return (_account_status_html(
                    message, "error" if refresh_error else "success",
                    profile_saved=not refresh_error),
                profile_html, session)
    except _account_client.AccountError as exc:
        return (_account_status_html(str(exc), "error"),
                _account_profile_html(session), session)
    except Exception as exc:
        log.exception("Could not save profile settings")
        return (_account_status_html(f"Could not save profile settings: {exc}", "error"),
                _account_profile_html(session), session)


def _account_photo_editor_html():
    return '''<section id="aee-photo-editor-modal" class="aee-photo-editor-modal"
        aria-label="Edit profile photo" aria-modal="true" role="dialog" hidden>
      <div class="aee-photo-editor-window">
        <header class="aee-photo-editor-header">
        <div class="aee-photo-editor-heading">
          <button class="aee-photo-editor-back" type="button"
            data-aee-photo-cancel aria-label="Cancel photo edit">‹</button>
          <strong class="aee-photo-editor-title">Edit media</strong>
        </div>
        <button class="aee-photo-editor-apply" type="button"
          data-aee-photo-apply>Apply</button>
      </header>
      <div class="aee-photo-editor-stage">
        <canvas id="aee-photo-crop-canvas" class="aee-photo-crop-canvas"
          tabindex="0" role="img"
          aria-label="Drag to position the photo inside the square crop"></canvas>
        <p id="aee-photo-editor-error" class="aee-photo-editor-error"
          role="alert" hidden></p>
      </div>
      <footer class="aee-photo-editor-zoom" aria-label="Photo zoom">
        <button class="aee-photo-zoom-button" type="button"
          data-aee-photo-zoom="-0.1" aria-label="Zoom out">−</button>
        <input id="aee-photo-zoom-range" class="aee-photo-zoom-range"
          type="range" min="1" max="3" step="0.01" value="1"
          aria-label="Zoom photo">
        <button class="aee-photo-zoom-button" type="button"
          data-aee-photo-zoom="0.1" aria-label="Zoom in">+</button>
      </footer>
      </div>
    </section>
    <input id="aee-photo-file-input" class="aee-photo-hidden-file" type="file"
      accept="image/jpeg,image/png,image/webp,.jpg,.jpeg,.png,.webp"
      aria-label="Choose a profile photo">'''


def _account_apply_photo_data_ui(data_url, session):
    try:
        header = "data:image/jpeg;base64,"
        max_encoded_length = ((_account_client.MAX_AVATAR_BYTES + 2) // 3) * 4
        if not isinstance(data_url, str) or len(data_url) > len(header) + max_encoded_length:
            raise _account_client.AccountError(
                "The edited profile photo is empty or larger than 5 MB."
            )
        if not data_url.startswith(header):
            raise _account_client.AccountError(
                "The edited profile photo could not be read. Please try again."
            )
        encoded = data_url[len(header):]
        if not re.fullmatch(r"[A-Za-z0-9+/]*={0,2}", encoded):
            raise _account_client.AccountError(
                "The edited profile photo data is invalid. Please try again."
            )
        image_bytes = base64.b64decode(encoded, validate=True)
        if not image_bytes or len(image_bytes) > _account_client.MAX_AVATAR_BYTES:
            raise _account_client.AccountError(
                "The edited profile photo is empty or larger than 5 MB."
            )
        _account_client._validate_profile_photo(".jpg", image_bytes)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as photo:
            photo.write(image_bytes)
            image_path = photo.name
        try:
            status, profile_html, session = _account_upload_photo_ui(image_path, session)
        finally:
            try:
                Path(image_path).unlink(missing_ok=True)
            except OSError:
                log.warning("Could not remove temporary profile photo: %s", image_path)
        if 'data-kind="success"' in status:
            result = gr.update(value=(
                f'<span data-photo-applied="true" data-version="{time.time_ns()}"></span>'
            ))
            return status, profile_html, session, result
        return status, profile_html, session, gr.update(
            value='<span data-photo-applied="false"></span>'
        )
    except _account_client.AccountError as exc:
        return (_account_status_html(str(exc), "error"), gr.skip(), gr.skip(),
                gr.update(value='<span data-photo-applied="false"></span>'))
    except (ValueError, base64.binascii.Error) as exc:
        log.warning("Could not decode edited profile photo: %s", exc)
        return (_account_status_html(
                    "The edited profile photo data is invalid. Please try again.",
                    "error"),
                gr.skip(), gr.skip(),
                gr.update(value='<span data-photo-applied="false"></span>'))
    except OSError as exc:
        log.exception("Could not prepare the edited profile photo")
        return (_account_status_html(
                    f"Could not prepare the edited profile photo: {exc}",
                    "error"),
                gr.skip(), gr.skip(),
                gr.update(value='<span data-photo-applied="false"></span>'))


def _account_upload_photo_ui(image_path, session):
    try:
        if not session:
            profile = _account_client.save_guest_profile_photo(image_path)
            return (_account_status_html(
                        "Guest profile photo saved on this computer.", "success",
                        profile_saved=True),
                    _account_profile_html(profile=profile), None)
        profile, session = _account_client.upload_profile_photo(session, image_path)
        profile_html, session, profile_error = _account_render_signed_in(session)
        message = "Profile photo updated."
        if profile_error:
            message += f" The updated photo could not be displayed: {profile_error}"
        return (_account_status_html(
                    message, "success" if not profile_error else "error",
                    profile_saved=not profile_error),
                profile_html, session)
    except _account_client.AccountError as exc:
        return _account_status_html(str(exc), "error"), _account_profile_html(session), session
    except Exception as exc:
        log.exception("Could not update the account photo")
        return (_account_status_html(f"Could not update profile photo: {exc}", "error"),
                _account_profile_html(session), session)


def _account_refresh_profile_ui(session):
    if not session:
        try:
            profile = _account_client.load_guest_profile()
            message = "Guest profile refreshed from this computer."
            return (_account_status_html(message, "neutral"),
                    _account_profile_html(profile=profile), None,
                    profile.get("display_name", ""))
        except _account_client.AccountError as exc:
            return (_account_status_html(str(exc), "error"),
                    _account_profile_html(), None, "")
    profile_html, session, profile_error = _account_render_signed_in(session)
    message = profile_error or "Account profile refreshed."
    return (_account_status_html(message, "error" if profile_error else "success"),
            profile_html, session, _account_profile_name_value(session))


def _account_save_config_ui(project_url, anon_key, session, access_ready):
    connection_verified = False
    settings_saved = False
    try:
        _account_client.test_connection(project_url, anon_key)
        connection_verified = True
        changed = _account_client.save_config(project_url, anon_key)
        settings_saved = True
        message = (
            "Supabase Auth connection verified. You can now sign in or create an "
            "account. Profile and photo features require the project setup SQL."
        )
        if changed:
            session = None
        guest_mode = _account_client.guest_mode_selected()
        profile_visible = bool(session) or guest_mode
        guest_profile = (_account_client.load_guest_profile()
                         if guest_mode and not session else {})
        access_ready = profile_visible
        return (_account_status_html(message, "success" if "verified" in message else "error"),
                _account_profile_html(session, guest_profile), session,
                _account_profile_name_value(session) or guest_profile.get("display_name", ""),
                gr.update(visible=not profile_visible),
                gr.update(visible=profile_visible),
                _account_access_html(access_ready), access_ready,
                gr.update(visible=profile_visible),
                gr.update(visible=bool(session)),
                gr.update(visible=guest_mode and not bool(session)),
                gr.update(visible=True))
    except _account_client.AccountError as exc:
        profile_visible = bool(session or access_ready)
        if not connection_verified:
            message = f"Supabase connection failed; settings were not saved: {exc}"
        elif not settings_saved:
            message = f"Supabase connection verified, but settings could not be saved: {exc}"
        else:
            message = (
                f"Supabase connection verified and saved, but account status could "
                f"not be refreshed: {exc}"
            )
        return (_account_status_html(
                    message, "error"), gr.skip(),
                session, gr.skip(), gr.update(visible=not profile_visible),
                gr.update(visible=profile_visible), _account_access_html(access_ready),
                access_ready, gr.update(visible=profile_visible),
                gr.update(visible=bool(session)), gr.skip(),
                _account_google_sign_in_visibility_ui())
    except Exception as exc:
        log.exception("Could not configure account connection")
        profile_visible = bool(session or access_ready)
        if not connection_verified:
            message = f"Supabase connection failed; settings were not saved: {exc}"
        elif not settings_saved:
            message = f"Supabase connection verified, but settings could not be saved: {exc}"
        else:
            message = (
                f"Supabase connection verified and saved, but account status could "
                f"not be refreshed: {exc}"
            )
        return (_account_status_html(
                    message, "error"),
                gr.skip(), session, gr.skip(),
                gr.update(visible=not profile_visible), gr.update(visible=profile_visible),
                _account_access_html(access_ready), access_ready,
                gr.update(visible=profile_visible), gr.update(visible=bool(session)),
                gr.skip(), _account_google_sign_in_visibility_ui())


def _check_updates_ui():
    for fraction, description, update, error, finished in _run_update_task(
        _update_client.check_for_updates, "Checking for updates"
    ):
        if error:
            if isinstance(error, _update_client.UpdateNotConfigured):
                tone = "neutral"
            else:
                tone = "error"
                if not isinstance(error, _update_client.UpdateError):
                    log.error(
                        "Update check failed",
                        exc_info=(type(error), error, error.__traceback__),
                    )
            yield (_update_status_html(
                str(error) if isinstance(error, _update_client.UpdateError)
                else f"Update check failed: {error}",
                tone, progress=_update_progress_html(fraction, description)),
                gr.update(visible=False), None, gr.update(visible=False))
            return
        if not finished:
            yield (_update_status_html(
                "Checking for the latest release…",
                "neutral", progress=_update_progress_html(fraction, description)),
                gr.update(), gr.skip(), gr.update())
            continue
        if update["available"]:
            if not update.get("package_url"):
                yield (_update_status_html(
                    "This release does not include a compatible in-app update package.",
                    "error"), gr.update(visible=False), update, gr.update(visible=False))
                return
            yield (_update_status_html("A new version is available.", "available", update),
                   gr.update(visible=True), update, gr.update(visible=False))
        else:
            current = _html.escape(update["installed_version"])
            yield (_update_status_html(
                f"No updates right now. You are using the latest version ({current}).",
                "success"), gr.update(visible=False), update, gr.update(visible=False))
        return


def _download_update_ui(update):
    for fraction, description, staged, error, finished in _run_update_task(
        lambda report: _update_client.download_and_stage_update(update, report),
        "Preparing update download",
    ):
        if error:
            if isinstance(error, _update_client.UpdateError):
                message = str(error)
            else:
                log.error(
                    "Could not download and prepare the update",
                    exc_info=(type(error), error, error.__traceback__),
                )
                message = f"Could not prepare the update: {error}"
            yield (_update_status_html(
                message, "error",
                update_available=bool(update and update.get("available"))),
                gr.update(visible=False), update)
        elif not finished:
            unknown_size = "(size unknown)" in description
            yield (_update_status_html(
                "Downloading and verifying update…", "neutral",
                update_available=bool(update and update.get("available")),
                progress=_update_progress_html(
                    fraction, description, indeterminate=unknown_size)),
                gr.update(), gr.skip())
        else:
            yield (_update_status_html(
            "The update is downloaded, checksum-verified, and ready to apply. "
            "Select Install Update to see the in-app installation progress.",
            "success", update_available=True,
            progress=_update_progress_html(
                1, "Download complete and verified")),
            gr.update(visible=True), staged)


def _apply_update_ui(staged):
    for fraction, description, result, error, finished in _run_update_task(
        lambda report: _update_client.apply_staged_update(staged, report),
        "Installing update",
    ):
        if error:
            if isinstance(error, _update_client.UpdateError):
                message = str(error)
            else:
                log.error(
                    "Could not start applying the update",
                    exc_info=(type(error), error, error.__traceback__),
                )
                message = f"Could not start the update: {error}"
            yield (_update_status_html(message, "error"),
                   gr.update(visible=True), gr.update(visible=False))
        elif not finished:
            yield (_update_status_html(
                "Applying update files…", "neutral",
                progress=_update_progress_html(fraction, description)),
                gr.skip(), gr.update(visible=False))
        else:
            yield (_update_status_html(
                f"Update {_html.escape(str(staged.get('version', '')))} installed successfully. "
                "The files are updated. Select Restart App below to finish and load the new version.",
                "success", progress=_update_progress_html(
                    1, "Installation complete")),
                gr.update(visible=False), gr.update(visible=True))


def _restart_app_ui():
    try:
        _update_client.restart_app()
        return (_update_status_html("Restarting AI Evidence Extractor…", "success"),
                gr.update(visible=False))
    except _update_client.UpdateError as exc:
        return _update_status_html(str(exc), "error"), gr.update(visible=True)
    except Exception as exc:
        log.exception("Could not restart the app")
        return (_update_status_html(f"Could not restart: {exc}", "error"),
                gr.update(visible=True))


def _updates_intro_html():
    try:
        app_dir = Path(os.environ.get("AEE_APP_DIR", Path(__file__).resolve().parent))
        installed_version = (app_dir / "version.txt").read_text(
            encoding="utf-8"
        ).strip()
        if not _update_client.VERSION_RE.fullmatch(installed_version):
            raise ValueError("The installed app version file is invalid.")
        version = _html.escape(installed_version)
    except (OSError, ValueError):
        try:
            version = _html.escape(_update_client.current_version())
        except _update_client.UpdateError:
            version = "unknown"
    return f'''<section class="aee-update-card">
      <div class="aee-update-eyebrow"><span class="aee-update-dot"></span> Update center</div>
      <h2 class="aee-update-title">A smoother way to stay current.</h2>
      <div class="aee-update-copy">Check for a release, review it, and update without opening a setup file.
        The download and each installation step show live progress in this window.</div>
      <div class="aee-version-pill">Installed version&nbsp; v{version}</div>
      <div class="aee-update-steps" aria-label="How app updates work">
        <div class="aee-update-step"><span class="aee-update-step-number">01</span>
          <div><strong>Check</strong><span>Look for the latest release securely.</span></div></div>
        <div class="aee-update-step"><span class="aee-update-step-number">02</span>
          <div><strong>Download</strong><span>Track download and checksum verification in the app.</span></div></div>
        <div class="aee-update-step"><span class="aee-update-step-number">03</span>
          <div><strong>Apply</strong><span>Watch file-by-file progress, then restart to load the update.</span></div></div>
      </div>
    </section>'''


def _side_navigation_html():
    return '''<div class="aee-navigation">
      <button class="aee-nav-toggle" type="button" data-aee-nav-toggle
        aria-label="Open navigation" aria-expanded="false" title="Open navigation">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round">
          <path d="M4 6h16M4 12h16M4 18h16"/>
        </svg>
      </button>
      <div class="aee-nav-backdrop" data-aee-nav-backdrop></div>
      <aside class="aee-nav-drawer" aria-label="App navigation">
        <div class="aee-nav-head">
          <div class="aee-nav-logo" aria-hidden="true"><svg viewBox="0 0 48 48" fill="none"><path d="M24 4.5 28.4 18l13.1 6-13.1 5.2L24 43.5l-4.8-14.3L6 24l13.2-6L24 4.5Z" fill="#60a5fa" stroke="white" stroke-opacity=".8" stroke-width="1.4" stroke-linejoin="round"/><path d="m35.8 5 .9 3.1 3.1.9-3.1.9-.9 3.1-.9-3.1-3.1-.9 3.1-.9.9-3.1Z" fill="#c4b5fd"/></svg></div>
          <div class="aee-nav-brand"><strong>AI Evidence Extractor</strong><span>Project menu</span></div>
          <button class="aee-nav-close" type="button" data-aee-nav-close aria-label="Close navigation">×</button>
        </div>
        <div>
          <div class="aee-nav-group-label">Workspace</div>
          <div class="aee-nav-items">
            <button class="aee-nav-item" type="button" data-aee-nav-tab="Overview">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><rect x="3" y="3" width="8" height="8" rx="2"/><rect x="13" y="3" width="8" height="5" rx="2"/><rect x="13" y="10" width="8" height="11" rx="2"/><rect x="3" y="13" width="8" height="8" rx="2"/></svg>
              <span>Back to workspace</span>
            </button>
          </div>
        </div>
        <div>
          <div class="aee-nav-group-label">Project</div>
          <div class="aee-nav-items">
            <button class="aee-nav-item" type="button" data-aee-nav-tab="Account">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="4"/><path d="M4 20a8 8 0 0 1 16 0"/></svg>
              <span>Account</span>
            </button>
            <button class="aee-nav-item" type="button" data-aee-nav-tab="Detailed History">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 4h16v16H4z"/><path d="M8 8h8M8 12h8M8 16h5"/></svg>
              <span>Detailed History</span>
            </button>
            <button class="aee-nav-item" type="button" data-aee-nav-tab="Updates">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M20 7v5h-5M4 17v-5h5"/><path d="M6.1 9A7 7 0 0 1 18 6l2 6M4 12l2 6a7 7 0 0 0 11.9-3"/></svg>
              <span>Updates</span>
            </button>
            <button class="aee-nav-item" type="button" data-aee-nav-tab="About Us">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M9.6 9a2.5 2.5 0 1 1 4.3 1.7c-1.1 1-1.9 1.3-1.9 2.8M12 17h.01"/></svg>
              <span>About Us</span>
            </button>
            <button class="aee-nav-item" type="button" data-aee-nav-tab="Contact Us">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 5h16v14H4z"/><path d="m4 7 8 6 8-6"/></svg>
              <span>Contact Us</span>
            </button>
          </div>
        </div>
        <div class="aee-nav-spacer"></div>
        <div class="aee-nav-foot">AI-assisted video evidence review<br>Human review remains essential.</div>
      </aside>
    </div>'''


def _settings_navigation_html():
    return '''<nav class="aee-settings-nav" aria-label="Settings sections">
      <div class="aee-settings-nav-label">Settings</div>
      <button type="button" data-aee-settings-section="profile">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="8" r="4"/><path d="M4 20a8 8 0 0 1 16 0"/></svg>
        <span>Account</span>
      </button>
      <details class="aee-settings-language-control">
        <summary>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a15 15 0 0 1 0 18M12 3a15 15 0 0 0 0 18"/></svg>
          <span>Language</span>
          <svg class="aee-settings-language-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m6 9 6 6 6-6"/></svg>
        </summary>
        <div class="aee-settings-language-menu" role="group" aria-label="Choose language">
          <button type="button" aria-pressed="true" data-aee-language-option="English">English</button>
          <button type="button" aria-pressed="false" data-aee-language-option="Arabic">Arabic</button>
        </div>
        <select id="aee-language-selector" class="aee-settings-language-native"
                aria-label="Language" tabindex="-1" aria-hidden="true">
          <option value="English">English</option>
          <option value="Arabic">Arabic</option>
        </select>
      </details>
      <button type="button" data-aee-settings-section="security">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="5" y="10" width="14" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/></svg>
        <span>Security</span>
      </button>
    </nav>'''


def build_ui():
    info = Ensemble.get().info
    (account_session_initial, account_status_initial, account_profile_initial,
     account_access_ready_initial, account_guest_mode_initial,
     account_profile_name_initial, account_profile_settings_initial) = (
        _account_initial_session()
    )
    try:
        account_config_initial = _account_client.load_config()
    except _account_client.AccountError as exc:
        log.warning("Could not load account connection settings: %s", exc)
        account_config_initial = {"project_url": "", "anon_key": ""}
        account_status_initial = _account_status_html(str(exc), "error")
    account_connection_ready = bool(
        account_config_initial["project_url"] and account_config_initial["anon_key"]
    )
    with gr.Blocks(title="AI Evidence Extractor", analytics_enabled=False) as app:
        gr.HTML(_side_navigation_html(), elem_id="aee-nav-host")
        gr.HTML(_hero_html(info))
        gr.HTML(_workflow_html())

        with gr.Row(equal_height=False, elem_classes="aee-workspace"):
            with gr.Column(scale=5, min_width=440, elem_classes="aee-upload-shell"):
                gr.HTML(_upload_header_html())

                video_input = gr.UploadButton(
                    "Upload video",
                    file_count="multiple",
                    file_types=[
                        ".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv",
                        ".m4v", ".webm", ".mpg", ".mpeg", ".3gp", ".3g2",
                        ".mts", ".m2ts", ".ts", ".vob", ".ogv", ".rm",
                        ".rmvb", ".asf", ".divx", ".f4v", ".mxf",
                    ],
                    type="filepath",
                    elem_classes="aee-upload-trigger",
                )

                gr.HTML(_formats_html())
                gr.HTML(value="", elem_id="aee-upload-progress-host")
                high_accuracy = gr.Checkbox(
                    label="Enhanced accuracy (slower)", value=False,
                    info="Use test-time augmentation on supported models for difficult or small objects.")

                with gr.Row(elem_classes="aee-actions"):
                    reset_btn = gr.Button("← Reset", size="sm",
                                          elem_classes="aee-btn-reset")
                    process_btn = gr.Button("⚡ Run analysis",
                                            variant="primary", size="lg",
                                            elem_classes="aee-btn-run")

                files_meta = gr.HTML(value="")

            with gr.Column(scale=4, min_width=340, elem_classes="aee-config-shell"):
                ensemble_panel = gr.HTML(value=_ensemble_panel_html(info))

        progress_html = gr.HTML(value="", elem_id="aee-analysis-progress")

        with gr.Tabs(elem_classes="aee-tabs"):
            with gr.Tab("Overview"):
                with gr.Column(elem_id="aee-master-wrap",
                               elem_classes="aee-master-wrap"):
                    master_video = gr.Video(label="", height=400,
                        show_label=False, interactive=False)
                overview_html = gr.HTML(value=_empty_html())

            with gr.Tab("Captures"):
                cap_filter = gr.Dropdown(label="", choices=["All"], value="All",
                    interactive=True)
                gallery = gr.Gallery(label="", show_label=False,
                    columns=4, rows=3, height=560,
                    object_fit="cover", allow_preview=True)

            with gr.Tab("Video summaries"):
                with gr.Row():
                    sum_select = gr.Dropdown(label="Preview summary",
                        choices=[], value=None, interactive=True, scale=1)
                with gr.Row():
                    sum_video = gr.Video(show_label=False, height=420, scale=3)
                    with gr.Column(scale=2):
                        gr.HTML('<div class="aee-card-title">Per-object reels</div>'
                            '<div class="aee-card-sub">Download any clip below</div>')
                        sum_files = gr.File(label="", file_count="multiple",
                            show_label=False)

            with gr.Tab("Object tracking"):
                gr.HTML('<div class="aee-card-title">🧭 Object track analytics</div>'
                        '<div class="aee-card-sub">Per-track lifecycle, persistence, '
                        'threat level, and re-identification events aggregated '
                        'across all analysed videos.</div>')
                track_html = gr.HTML(value=Tracker().panel_html())
                track_table = gr.Dataframe(label="Track detail",
                                           interactive=False, wrap=True)
                with gr.Row():
                    track_csv_btn = gr.Button("⬇  Export tracks CSV", size="lg")
                track_csv = gr.File(show_label=False)

            with gr.Tab("Explainability"):
                gr.HTML('<div class="aee-card-title">🔍 Model explainability</div>'
                        '<div class="aee-card-sub">Class-agnostic activation heatmaps '
                        'per weapon detection. Heatmap-only and blended views are cached '
                        'for every detection.</div>')
                cam_html = gr.HTML(value=CAMCache().panel_html())
                cam_gallery = gr.Gallery(label="Activation heatmaps",
                    show_label=False, columns=4, rows=3, height=560,
                    object_fit="cover", allow_preview=True)

            with gr.Tab("Evidence log"):
                with gr.Row():
                    ev_search = gr.Textbox(label="Search",
                        placeholder="Filter by object, source, or video", scale=3)
                    ev_class = gr.Dropdown(label="Class", choices=["All"], value="All",
                                           interactive=True, scale=1)
                ev_table = gr.Dataframe(interactive=False, wrap=True)

            with gr.Tab("Reports & exports"):
                with gr.Row():
                    with gr.Column():
                        gr.HTML('<div class="aee-card-title">🎬 Master summary video</div>'
                            '<div class="aee-card-sub">Union of every detection.</div>')
                        master_dl = gr.DownloadButton(
                            value=_latest_published("*_evidence_summary.mp4"),
                            label="⬇  Download Master MP4",
                            variant="primary", size="lg")
                    with gr.Column():
                        gr.HTML('<div class="aee-card-title">📄 Forensic PDF report</div>'
                            '<div class="aee-card-sub">Evidence log + object-track '
                            'threat analytics.</div>')
                        pdf_dl = gr.DownloadButton(
                            value=_latest_published("*_evidence_report.pdf"),
                            label="⬇  Download PDF Report",
                            variant="primary", size="lg")
                with gr.Row():
                    csv_btn = gr.Button("⬇  Export detections CSV", size="lg")
                    json_btn = gr.Button("⬇  Export detections JSON", size="lg")
                csv_file = gr.File(show_label=False)
                json_file = gr.File(show_label=False)

            with gr.Tab("History"):
                gr.HTML('<div class="aee-card-title">Recent analyzed videos</div>'
                        '<div class="aee-card-sub">A simple overview of completed analyses saved on this computer.</div>')
                history_refresh = gr.Button("Refresh history", size="sm")
                history_table = gr.Dataframe(
                    value=_history_dataframe(), headers=HISTORY_COLUMNS,
                    datatype=["str", "str", "number", "number", "str", "str"],
                    interactive=False, wrap=True, show_row_numbers=True)

        history_items = _history_list_items()
        history_selected = history_items[0]["id"] if history_items else None
        history_initial = _history_detail_ui(history_selected)
        history_refresh_state = gr.State(0)
        with gr.Group(visible=True, elem_id="aee-detailed-history-window",
                      elem_classes=["aee-utility-window-shell", "aee-history-window"]):
            with gr.Column(elem_classes=["wrap"]):
                with gr.Row(elem_classes=["aee-utility-window-header"]):
                    gr.HTML('''<header class="aee-project-page-heading">
                      <span class="aee-project-page-mark" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M5 4h14v16H5z"/><path d="M8 8h8M8 12h8M8 16h5"/></svg></span>
                      <div><small>ANALYSIS ARCHIVE</small><h1>Detailed History</h1>
                        <p>Browse saved analyses, evidence files, detection records, and object tracks.</p></div>
                    </header>''')
                    detailed_history_close = gr.Button(
                        "Back to app", size="sm",
                        elem_classes=["aee-utility-window-close"],
                        elem_id="aee-detailed-history-close")
                with gr.Column(elem_classes=["aee-history-page"]):
                    gr.HTML(
                        '<header class="aee-history-heading">'
                        '<div><h2>Your analysis archive</h2>'
                        '<p>Browse your analyzed videos in order. Open details, download an analysis ZIP, '
                        'or remove an item from this computer.</p>'
                        '</div></header>')
                    with gr.Row(elem_classes=["aee-history-toolbar"]):
                        detailed_history_refresh = gr.Button(
                            "Refresh history", size="sm", scale=0,
                            elem_classes=["aee-history-refresh"], elem_id="aee-history-refresh")
                        all_history_zip = gr.DownloadButton(
                            label="Download all history ZIP",
                            variant="secondary", size="sm",
                            elem_classes=["aee-history-download-all"],
                            elem_id="aee-history-download-all", scale=0)
                        clear_history_btn = gr.Button(
                            "Clear all history", size="sm", variant="stop",
                            elem_classes=["aee-history-clear"], elem_id="aee-history-clear",
                            scale=0)
                    history_zip_status = gr.HTML()
                    with gr.Column(elem_classes=["aee-history-list"]):
                        @gr.render(inputs=[history_refresh_state])
                        def render_history_rows(_refresh_count):
                            items = _history_list_items()
                            if not items:
                                gr.HTML(
                                    '<section class="aee-history-empty"><strong>No saved analyses yet</strong>'
                                    '<span>Completed video analyses will appear here.</span></section>'
                                )
                                return
                            for item in items:
                                with gr.Row(elem_classes=["aee-history-row"]):
                                    with gr.Column(elem_classes=["aee-history-row-info"], scale=1):
                                        gr.HTML(
                                            '<div class="aee-history-row-title">'
                                            f'{_html.escape(item["video"])}</div>'
                                            '<div class="aee-history-row-meta">'
                                            f'{_html.escape(item["analyzed"])}'
                                            f'{" · " + _html.escape(item["case"]) if item["case"] else ""}'
                                            f' · {item["detections"]} detections</div>'
                                        )
                                    with gr.Row(elem_classes=["aee-history-row-actions"], scale=0):
                                        details_btn = gr.Button(
                                            "Details", size="sm",
                                            elem_classes=["aee-history-details"],
                                            elem_id="aee-history-details")
                                        download_btn = gr.Button(
                                            "Download ZIP", size="sm",
                                            elem_classes=["aee-history-download"],
                                            elem_id="aee-history-download")
                                        remove_btn = gr.Button(
                                            "Remove", size="sm", variant="stop",
                                            elem_classes=["aee-history-remove"],
                                            elem_id="aee-history-remove")
                                details_btn.click(
                                    fn=lambda entry_id=item["id"]: _history_detail_ui(entry_id),
                                    inputs=[],
                                    outputs=[history_detail, history_gallery, history_files,
                                             history_detections, history_tracks],
                                    queue=False,
                                )
                                download_btn.click(
                                    fn=lambda entry_id=item["id"]: _history_zip_ui(entry_id),
                                    inputs=[],
                                    outputs=[all_history_zip, history_zip_status],
                                )
                                remove_btn.click(
                                    fn=lambda entry_id=item["id"], count=_refresh_count:
                                        _history_remove_ui(entry_id, count),
                                    inputs=[],
                                    outputs=[history_refresh_state, history_detail, history_gallery,
                                             history_files, history_detections, history_tracks,
                                             all_history_zip, history_zip_status],
                                )
                    history_detail = gr.HTML(value=history_initial[0])
                    history_gallery = gr.Gallery(
                        value=history_initial[1],
                        label="Captures and CAM evidence",
                        columns=4, rows=2, height=430, object_fit="contain",
                        allow_preview=True, elem_classes=["aee-history-gallery"])
                    history_files = gr.File(
                        value=history_initial[2],
                        label="Reports and generated files",
                        file_count="multiple", interactive=False,
                        elem_classes=["aee-history-files"])
                    gr.HTML(
                        '<div class="aee-history-section-heading"><h2>Detection records</h2>'
                        '<p>Complete saved detections for the selected analysis.</p></div>')
                    history_detections = gr.Dataframe(
                        value=history_initial[3], interactive=False, wrap=True,
                        show_row_numbers=True, elem_classes=["aee-history-detections"])
                    gr.HTML(
                        '<div class="aee-history-section-heading"><h2>Object tracks</h2>'
                        '<p>Saved per-object tracking and threat analytics.</p></div>')
                    history_tracks = gr.Dataframe(
                        value=history_initial[4], interactive=False, wrap=True,
                        show_row_numbers=True, elem_classes=["aee-history-detections"])
                    clear_history_btn.click(
                        fn=_history_clear_ui,
                        inputs=[history_refresh_state],
                        outputs=[history_refresh_state, history_detail, history_gallery,
                                 history_files, history_detections, history_tracks,
                                 history_zip_status],
                    )

        with gr.Group(visible=True, elem_id="aee-account-window",
                      elem_classes=["aee-utility-window-shell", "aee-account-shell"]):
            with gr.Column(elem_classes=["wrap"]):
                gr.HTML(
                    '<span data-aee-account-startup="required" hidden></span>'
                    if not account_access_ready_initial else
                    '<span data-aee-account-startup="ready" hidden></span>'
                )
                account_access_marker = gr.HTML(
                    value=_account_access_html(account_access_ready_initial))
                with gr.Row(elem_classes=["aee-utility-window-header"]):
                    gr.HTML('''<header class="aee-project-page-heading">
                      <span class="aee-project-page-mark" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="m19.4 15 .1.1 1.4 1.1-1.4 2.4-1.7-.7a8 8 0 0 1-1.7 1l-.3 1.8h-2.8l-.3-1.8a8 8 0 0 1-1.7-1l-1.7.7-1.4-2.4 1.4-1.1a8 8 0 0 1 0-2l-1.4-1.1 1.4-2.4 1.7.7a8 8 0 0 1 1.7-1l.3-1.8h2.8l.3 1.8a8 8 0 0 1 1.7 1l1.7-.7 1.4 2.4-1.4 1.1a8 8 0 0 1 0 2Z"/></svg></span>
                      <div><small>ACCOUNT CENTER</small><h1>Account</h1>
                        <p>Manage your profile, sign-in, and privacy.</p></div>
                    </header>''')
                    account_close_btn = gr.Button(
                        "Back to app", size="sm",
                        elem_classes=["aee-utility-window-close"], elem_id="aee-account-close")
                account_session = gr.State(account_session_initial)
                account_access_state = gr.State(account_access_ready_initial)
                account_auth_mode = gr.State("signup")
                account_status = gr.HTML(
                    value=account_status_initial, elem_id="aee-account-status")
                with gr.Group(
                    visible=not (bool(account_session_initial) or account_guest_mode_initial),
                    elem_classes=["aee-account-auth-form"],
                ) as account_auth_group:
                    account_auth_intro = gr.HTML(value=_account_auth_intro_html())
                    with gr.Group(elem_classes=["aee-account-name-field"]) as account_name_group:
                        account_display_name_input = gr.Textbox(
                            label="Your name", placeholder="Enter your name",
                            max_lines=1)
                    account_email_input = gr.Textbox(
                        label="Email address", type="email",
                        placeholder="you@example.com", max_lines=1)
                    account_password_input = gr.Textbox(
                        label="Password", type="password",
                        placeholder="Enter your password", max_lines=1)
                    with gr.Row(elem_classes=["aee-account-actions"]):
                        account_signup_btn = gr.Button(
                            "Create my account", variant="primary", size="lg",
                            elem_classes=["aee-account-submit"])
                        account_signin_btn = gr.Button(
                            "Log in", variant="primary", size="lg", visible=False,
                            elem_classes=["aee-account-submit"])
                        account_auth_mode_btn = gr.Button(
                            "Already have an account? Log in", variant="secondary",
                            elem_classes=["aee-account-mode-switch"])
                    gr.HTML('<div class="aee-account-divider"><span>or continue with</span></div>')
                    account_google_btn = gr.Button(
                        "Continue with Google", variant="secondary",
                        visible=account_connection_ready,
                        elem_classes=["aee-account-google"])
                    gr.HTML('<div class="aee-account-divider"><span>or</span></div>')
                    account_guest_btn = gr.Button(
                        "Continue as a guest", variant="secondary",
                        elem_classes=["aee-account-guest"])
                    gr.HTML('''<div class="aee-account-auth-privacy">
                      <strong>Your work stays private.</strong> Videos, analysis history, and reports
                      remain on this device, whether you create an account or continue as a guest.</div>''')
                    if not account_connection_ready:
                        gr.HTML('''<div class="aee-account-setup-hint">
                          To enable online sign-in on this computer, expand Connection setup below
                          and enter your Supabase project URL and public API key.</div>''')
                with gr.Accordion(
                    "Connection setup",
                    open=False,
                    visible=not account_connection_ready,
                    elem_classes=["aee-settings-panel", "aee-settings-card",
                                  "aee-account-settings", "aee-account-connection-settings"],
                ) as account_connection_setup:
                    gr.HTML('''<div class="aee-account-settings-copy">
                      <strong>Connect your Supabase project</strong>
                      <p>Copy the project URL and public publishable key (or legacy anon key)
                      from your Supabase project API settings. Never use a service-role or secret key.
                      Run the setup SQL below in the Supabase SQL Editor. The connection is tested
                      before these settings are saved, and the SQL is safe to rerun.</p>
                      <a href="https://github.com/othmanola/AI-Evidence-Extractor/blob/main/SUPABASE_SETUP.md"
                         target="_blank" rel="noopener noreferrer">View setup instructions</a>
                    </div>''')
                    account_project_url = gr.Textbox(
                        label="Supabase project URL",
                        value=account_config_initial["project_url"],
                        placeholder="https://your-project.supabase.co")
                    account_anon_key = gr.Textbox(
                        label="Supabase public API key",
                        type="password", value=account_config_initial["anon_key"],
                        placeholder="sb_publishable_… or legacy anon key")
                    account_save_config_btn = gr.Button(
                        "Save and test connection", variant="primary")
                    gr.Code(
                        value=_ACCOUNT_SETUP_SQL, language="sql",
                        label="Run in Supabase SQL Editor",
                        interactive=False, lines=25)
                with gr.Row(elem_classes=["aee-settings-layout"]):
                    with gr.Column(scale=1, min_width=180,
                                   elem_classes=["aee-settings-sidebar"]):
                        gr.HTML(_settings_navigation_html())
                    with gr.Column(scale=4, min_width=0,
                                   elem_classes=["aee-settings-main"]):
                        with gr.Group(
                            visible=bool(account_session_initial) or account_guest_mode_initial,
                            elem_classes=["aee-account-profile-form"],
                        ) as account_profile_group:
                            with gr.Group(
                                elem_id="aee-settings-profile-section",
                                elem_classes=["aee-settings-panel"],
                            ) as account_profile_settings:
                                with gr.Row(elem_classes=["aee-settings-profile-header"]):
                                    account_profile = gr.HTML(
                                        value=account_profile_initial,
                                        elem_id="aee-account-profile-source")
                                    with gr.Column(
                                        elem_classes=["aee-settings-photo-actions"]
                                    ):
                                        account_photo_picker = gr.Button(
                                            "Change photo",
                                            elem_id="aee-photo-change-button",
                                            elem_classes=["aee-settings-photo-button"])
                                account_photo_editor = gr.HTML(
                                    value=_account_photo_editor_html(),
                                    elem_id="aee-photo-editor-root")
                                account_photo_payload = gr.Textbox(
                                    value="", elem_id="aee-photo-payload",
                                    elem_classes=["aee-photo-transfer"])
                                account_apply_photo_btn = gr.Button(
                                    "Apply edited profile photo",
                                    elem_id="aee-photo-submit",
                                    elem_classes=["aee-photo-transfer"])
                                account_photo_apply_result = gr.HTML(
                                    value="", elem_id="aee-photo-apply-result",
                                    elem_classes=["aee-photo-editor-result"])
                                with gr.Group(elem_classes=["aee-settings-card"]):
                                    with gr.Row(elem_classes=["aee-settings-card-heading"]):
                                        gr.HTML('''<div><h2>Personal information</h2>
                                          <p>Update the profile details associated with your account.</p></div>''')
                                        account_save_name_btn = gr.Button(
                                            "Save changes", variant="primary",
                                            elem_classes=["aee-settings-save-button"])
                                    gr.HTML('''<div class="aee-settings-dirty"
                                      data-aee-settings-dirty hidden>Unsaved changes · Save to keep your updates.</div>''')
                                    with gr.Row(elem_classes=["aee-settings-fields"]):
                                        account_profile_name = gr.Textbox(
                                            label="Display name",
                                            value=account_profile_settings_initial.get("display_name", ""),
                                            placeholder="Your name", max_lines=1)
                                        account_profile_username = gr.Textbox(
                                            label="Username",
                                            value=("@" + account_profile_settings_initial["username"]
                                                   if account_profile_settings_initial.get("username") else ""),
                                            placeholder="@username", max_lines=1)
                                        account_profile_email = gr.Textbox(
                                            label="Email",
                                            value=account_profile_settings_initial.get("email", ""),
                                            placeholder="you@example.com",
                                            type="email", max_lines=1,
                                            visible=bool(account_session_initial),
                                            elem_id="aee-settings-profile-email")
                                    gr.HTML('''<div class="aee-account-settings-copy">
                                      Guest profile details stay on this computer.
                                      Signed-in profile details are saved to your Supabase account.
                                      Email changes require confirmation from your inbox.</div>''')
                                    with gr.Row(elem_classes=["aee-account-profile-actions"]):
                                        account_refresh_profile_btn = gr.Button(
                                            "Refresh profile", variant="secondary",
                                            visible=False)
                            with gr.Group(
                                elem_id="aee-settings-security-section",
                                elem_classes=["aee-settings-panel", "aee-settings-card"],
                            ):
                                with gr.Row(elem_classes=["aee-settings-card-heading"]):
                                    gr.HTML('''<div><h2>Security &amp; account</h2>
                                      <p>Manage how you sign in and connect your account.</p></div>''')
                                with gr.Column(elem_classes=["aee-settings-option-list"]):
                                    account_signout_btn = gr.Button(
                                        "Log out", variant="secondary",
                                        visible=bool(account_session_initial),
                                        elem_classes=["aee-account-signout"],
                                        elem_id="aee-account-signout")
                                    account_settings_google_btn = gr.Button(
                                        "Sign in with Google", variant="primary",
                                        visible=(account_connection_ready
                                                 and account_guest_mode_initial))
                                    account_change_account_btn = gr.Button(
                                        "Change account", variant="secondary")
                account_oauth_transaction = gr.State(None)
                account_oauth_timer = gr.Timer(value=1, active=False)

        # Project pages live in independent fullscreen app panels, not workspace tabs.
        with gr.Group(visible=True, elem_id="aee-updates-window",
                      elem_classes=["aee-utility-window-shell"]) as updates_window:
            with gr.Column(elem_classes=["wrap"]):
                with gr.Row(elem_classes=["aee-utility-window-header"]):
                    gr.HTML('''<header class="aee-project-page-heading">
                      <span class="aee-project-page-mark" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M20 7v5h-5M4 17v-5h5"/><path d="M6.1 9A7 7 0 0 1 18 6l2 6M4 12l2 6a7 7 0 0 0 11.9-3"/></svg></span>
                      <div><small>APP CENTER</small><h1>Updates</h1>
                        <p>Check for the latest improvements and install updates.</p></div>
                    </header>''')
                    update_close_btn = gr.Button("Back to app", size="sm",
                        elem_classes=["aee-utility-window-close"], elem_id="aee-updates-close")
                gr.HTML(_updates_intro_html())
                update_status = gr.HTML(
                    value=_update_status_html("Select Check for Updates to check for a new release."),
                    elem_id="aee-update-status")
                update_state = gr.State(None)
                with gr.Row(elem_classes=["aee-update-actions"]):
                    update_check_btn = gr.Button(
                        "Check for Updates", variant="primary", size="lg")
                    update_download_btn = gr.Button(
                        "Download Update", variant="secondary", size="lg", visible=False)
                    update_apply_btn = gr.Button(
                        "Install Update", variant="primary", size="lg", visible=False)
                    update_restart_btn = gr.Button(
                        "🔄 Restart App", variant="primary", size="lg",
                        visible=False, elem_classes=["aee-restart-btn"])

        with gr.Group(visible=True, elem_id="aee-about-us-window",
                      elem_classes=["aee-utility-window-shell"]) as about_window:
            with gr.Column(elem_classes=["wrap"]):
                with gr.Row(elem_classes=["aee-utility-window-header"]):
                    gr.HTML('''<header class="aee-project-page-heading">
                      <span class="aee-project-page-mark" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M9.6 9a2.5 2.5 0 1 1 4.3 1.7c-1.1 1-1.9 1.3-1.9 2.8M12 17h.01"/></svg></span>
                      <div><small>THE PROJECT</small><h1>About Us</h1>
                        <p>Learn about the workspace and how it supports evidence review.</p></div>
                    </header>''')
                    about_close_btn = gr.Button("Back to app", size="sm",
                        elem_classes=["aee-utility-window-close"], elem_id="aee-about-close")
                gr.HTML('''<section class="aee-about-card">
                  <div class="aee-about-eyebrow"><span class="aee-about-eyebrow-mark">✦</span>
                    Built for careful review</div>
                  <h2 class="aee-about-title">Evidence deserves a clearer picture.</h2>
                  <div class="aee-about-copy">
                    <p>AI Evidence Extractor brings video analysis and evidence review together in one
                    desktop workspace. Explore object detections, follow tracks, inspect model
                    explanations, and prepare reports without losing sight of the source footage.</p>
                  </div>
                  <div class="aee-project-feature-grid" aria-label="Project capabilities">
                    <div class="aee-project-feature"><span class="aee-project-feature-index">01 / ANALYZE</span>
                      <strong>Video analysis</strong><span>Review footage with an ensemble of detection models.</span></div>
                    <div class="aee-project-feature"><span class="aee-project-feature-index">02 / EXPLORE</span>
                      <strong>Evidence review</strong><span>Explore captures, object tracks, and model explanations.</span></div>
                    <div class="aee-project-feature"><span class="aee-project-feature-index">03 / ORGANIZE</span>
                      <strong>Case context</strong><span>Keep analysis history and case details close at hand.</span></div>
                    <div class="aee-project-feature"><span class="aee-project-feature-index">04 / EXPORT</span>
                      <strong>Report exports</strong><span>Prepare evidence summaries and downloadable reports.</span></div>
                  </div>
                  <div class="aee-about-note"><span class="aee-about-note-mark">✓</span>
                    <div><strong>Human review stays in control</strong>
                      <span>Detections, threat scores, and explanations are decision-support tools.
                      Always verify findings against the original video.</span></div>
                  </div>
                </section>''')

        with gr.Group(visible=True, elem_id="aee-contact-us-window",
                      elem_classes=["aee-utility-window-shell"]) as contact_window:
            with gr.Column(elem_classes=["wrap"]):
                with gr.Row(elem_classes=["aee-utility-window-header"]):
                    gr.HTML('''<header class="aee-project-page-heading">
                      <span class="aee-project-page-mark" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 5h16v14H4z"/><path d="m4 7 8 6 8-6"/></svg></span>
                      <div><small>SUPPORT</small><h1>Contact Us</h1>
                        <p>Get help, share feedback, or report an issue.</p></div>
                    </header>''')
                    contact_close_btn = gr.Button("Back to app", size="sm",
                        elem_classes=["aee-utility-window-close"], elem_id="aee-contact-close")
                gr.HTML('''<section class="aee-contact-card">
                  <h1 class="aee-contact-title">Contact the project team</h1>
                  <div class="aee-contact-copy">Share questions, feature ideas, or bug reports through the
                    project issue tracker. For a useful report, include the app version and steps to reproduce
                    the issue. Do not post private videos, case details, or other sensitive evidence.</div>
                  <a class="aee-contact-link" href="https://github.com/othmanola/AI-Evidence-Extractor/issues"
                    target="_blank" rel="noopener noreferrer">Open GitHub Issues <span aria-hidden="true">â†—</span></a>
                </section>''')

        gr.HTML(f"""<div class="aee-footer">
            <b>AI Evidence Extractor</b> · Threat Scoring + Object Tracking + EigenCAM
            · {info.get('n_models',0)} models · Learned Fusion
            <br><span style="opacity:.6;font-size:.7rem;">
            Shortcuts · <b>A</b> analyze · <b>⌘K</b> palette ·
            <b>T</b> theme · <b>1–8</b> tabs</span></div>""")

        rows_state = gr.State([]); caps_state = gr.State([])
        sums_state = gr.State([]); cam_heat_state = gr.State([])
        cam_blend_state = gr.State([]); track_rows_state = gr.State([])
        case_number = gr.State(""); evidence_id = gr.State("")
        examiner = gr.State(""); agency = gr.State(""); case_notes = gr.State("")

        OUT = [master_dl, pdf_dl, gallery, sum_files, sum_video,
               cap_filter, ev_class, ev_table,
               overview_html, progress_html,
               rows_state, caps_state, sums_state,
               master_video,
               cam_heat_state, cam_blend_state, cam_gallery, track_rows_state,
               track_html, cam_html, track_table]

        CASE_INPUTS = [case_number, evidence_id, examiner, agency, case_notes]

        def _empty_tuple_with_status(status=""):
            clean_df = _det_df([])
            return (None, None, [], [], None,
                gr.update(choices=["All"], value="All"),
                gr.update(choices=["All"], value="All"),
                clean_df, _empty_html(), status,
                [], [], [], None,
                [], [], [], [],
                Tracker().panel_html(), CAMCache().panel_html(), clean_df)

        def handle_upload_change(files):
            cancel_active_analysis()
            if files:
                n = len(files) if isinstance(files, list) else 1
                status = (f'<span data-aee-upload-ready="{time.time_ns()}">'
                          f'📹 {n} file{"s" if n != 1 else ""} ready · '
                          f'press Run analysis</span>')
            else:
                status = "Drop or upload video files to begin"
            tup = _empty_tuple_with_status(status)
            return (_upload_files_html(files), *tup,
                    gr.update(choices=[], value=None), None, None, None)

        video_input.upload(fn=handle_upload_change, inputs=[video_input],
            outputs=[files_meta, *OUT, sum_select, csv_file, json_file, track_csv],
            queue=False)

        def reset_all():
            cancel_active_analysis()
            tup = _empty_tuple_with_status(
                "Results cleared · ready for a new analysis")
            return ("", *tup,
                    gr.update(choices=[], value=None), None, None, None)

        reset_btn.click(fn=reset_all, inputs=[],
            outputs=[files_meta, *OUT, sum_select, csv_file, json_file, track_csv],
            queue=False)

        process_event = process_btn.click(
            fn=process_batch, inputs=[video_input, *CASE_INPUTS, high_accuracy],
            outputs=OUT, show_progress="hidden", api_name="analyze")

        def _post(master, pdf, caps, sums, prev, capc, evc, df, ov, prog,
                  rows, caps_s, sums_s, mv, cam_h, cam_b, cam_disp, tr_rows,
                  tr_html, cam_html_v, tr_df):
            return (master, pdf, caps, sums, sums[0] if sums else None,
                    capc, evc, df, ov, prog, rows, caps, sums, master,
                    cam_h, cam_b, cam_disp, tr_rows, tr_html, cam_html_v, tr_df)
        process_event = process_event.then(
            fn=_post, inputs=OUT, outputs=OUT, show_progress="hidden")

        def _post2(sums):
            return gr.update(choices=[Path(s).stem for s in sums] if sums else [],
                             value=Path(sums[0]).stem if sums else None)
        process_event.then(
            fn=_post2, inputs=[sums_state], outputs=[sum_select],
            show_progress="hidden")
        history_refresh.click(fn=_history_dataframe, inputs=[],
                               outputs=[history_table], queue=False)
        detailed_history_refresh.click(
            fn=_history_refresh_ui, inputs=[history_refresh_state],
            outputs=[history_refresh_state], queue=False)
        all_history_zip.click(
            fn=lambda: _history_zip_ui(),
            inputs=[], outputs=[all_history_zip, history_zip_status])
        process_event.then(
            fn=_history_dataframe, inputs=[], outputs=[history_table],
            show_progress="hidden")
        process_event.then(
            fn=_history_refresh_ui, inputs=[history_refresh_state],
            outputs=[history_refresh_state], show_progress="hidden")
        account_auth_mode_btn.click(
            fn=_account_auth_mode_ui, inputs=[account_auth_mode],
            outputs=[account_auth_mode, account_auth_intro, account_name_group,
                     account_signup_btn, account_signin_btn, account_auth_mode_btn],
            queue=False)
        account_google_btn.click(
            fn=_account_google_start_ui, inputs=[],
            outputs=[account_oauth_transaction, account_status,
                     account_google_btn, account_settings_google_btn,
                     account_oauth_timer],
            queue=False)
        account_settings_google_btn.click(
            fn=_account_google_start_ui, inputs=[],
            outputs=[account_oauth_transaction, account_status,
                     account_google_btn, account_settings_google_btn,
                     account_oauth_timer],
            queue=False)
        account_profile_settings_fields = [
            account_profile_name, account_profile_username, account_profile_email,
        ]
        account_oauth_poll_event = account_oauth_timer.tick(
            fn=_account_google_poll_ui,
            inputs=[account_oauth_transaction, account_session, account_access_state],
            outputs=[account_oauth_transaction, account_status, account_profile,
                     account_session, account_profile_name, account_auth_group,
                     account_profile_group, account_access_marker, account_access_state,
                     account_oauth_timer, account_google_btn, account_profile_settings,
                     account_signout_btn, account_settings_google_btn],
            queue=False)
        account_oauth_poll_event.then(
            fn=_account_profile_settings_refresh_ui, inputs=[account_session],
            outputs=account_profile_settings_fields, queue=False).then(
            fn=_account_profile_email_visibility_ui, inputs=[account_session],
            outputs=[account_profile_email], queue=False)
        account_signin_event = account_signin_btn.click(
            fn=_account_sign_in_ui,
            inputs=[account_email_input, account_password_input, account_session,
                    account_access_state],
            outputs=[account_status, account_profile, account_session, account_profile_name,
                     account_auth_group, account_profile_group,
                     account_access_marker, account_access_state,
                     account_profile_settings, account_signout_btn,
                     account_settings_google_btn],
            queue=False)
        account_signin_event.then(
            fn=_account_profile_settings_values_ui, inputs=[account_session],
            outputs=account_profile_settings_fields, queue=False).then(
            fn=_account_profile_email_visibility_ui, inputs=[account_session],
            outputs=[account_profile_email], queue=False)
        account_signup_event = account_signup_btn.click(
            fn=_account_sign_up_ui,
            inputs=[account_display_name_input, account_email_input,
                    account_password_input, account_session, account_access_state],
            outputs=[account_status, account_profile, account_session, account_profile_name,
                     account_auth_group, account_profile_group,
                     account_access_marker, account_access_state,
                     account_profile_settings, account_signout_btn,
                     account_settings_google_btn],
            queue=False)
        account_signup_event.then(
            fn=_account_profile_settings_values_ui, inputs=[account_session],
            outputs=account_profile_settings_fields, queue=False).then(
            fn=_account_profile_email_visibility_ui, inputs=[account_session],
            outputs=[account_profile_email], queue=False)
        account_guest_event = account_guest_btn.click(
            fn=_account_guest_ui, inputs=[account_session, account_access_state],
            outputs=[account_status, account_profile, account_session, account_profile_name,
                     account_auth_group, account_profile_group,
                     account_access_marker, account_access_state,
                     account_profile_settings, account_signout_btn,
                     account_settings_google_btn],
            queue=False)
        account_guest_event.then(
            fn=_account_profile_settings_values_ui, inputs=[account_session],
            outputs=account_profile_settings_fields, queue=False).then(
            fn=_account_profile_email_visibility_ui, inputs=[account_session],
            outputs=[account_profile_email], queue=False)
        account_signout_event = account_signout_btn.click(
            fn=_account_guest_ui, inputs=[account_session, account_access_state],
            outputs=[account_status, account_profile, account_session, account_profile_name,
                     account_auth_group, account_profile_group,
                     account_access_marker, account_access_state,
                     account_profile_settings, account_signout_btn,
                     account_settings_google_btn],
            queue=False)
        account_signout_event.then(
            fn=_account_profile_settings_values_ui, inputs=[account_session],
            outputs=account_profile_settings_fields, queue=False).then(
            fn=_account_profile_email_visibility_ui, inputs=[account_session],
            outputs=[account_profile_email], queue=False)
        account_change_account_event = account_change_account_btn.click(
            fn=_account_change_account_ui, inputs=[account_session],
            outputs=[account_status, account_profile, account_session, account_profile_name,
                     account_auth_group, account_profile_group,
                     account_access_marker, account_access_state,
                     account_profile_settings, account_signout_btn,
                     account_settings_google_btn],
            queue=False)
        account_change_account_event.then(
            fn=_account_profile_settings_values_ui, inputs=[account_session],
            outputs=account_profile_settings_fields, queue=False).then(
            fn=_account_profile_email_visibility_ui, inputs=[account_session],
            outputs=[account_profile_email], queue=False)
        account_refresh_profile_btn.click(
            fn=_account_refresh_profile_ui, inputs=[account_session],
            outputs=[account_status, account_profile, account_session, account_profile_name],
            queue=False)
        account_save_name_btn.click(
            fn=_account_save_profile_settings_ui,
            inputs=[account_profile_name, account_profile_username,
                    account_profile_email, account_session],
            outputs=[account_status, account_profile, account_session],
            queue=False)
        account_apply_photo_btn.click(
            fn=_account_apply_photo_data_ui,
            inputs=[account_photo_payload, account_session],
            outputs=[account_status, account_profile, account_session,
                     account_photo_apply_result],
            queue=False)
        account_save_config_event = account_save_config_btn.click(
            fn=_account_save_config_ui,
            inputs=[account_project_url, account_anon_key, account_session,
                    account_access_state],
            outputs=[account_status, account_profile, account_session, account_profile_name,
                     account_auth_group, account_profile_group,
                     account_access_marker, account_access_state,
                     account_profile_settings, account_signout_btn,
                     account_settings_google_btn, account_google_btn],
            queue=False)
        account_save_config_event.then(
            fn=_account_profile_settings_values_ui, inputs=[account_session],
            outputs=account_profile_settings_fields, queue=False).then(
            fn=_account_profile_email_visibility_ui, inputs=[account_session],
            outputs=[account_profile_email], queue=False).then(
            fn=_account_connection_setup_visibility_ui, inputs=[],
            outputs=[account_connection_setup], queue=False).then(
            fn=_account_google_sign_in_visibility_ui, inputs=[],
            outputs=[account_google_btn], queue=False)
        update_check_btn.click(fn=_check_updates_ui, inputs=[],
            outputs=[update_status, update_download_btn, update_state,
                     update_apply_btn])
        update_download_btn.click(fn=_download_update_ui, inputs=[update_state],
            outputs=[update_status, update_apply_btn, update_state])
        update_apply_btn.click(fn=_apply_update_ui, inputs=[update_state],
            outputs=[update_status, update_apply_btn, update_restart_btn])
        update_restart_btn.click(fn=_restart_app_ui, inputs=[],
            outputs=[update_status, update_restart_btn], queue=False)
        app.load(fn=_check_updates_ui, inputs=[],
            outputs=[update_status, update_download_btn, update_state,
                     update_apply_btn])
        sum_select.change(fn=pick_summary, inputs=[sum_select, sums_state],
                          outputs=[sum_video])
        cap_filter.change(fn=filter_gallery,
            inputs=[cap_filter, caps_state, rows_state], outputs=[gallery])
        ev_search.change(fn=filter_evidence,
            inputs=[ev_search, ev_class, rows_state], outputs=[ev_table])
        ev_class.change(fn=filter_evidence,
            inputs=[ev_search, ev_class, rows_state], outputs=[ev_table])
        csv_btn.click(fn=export_csv, inputs=[rows_state, case_number], outputs=[csv_file])
        json_btn.click(fn=export_json,
                       inputs=[rows_state, case_number, evidence_id, examiner, agency, case_notes],
                       outputs=[json_file])
        track_csv_btn.click(fn=export_tracks_csv,
                            inputs=[track_rows_state, case_number], outputs=[track_csv])

    return app


if __name__ == "__main__":
    log.info("Starting AI Evidence Extractor in its native desktop window")
    from desktop_launcher import main as launch_desktop
    launch_desktop()
