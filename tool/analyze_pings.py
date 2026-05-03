#!/usr/bin/env python3
"""Summarize one or more PINGS experiment result directories."""

from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_INPUT_DIR = Path("/home/li0/lio_ws/MPINGS/4090")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PINGS Experiment Results Analyzer")
    parser.add_argument(
        "input_path",
        nargs="?",
        default=None,
        help="Single experiment directory or a directory containing multiple experiment directories.",
    )
    parser.add_argument(
        "--dir",
        dest="legacy_dir",
        default=None,
        help="Backward-compatible alias of input_path.",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "single", "batch"),
        default="auto",
        help="How to interpret the input path. Default: auto",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated summaries. Default: current directory for batch, input directory for single.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="summary",
        help="Base filename for generated files, without extension. Default: summary",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Create a zip archive of the generated analysis files.",
    )
    parser.add_argument(
        "--print-markdown",
        action="store_true",
        help="Print a Markdown table to stdout.",
    )
    return parser.parse_args()


def safe_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def display_float(value, digits: int = 3, none_text: str = "N/A") -> str:
    if value is None:
        return none_text
    return f"{value:.{digits}f}"


def safe_int(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def get_dir_size(path: Path) -> float:
    total = 0
    if not path.exists():
        return 0.0
    for file_path in path.glob("**/*"):
        if file_path.is_file():
            total += file_path.stat().st_size
    return total / (1024 * 1024)


def is_experiment_dir(path: Path) -> bool:
    markers = ("pose_eval.csv", "time_table.npy", "log", "meta/config_all.yaml", "run.sh")
    return any((path / marker).exists() for marker in markers)


def resolve_input_path(args: argparse.Namespace) -> Path:
    raw = args.input_path or args.legacy_dir
    if raw is None:
        return DEFAULT_INPUT_DIR
    return Path(raw).expanduser().resolve()


def list_experiment_dirs(input_path: Path, mode: str) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Directory {input_path} does not exist.")

    if mode == "single":
        return [input_path]

    if mode == "batch":
        return sorted(path for path in input_path.iterdir() if path.is_dir() and is_experiment_dir(path))

    if is_experiment_dir(input_path):
        return [input_path]

    return sorted(path for path in input_path.iterdir() if path.is_dir() and is_experiment_dir(path))


def get_log_dirs(exp_dir: Path) -> list[Path]:
    log_root = exp_dir / "log"
    if not log_root.exists():
        return []
    return sorted(path for path in log_root.iterdir() if path.is_dir())


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def parse_pose_eval(exp_dir: Path, results: dict[str, object]) -> None:
    pose_eval = exp_dir / "pose_eval.csv"
    if not pose_eval.exists():
        return
    try:
        with pose_eval.open("r", newline="") as file_obj:
            reader = list(csv.reader(file_obj))
        if len(reader) >= 2 and len(reader[1]) >= 5:
            results["RPE_trans"] = safe_float(reader[1][0])
            results["RPE_rot"] = safe_float(reader[1][1])
            results["ATE"] = safe_float(reader[1][2])
            latency_s = safe_float(reader[1][4])
            if latency_s is not None:
                results["latency_avg_ms"] = latency_s * 1000.0
            results["success"] = "Success"
    except Exception as exc:  # noqa: BLE001
        results["notes"] += f"Error parsing pose_eval: {exc}; "


def parse_time_table(exp_dir: Path, results: dict[str, object]) -> None:
    time_table = exp_dir / "time_table.npy"
    if not time_table.exists():
        return
    try:
        data = np.load(time_table)
        if data.ndim == 2:
            total_latencies = np.sum(data, axis=1) * 1000.0
            if float(np.mean(total_latencies)) > 10000.0:
                total_latencies = np.sum(data, axis=1)
            results["latency_avg_ms"] = float(np.mean(total_latencies))
            results["latency_median_ms"] = float(np.median(total_latencies))
            results["latency_max_ms"] = float(np.max(total_latencies))
            results["latency_p95_ms"] = float(np.percentile(total_latencies, 95))
            results["latency_total_s"] = float(np.sum(total_latencies) / 1000.0)
            results["frame_count"] = int(total_latencies.shape[0])
            if results["latency_avg_ms"] and results["latency_avg_ms"] > 0:
                results["FPS"] = 1000.0 / results["latency_avg_ms"]
            else:
                results["FPS"] = 0.0
    except Exception as exc:  # noqa: BLE001
        results["notes"] += f"Error parsing time_table: {exc}; "


def parse_gpu_and_logs(exp_dir: Path, results: dict[str, object]) -> None:
    ram_candidates: list[Path] = []
    timing_candidates: list[Path] = []
    for log_dir in get_log_dirs(exp_dir):
        ram_candidates.append(log_dir / "ram_log_raw.csv")
        timing_candidates.append(log_dir)
        gpu_summary = log_dir / "charts" / "gpu_summary.csv"
        if gpu_summary.exists():
            try:
                with gpu_summary.open("r", newline="") as file_obj:
                    reader = csv.DictReader(file_obj)
                    for row in reader:
                        if row.get("metric") == "mem_used_mb":
                            results["gpu_vram_avg_mb"] = safe_float(row.get("mean"))
                            results["gpu_vram_peak_mb"] = safe_float(row.get("max"))
            except Exception as exc:  # noqa: BLE001
                results["notes"] += f"Error parsing gpu_summary: {exc}; "

        stdout_log = log_dir / "stdout.log"
        if stdout_log.exists():
            parse_stdout_log(stdout_log, results)
        stage3_monitor = log_dir / "stage3_monitor.csv"
        if stage3_monitor.exists():
            parse_stage3_monitor(stage3_monitor, results)

    ram_candidates.append(exp_dir / "ram_log_raw.csv")
    for ram_log in ram_candidates:
        if not ram_log.exists():
            continue
        try:
            with ram_log.open("r", newline="") as file_obj:
                reader = csv.DictReader(file_obj)
                ram_vals = [safe_float(row.get("mem_used_mb")) for row in reader]
            ram_vals = [value for value in ram_vals if value is not None]
            if ram_vals:
                results["system_ram_peak_mb"] = max(ram_vals)
                break
        except Exception as exc:  # noqa: BLE001
            results["notes"] += f"Error parsing ram_log: {exc}; "

    for timing_dir in timing_candidates:
        start_ms = safe_int(read_text_if_exists(timing_dir / "run_start_ms.txt"))
        end_ms = safe_int(read_text_if_exists(timing_dir / "run_end_ms.txt"))
        if start_ms is not None and end_ms is not None and end_ms >= start_ms:
            results["wall_clock_total_s"] = (end_ms - start_ms) / 1000.0
            break


def read_text_if_exists(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def has_runtime_anomaly(text: str) -> bool:
    anomaly_patterns = (
        r"\btraceback\b",
        r"\bruntimeerror\b",
        r"\bexception\b",
        r"\bsegmentation fault\b",
        r"\bcuda out of memory\b",
        r"\bnan\b",
        r"\binf\b",
        r"\bassertionerror\b",
        r"\bvalueerror\b",
        r"\btypeerror\b",
        r"\bkeyerror\b",
        r"\bindexerror\b",
        r"\battributeerror\b",
        r"\bmodulenotfounderror\b",
    )
    lower_text = text.lower()
    return any(re.search(pattern, lower_text) for pattern in anomaly_patterns)


def parse_stdout_log(stdout_log: Path, results: dict[str, object]) -> None:
    text = stdout_log.read_text(errors="replace")

    init_matches = re.findall(r"# Global neural point:\s*(\d+)", text)
    if init_matches:
        init_vals = [int(value) for value in init_matches]
        results["point_count_init"] = init_vals[0]
        results["point_count_peak"] = max(init_vals)

    prune_match = re.search(r"# Prune neural points:\s*(\d+)", text)
    if prune_match:
        results["point_pruned"] = int(prune_match.group(1))

    final_match = re.search(r"Final neural point count:\s*(\d+)", text)
    if final_match:
        results["point_count_final"] = int(final_match.group(1))

    psnr_match = re.search(r"Average .*? PSNR.*?:\s*([0-9.]+)", text)
    if psnr_match:
        results["psnr"] = safe_float(psnr_match.group(1))

    ssim_match = re.search(r"Average .*? SSIM.*?:\s*([0-9.]+)", text)
    if ssim_match:
        results["ssim"] = safe_float(ssim_match.group(1))

    lpips_match = re.search(r"Average .*? LPIPS.*?:\s*([0-9.]+)", text)
    if lpips_match:
        results["lpips"] = safe_float(lpips_match.group(1))

    if has_runtime_anomaly(text):
        results["success"] = "Warning/Fail"
        results["notes"] += "Crashes or NaNs detected in log; "


def parse_stage3_monitor(stage3_monitor: Path, results: dict[str, object]) -> None:
    try:
        with stage3_monitor.open("r", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            rows = list(reader)
    except Exception as exc:  # noqa: BLE001
        results["notes"] += f"Error parsing stage3_monitor: {exc}; "
        return

    if not rows:
        return

    metric_names = [
        "affine/scale_mean_abs_dev",
        "affine/scale_max_abs_dev",
        "affine/quat_norm_mean_abs_dev",
        "affine/quat_norm_max_abs_dev",
    ]
    for metric_name in metric_names:
        values = [safe_float(row.get(metric_name)) for row in rows]
        values = [value for value in values if value is not None]
        if not values:
            continue
        key = metric_name.replace("/", "_").replace(".", "_")
        results[f"{key}_last"] = values[-1]
        results[f"{key}_peak"] = max(values)


def parse_map_compression(exp_dir: Path, results: dict[str, object]) -> None:
    meta = load_json(exp_dir / "meta" / "map_compression.json")
    if not isinstance(meta, dict):
        return

    results["map_compression_written"] = True
    results["map_original_point_count"] = safe_int(meta.get("original_point_count"))
    results["map_simplified_point_count"] = safe_int(meta.get("simplified_point_count"))
    results["map_flat_point_count"] = safe_int(meta.get("flat_point_count"))
    results["map_detail_point_count"] = safe_int(meta.get("detail_point_count"))
    results["map_simplified_flat_region_count"] = safe_int(meta.get("simplified_flat_region_count"))
    results["map_refined_region_count"] = safe_int(meta.get("refined_region_count"))
    results["map_region_to_source_written_point_count"] = safe_int(
        meta.get("region_to_source_written_point_count")
    )
    results["map_initial_consistency_l1"] = safe_float(meta.get("initial_consistency_l1"))
    results["map_final_consistency_l1"] = safe_float(meta.get("final_consistency_l1"))
    results["map_mean_abs_shift"] = safe_float(meta.get("mean_abs_shift"))
    results["map_max_abs_shift"] = safe_float(meta.get("max_abs_shift"))
    results["map_affine_scale_mean_abs_dev"] = safe_float(meta.get("affine_scale_mean_abs_dev"))
    results["map_affine_scale_max_abs_dev"] = safe_float(meta.get("affine_scale_max_abs_dev"))
    results["map_affine_quat_mean_abs_dev"] = safe_float(meta.get("affine_quat_mean_abs_dev"))
    results["map_affine_quat_max_abs_dev"] = safe_float(meta.get("affine_quat_max_abs_dev"))
    consistency_mode = meta.get("consistency_finetune")
    if consistency_mode is not None:
        results["map_consistency_finetune"] = str(consistency_mode)


def parse_mesh_and_model(exp_dir: Path, results: dict[str, object]) -> None:
    mesh_dir = exp_dir / "mesh"
    model_dir = exp_dir / "model"
    mesh_files = sorted(mesh_dir.glob("*.ply")) if mesh_dir.exists() else []
    results["mesh_size_mb"] = get_dir_size(mesh_dir)
    results["model_size_mb"] = get_dir_size(model_dir)
    results["mesh_exported"] = bool(mesh_files)
    results["mesh_file_count"] = len(mesh_files)
    if mesh_files:
        results["mesh_primary_file"] = mesh_files[0].name


def parse_loop_diagnostics(exp_dir: Path, results: dict[str, object]) -> None:
    loop_log = exp_dir / "loop_log.txt"
    pose_graph = exp_dir / "final_pose_graph.g2o"
    traj2d = exp_dir / "traj_plot_2d.png"
    traj3d = exp_dir / "traj_plot_3d.png"

    results["loop_log_written"] = loop_log.exists()
    results["pose_graph_written"] = pose_graph.exists()
    results["traj_plot_2d_written"] = traj2d.exists()
    results["traj_plot_3d_written"] = traj3d.exists()

    if not loop_log.exists():
        return

    first_loop_line = ""
    try:
        for line in loop_log.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if stripped:
                first_loop_line = stripped
                break
    except OSError as exc:
        results["notes"] += f"Error parsing loop_log: {exc}; "
        return

    if not first_loop_line:
        return

    results["first_loop_entry"] = first_loop_line
    match = re.match(r"^\s*(\d+)\s+(\d+)", first_loop_line)
    if match:
        results["first_loop_from_frame"] = int(match.group(1))
        results["first_loop_to_frame"] = int(match.group(2))


def analyze_experiment(exp_dir: Path) -> dict[str, object]:
    results: dict[str, object] = {
        "exp_name": exp_dir.name,
        "exp_path": str(exp_dir),
        "ATE": None,
        "RPE_trans": None,
        "RPE_rot": None,
        "latency_avg_ms": None,
        "latency_median_ms": None,
        "latency_max_ms": None,
        "latency_p95_ms": None,
        "latency_total_s": None,
        "wall_clock_total_s": None,
        "frame_count": None,
        "FPS": None,
        "gpu_vram_avg_mb": None,
        "gpu_vram_peak_mb": None,
        "system_ram_peak_mb": None,
        "point_count_init": None,
        "point_count_peak": None,
        "point_count_final": None,
        "point_pruned": None,
        "mesh_size_mb": None,
        "mesh_exported": False,
        "mesh_file_count": 0,
        "mesh_primary_file": "",
        "loop_log_written": False,
        "pose_graph_written": False,
        "traj_plot_2d_written": False,
        "traj_plot_3d_written": False,
        "first_loop_entry": "",
        "first_loop_from_frame": None,
        "first_loop_to_frame": None,
        "model_size_mb": None,
        "map_compression_written": False,
        "map_original_point_count": None,
        "map_simplified_point_count": None,
        "map_flat_point_count": None,
        "map_detail_point_count": None,
        "map_simplified_flat_region_count": None,
        "map_refined_region_count": None,
        "map_region_to_source_written_point_count": None,
        "map_consistency_finetune": "",
        "map_initial_consistency_l1": None,
        "map_final_consistency_l1": None,
        "map_mean_abs_shift": None,
        "map_max_abs_shift": None,
        "map_affine_scale_mean_abs_dev": None,
        "map_affine_scale_max_abs_dev": None,
        "map_affine_quat_mean_abs_dev": None,
        "map_affine_quat_max_abs_dev": None,
        "affine_scale_mean_abs_dev_last": None,
        "affine_scale_mean_abs_dev_peak": None,
        "affine_scale_max_abs_dev_last": None,
        "affine_scale_max_abs_dev_peak": None,
        "affine_quat_norm_mean_abs_dev_last": None,
        "affine_quat_norm_mean_abs_dev_peak": None,
        "affine_quat_norm_max_abs_dev_last": None,
        "affine_quat_norm_max_abs_dev_peak": None,
        "psnr": None,
        "ssim": None,
        "lpips": None,
        "success": "Unknown",
        "notes": "",
    }

    parse_pose_eval(exp_dir, results)
    parse_time_table(exp_dir, results)
    parse_gpu_and_logs(exp_dir, results)
    parse_map_compression(exp_dir, results)
    parse_mesh_and_model(exp_dir, results)
    parse_loop_diagnostics(exp_dir, results)
    return results


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_markdown(rows: Iterable[dict[str, object]]) -> str:
    lines = [
        "| Experiment | ATE (m) | Avg / Med / p95 (ms) | GPU Peak (MB) | RAM Peak (MB) | Final / Peak Points | Mesh | Status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {exp_name} | {ate} | {latency_triplet} | {gpu} | {ram} | {points} | {mesh} | {status} |".format(
                exp_name=row["exp_name"],
                ate=display_float(row["ATE"], digits=3),
                latency_triplet="{avg} / {med} / {p95}".format(
                    avg=display_float(row["latency_avg_ms"], digits=1),
                    med=display_float(row["latency_median_ms"], digits=1),
                    p95=display_float(row["latency_p95_ms"], digits=1),
                ),
                gpu=display_float(row["gpu_vram_peak_mb"], digits=0),
                ram=display_float(row["system_ram_peak_mb"], digits=0),
                points="{final} / {peak}".format(
                    final=row["point_count_final"] if row["point_count_final"] is not None else "N/A",
                    peak=row["point_count_peak"] if row["point_count_peak"] is not None else "N/A",
                ),
                mesh="yes" if row["mesh_exported"] else "no",
                status=row["success"],
            )
        )
    return "\n".join(lines) + "\n"


def write_json(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(rows, indent=2) + "\n")


def archive_outputs(output_dir: Path, output_paths: list[Path], archive_name: str) -> Path:
    archive_path = output_dir / f"{archive_name}.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_obj:
        for file_path in output_paths:
            zip_obj.write(file_path, arcname=file_path.name)
    return archive_path


def main() -> None:
    args = parse_args()
    input_path = resolve_input_path(args)
    exp_dirs = list_experiment_dirs(input_path, args.mode)

    if not exp_dirs:
        print(f"No experiment directories found under {input_path}")
        return

    is_single = len(exp_dirs) == 1 and (args.mode != "batch")
    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
    elif is_single:
        output_dir = exp_dirs[0]
    else:
        output_dir = input_path
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(exp_dirs)} experiment director{'y' if len(exp_dirs) == 1 else 'ies'}.")

    all_results = []
    for exp_dir in exp_dirs:
        print(f"Analyzing {exp_dir.name}...")
        all_results.append(analyze_experiment(exp_dir))

    csv_path = output_dir / f"{args.output_name}.csv"
    md_path = output_dir / f"{args.output_name}.md"
    json_path = output_dir / f"{args.output_name}.json"

    write_csv(csv_path, all_results)
    markdown = build_markdown(all_results)
    md_path.write_text(markdown)
    write_json(json_path, all_results)

    print(f"Summary CSV saved to {csv_path}")
    print(f"Summary Markdown saved to {md_path}")
    print(f"Summary JSON saved to {json_path}")

    print("\n### Experiment Summary Table ###")
    print(markdown.rstrip())

    if args.print_markdown:
        print("\n### Markdown Copy ###")
        print(markdown.rstrip())

    missing = []
    if all(row["psnr"] is None for row in all_results):
        missing.append("PSNR/SSIM/LPIPS (Rendering Quality)")
    if all((row["mesh_size_mb"] or 0) == 0 for row in all_results):
        missing.append("Mesh/Model size (Geometry)")

    if missing:
        print("\nNote: The following metrics were missing from ALL runs:")
        for metric in missing:
            print(f"- {metric}")
        print("\nPossible reasons:")
        print("- Rendering quality metrics only appear if evaluation is enabled (check VIS_MODE/GS_MODE).")
        print("- Mesh/Model folders are empty because SAVE_MESH/SAVE_MAP were set to 'off' in run_meta.env.")

    if args.archive:
        archive_path = archive_outputs(output_dir, [csv_path, md_path, json_path], args.output_name)
        print(f"Analysis archive saved to {archive_path}")


if __name__ == "__main__":
    main()
