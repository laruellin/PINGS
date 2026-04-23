#!/usr/bin/env python3
"""Build readable charts for one or more MPINGS experiment log directories.

Examples:
    python3 make_exp_log_charts.py 2080_gs_on_smoke-10
    python3 make_exp_log_charts.py --input-dir /path/to/MPINGS/exp_logs/run-name
    python3 make_exp_log_charts.py --all /path/to/MPINGS/exp_logs

The script intentionally uses only Python's standard library. It writes SVG
charts, CSV summaries, and a small HTML report into <log_dir>/charts.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_LOG_DIR = ROOT / "2080_gs_on_smoke-10"

GPU_COLUMNS = [
    "timestamp",
    "gpu_id",
    "gpu_name",
    "gpu_util_pct",
    "mem_util_pct",
    "mem_used_mb",
    "mem_total_mb",
    "power_w",
]

COLOR = {
    "blue": "#2563eb",
    "green": "#059669",
    "orange": "#d97706",
    "red": "#dc2626",
    "purple": "#7c3aed",
    "slate": "#475569",
    "cyan": "#0891b2",
}


@dataclass
class Series:
    name: str
    values: list[tuple[float, float]]
    color: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MPINGS exp_logs into readable SVG/HTML charts."
    )
    parser.add_argument(
        "log_dir",
        nargs="?",
        type=Path,
        default=None,
        help="Experiment log directory. Relative paths are resolved under this exp_logs folder.",
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        default=None,
        help="Experiment log directory. Same as the positional log_dir argument.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for one input. Default: <log_dir>/charts",
    )
    parser.add_argument(
        "--all",
        nargs="?",
        const=ROOT,
        type=Path,
        default=None,
        metavar="EXP_LOGS_DIR",
        help="Process every direct child experiment directory under EXP_LOGS_DIR. Default: this exp_logs folder.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=9,
        help="Moving-average window for noisy loss/rate charts. Default: 9",
    )
    return parser.parse_args()


def resolve_log_dir(path: Path | None) -> Path:
    if path is None:
        return DEFAULT_LOG_DIR.resolve()
    if path.is_absolute():
        return path.resolve()
    candidate = (ROOT / path).resolve()
    if candidate.exists():
        return candidate
    return path.resolve()


def has_log_inputs(path: Path) -> bool:
    return any((path / name).exists() for name in ("gpu_log_raw.csv", "stdout.log", "stderr.log", "run_meta.env"))


def list_experiment_dirs(exp_logs_dir: Path) -> list[Path]:
    exp_logs_dir = resolve_log_dir(exp_logs_dir)
    if has_log_inputs(exp_logs_dir):
        return [exp_logs_dir]
    if not exp_logs_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {exp_logs_dir}")
    return sorted(path for path in exp_logs_dir.iterdir() if path.is_dir() and has_log_inputs(path))


def parse_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    env: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def read_int_file(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def parse_gpu_log(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not path.exists():
        return rows

    with path.open(newline="", errors="replace") as f:
        reader = csv.reader(f)
        for raw in reader:
            if len(raw) < 8:
                continue
            raw = [item.strip() for item in raw[:8]]
            try:
                ts = datetime.strptime(raw[0], "%Y/%m/%d %H:%M:%S.%f")
                row = {
                    "timestamp": ts,
                    "gpu_id": int(raw[1]),
                    "gpu_name": raw[2],
                    "gpu_util_pct": float(raw[3]),
                    "mem_util_pct": float(raw[4]),
                    "mem_used_mb": float(raw[5]),
                    "mem_total_mb": float(raw[6]),
                    "power_w": float(raw[7]),
                }
            except ValueError:
                continue
            rows.append(row)

    if rows:
        first = rows[0]["timestamp"]
        assert isinstance(first, datetime)
        for row in rows:
            ts = row["timestamp"]
            assert isinstance(ts, datetime)
            row["t_sec"] = (ts - first).total_seconds()
            row["mem_used_pct"] = (
                float(row["mem_used_mb"]) / float(row["mem_total_mb"]) * 100.0
            )
    return rows


def parse_stdout(path: Path) -> tuple[dict[str, list[float]], dict[str, str], dict[str, list[float]]]:
    metrics: dict[str, list[float]] = {}
    final: dict[str, str] = {}
    paired: dict[str, list[float]] = {}
    if not path.exists():
        return metrics, final, paired

    simple_patterns = [
        ("metric3d_ms", re.compile(r"Metric3D prediction time\s+\(ms\):\s*([0-9.eE+-]+)")),
        ("mono_depth_rmse_m", re.compile(r"mono depth rmse \(m\):\s*([0-9.eE+-]+)")),
        ("depth_fitting_rmse_m", re.compile(r"depth fitting rmse \(m\):\s*([0-9.eE+-]+)")),
        ("sky_loss", re.compile(r"Sky loss:\s*([0-9.eE+-]+)")),
        ("depth_rendering_loss_m", re.compile(r"Depth rendering loss \(m\):\s*([0-9.eE+-]+)")),
        ("normal_depth_consistency_loss", re.compile(r"Normal-depth consistency loss:\s*([0-9.eE+-]+)")),
        ("sdf_cons_loss", re.compile(r"SDF cons loss:\s*([0-9.eE+-]+)")),
        ("sdf_normal_cons_loss", re.compile(r"SDF normal cons loss:\s*([0-9.eE+-]+)")),
        ("sdf_bce_loss", re.compile(r"SDF BCE loss:\s*([0-9.eE+-]+)")),
        ("sdf_eikonal_loss", re.compile(r"SDF Eikonal loss:\s*([0-9.eE+-]+)")),
        ("map_memory_mb", re.compile(r"Current map memory consumption:\s*([0-9.eE+-]+)")),
        ("total_sample_pool", re.compile(r"# Total sample in pool:\s*([0-9.eE+-]+)")),
        ("current_sample", re.compile(r"# Current sample\s*:\s*([0-9.eE+-]+)")),
        ("local_sample", re.compile(r"# Local sample\s*:\s*([0-9.eE+-]+)")),
    ]
    time_pattern = re.compile(r"time for (.+?)\s+\((ms|s)\):\s*([0-9.eE+-]+)")
    final_patterns = [
        ("consuming_time_per_frame_s", re.compile(r"Consuming time per frame\s+\(s\):\s*([0-9.eE+-]+)")),
        ("calculated_frames", re.compile(r"Calculated over\s+([0-9]+)\s+frames")),
        ("avg_translation_error_pct", re.compile(r"Average Translation Error\s+\(%\):\s*([0-9.eE+-]+)")),
        ("avg_rotational_error_deg_100m", re.compile(r"Average Rotational Error \(deg/100m\):\s*([0-9.eE+-]+)")),
        ("absolute_trajectory_error_m", re.compile(r"Absoulte Trajectory Error\s+\(m\):\s*([0-9.eE+-]+)")),
        ("final_neural_point_count", re.compile(r"Final neural point count:\s*([0-9]+)")),
    ]

    for line in path.read_text(errors="replace").splitlines():
        for key, pattern in simple_patterns:
            match = pattern.search(line)
            if match:
                metrics.setdefault(key, []).append(float(match.group(1)))

        match = re.search(r"# Global neural point:\s*([0-9]+).*?([0-9]+) valid", line)
        if match:
            metrics.setdefault("global_neural_points", []).append(float(match.group(1)))
            metrics.setdefault("global_valid_neural_points", []).append(float(match.group(2)))

        match = re.search(r"# Local\s+neural point:\s*([0-9]+).*?([0-9]+) valid", line)
        if match:
            metrics.setdefault("local_neural_points", []).append(float(match.group(1)))
            metrics.setdefault("local_valid_neural_points", []).append(float(match.group(2)))

        match = re.search(r"SDF Valid gaussian count:\s*([0-9]+)\s+from\s+([0-9]+)", line)
        if match:
            metrics.setdefault("sdf_valid_gaussians", []).append(float(match.group(1)))
            metrics.setdefault("sdf_total_gaussians", []).append(float(match.group(2)))
            denom = float(match.group(2))
            if denom:
                metrics.setdefault("sdf_valid_gaussian_pct", []).append(float(match.group(1)) / denom * 100.0)

        match = time_pattern.search(line)
        if match:
            label = "time_" + slug(match.group(1)) + "_" + match.group(2)
            final[label] = match.group(3)

        for key, pattern in final_patterns:
            match = pattern.search(line)
            if match:
                final[key] = match.group(1)

    for key in (
        "global_neural_points",
        "local_neural_points",
        "global_valid_neural_points",
        "local_valid_neural_points",
        "map_memory_mb",
    ):
        if key in metrics:
            paired[key] = metrics[key]
    return metrics, final, paired


def parse_stderr_rates(path: Path) -> dict[str, list[float]]:
    rates: dict[str, list[float]] = {}
    if not path.exists():
        return rates
    pattern = re.compile(r"([A-Za-z0-9_+ -]+):\s+\d+%\|.*?,\s*([0-9.]+)it/s")
    for line in path.read_text(errors="replace").splitlines():
        for stage, value in pattern.findall(line):
            rates.setdefault(stage.strip(), []).append(float(value))
    return rates


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def strip_prefix_suffix(text: str, prefix: str = "", suffix: str = "") -> str:
    if prefix and text.startswith(prefix):
        text = text[len(prefix) :]
    if suffix and text.endswith(suffix):
        text = text[: -len(suffix)]
    return text


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) < window:
        return values
    out: list[float] = []
    half = window // 2
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out.append(mean(values[lo:hi]))
    return out


def downsample(points: list[tuple[float, float]], limit: int = 900) -> list[tuple[float, float]]:
    if len(points) <= limit:
        return points
    step = math.ceil(len(points) / limit)
    return points[::step]


def nice_range(values: Iterable[float], pad_ratio: float = 0.08) -> tuple[float, float]:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return 0.0, 1.0
    lo, hi = min(vals), max(vals)
    if lo == hi:
        delta = abs(lo) * 0.1 or 1.0
        return lo - delta, hi + delta
    pad = (hi - lo) * pad_ratio
    return lo - pad, hi + pad


def polyline(points: list[tuple[float, float]], xmap, ymap, color: str) -> str:
    if not points:
        return ""
    coords = " ".join(f"{xmap(x):.1f},{ymap(y):.1f}" for x, y in downsample(points))
    return f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.2" />'


def line_chart(
    title: str,
    x_label: str,
    y_label: str,
    series: list[Series],
    y_min: float | None = None,
    y_max: float | None = None,
    width: int = 1060,
    height: int = 430,
) -> str:
    margin = {"left": 72, "right": 26, "top": 54, "bottom": 58}
    xs = [x for item in series for x, _ in item.values]
    ys = [y for item in series for _, y in item.values]
    if not xs or not ys:
        return empty_svg(title, "No data available", width, height)
    x_min, x_max = nice_range(xs, 0.0)
    yr = nice_range(ys)
    y_min = yr[0] if y_min is None else y_min
    y_max = yr[1] if y_max is None else y_max
    if x_min == x_max:
        x_max = x_min + 1
    if y_min == y_max:
        y_max = y_min + 1

    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]

    def xmap(x: float) -> float:
        return margin["left"] + (x - x_min) / (x_max - x_min) * plot_w

    def ymap(y: float) -> float:
        return margin["top"] + (y_max - y) / (y_max - y_min) * plot_h

    grid = []
    for i in range(6):
        y = margin["top"] + i / 5 * plot_h
        value = y_max - i / 5 * (y_max - y_min)
        grid.append(
            f'<line x1="{margin["left"]}" y1="{y:.1f}" x2="{width - margin["right"]}" y2="{y:.1f}" stroke="#e2e8f0" />'
            f'<text x="{margin["left"] - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="12" fill="#64748b">{value:.2g}</text>'
        )
    for i in range(6):
        x = margin["left"] + i / 5 * plot_w
        value = x_min + i / 5 * (x_max - x_min)
        grid.append(
            f'<line x1="{x:.1f}" y1="{margin["top"]}" x2="{x:.1f}" y2="{height - margin["bottom"]}" stroke="#f1f5f9" />'
            f'<text x="{x:.1f}" y="{height - margin["bottom"] + 22}" text-anchor="middle" font-size="12" fill="#64748b">{value:.0f}</text>'
        )

    lines = [polyline(item.values, xmap, ymap, item.color) for item in series]
    legend_x = margin["left"]
    legend = []
    for i, item in enumerate(series):
        x = legend_x + i * 190
        legend.append(
            f'<rect x="{x}" y="24" width="12" height="12" fill="{item.color}" rx="2" />'
            f'<text x="{x + 18}" y="34" font-size="13" fill="#334155">{html.escape(item.name)}</text>'
        )

    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
            '<rect width="100%" height="100%" fill="#ffffff" />',
            f'<text x="{margin["left"]}" y="18" font-size="18" font-weight="700" fill="#0f172a">{html.escape(title)}</text>',
            *legend,
            *grid,
            f'<line x1="{margin["left"]}" y1="{height - margin["bottom"]}" x2="{width - margin["right"]}" y2="{height - margin["bottom"]}" stroke="#94a3b8" />',
            f'<line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{height - margin["bottom"]}" stroke="#94a3b8" />',
            *lines,
            f'<text x="{width / 2:.1f}" y="{height - 12}" text-anchor="middle" font-size="13" fill="#475569">{html.escape(x_label)}</text>',
            f'<text x="18" y="{height / 2:.1f}" transform="rotate(-90 18 {height / 2:.1f})" text-anchor="middle" font-size="13" fill="#475569">{html.escape(y_label)}</text>',
            "</svg>",
        ]
    )


def bar_chart(title: str, labels: list[str], values: list[float], unit: str = "", width: int = 1060, height: int = 430) -> str:
    if not labels or not values:
        return empty_svg(title, "No data available", width, height)
    margin = {"left": 220, "right": 38, "top": 50, "bottom": 42}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    max_value = max(values) or 1.0
    row_h = plot_h / len(values)
    bars = []
    for i, (label, value) in enumerate(zip(labels, values)):
        y = margin["top"] + i * row_h + row_h * 0.18
        h = row_h * 0.64
        w = value / max_value * plot_w
        bars.append(
            f'<text x="{margin["left"] - 10}" y="{y + h * 0.65:.1f}" text-anchor="end" font-size="12" fill="#334155">{html.escape(label)}</text>'
            f'<rect x="{margin["left"]}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{COLOR["blue"]}" rx="3" />'
            f'<text x="{margin["left"] + w + 7:.1f}" y="{y + h * 0.65:.1f}" font-size="12" fill="#475569">{value:.2f}{html.escape(unit)}</text>'
        )
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
            '<rect width="100%" height="100%" fill="#ffffff" />',
            f'<text x="{margin["left"]}" y="24" font-size="18" font-weight="700" fill="#0f172a">{html.escape(title)}</text>',
            *bars,
            "</svg>",
        ]
    )


def empty_svg(title: str, message: str, width: int, height: int) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#ffffff" />',
            f'<text x="32" y="40" font-size="18" font-weight="700" fill="#0f172a">{html.escape(title)}</text>',
            f'<text x="32" y="78" font-size="14" fill="#64748b">{html.escape(message)}</text>',
            "</svg>",
        ]
    )


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize_gpu(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return []
    numeric = ["gpu_util_pct", "mem_util_pct", "mem_used_mb", "mem_used_pct", "power_w"]
    summary = []
    for key in numeric:
        vals = [float(row[key]) for row in rows]
        summary.append(
            {
                "metric": key,
                "count": len(vals),
                "min": f"{min(vals):.4f}",
                "mean": f"{mean(vals):.4f}",
                "max": f"{max(vals):.4f}",
            }
        )
    return summary


def metric_summary(metrics: dict[str, list[float]]) -> list[dict[str, object]]:
    rows = []
    for key, vals in sorted(metrics.items()):
        if not vals:
            continue
        rows.append(
            {
                "metric": key,
                "count": len(vals),
                "first": f"{vals[0]:.6g}",
                "last": f"{vals[-1]:.6g}",
                "min": f"{min(vals):.6g}",
                "mean": f"{mean(vals):.6g}",
                "max": f"{max(vals):.6g}",
            }
        )
    return rows


def write_text_summary(
    path: Path,
    log_dir: Path,
    env: dict[str, str],
    gpu_rows: list[dict[str, object]],
    final: dict[str, str],
) -> None:
    lines = [f"log_dir={log_dir}"]
    for key in sorted(env):
        lines.append(f"{key}={env[key]}")
    if gpu_rows:
        duration = float(gpu_rows[-1]["t_sec"]) - float(gpu_rows[0]["t_sec"])
        lines.append(f"gpu_samples={len(gpu_rows)}")
        lines.append(f"gpu_log_duration_sec={duration:.3f}")
        lines.append(f"gpu_name={gpu_rows[0]['gpu_name']}")
    for key in sorted(final):
        lines.append(f"{key}={final[key]}")
    path.write_text("\n".join(lines) + "\n")


def save_chart(path: Path, svg: str) -> None:
    path.write_text(svg)


def build_report(log_dir: Path, out_dir: Path | None = None, smooth: int = 9) -> Path:
    log_dir = resolve_log_dir(log_dir)
    if not has_log_inputs(log_dir):
        raise FileNotFoundError(
            f"No supported MPINGS log files found in: {log_dir}\n"
            "Expected one or more of: gpu_log_raw.csv, stdout.log, stderr.log, run_meta.env"
        )
    out_dir = (out_dir or log_dir / "charts").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu_rows = parse_gpu_log(log_dir / "gpu_log_raw.csv")
    env = parse_env(log_dir / "run_meta.env")
    final_from_stdout: dict[str, str]
    metrics, final_from_stdout, paired = parse_stdout(log_dir / "stdout.log")
    rates = parse_stderr_rates(log_dir / "stderr.log")

    start_ms = read_int_file(log_dir / "run_start_ms.txt")
    end_ms = read_int_file(log_dir / "run_end_ms.txt")
    if start_ms is not None and end_ms is not None:
        final_from_stdout["wall_time_from_ms_files_s"] = f"{(end_ms - start_ms) / 1000.0:.3f}"

    if gpu_rows:
        write_csv(
            out_dir / "gpu_timeseries.csv",
            [
                {
                    key: (
                        row[key].isoformat(sep=" ")
                        if key == "timestamp" and isinstance(row[key], datetime)
                        else row.get(key, "")
                    )
                    for key in GPU_COLUMNS + ["t_sec", "mem_used_pct"]
                }
                for row in gpu_rows
            ],
            GPU_COLUMNS + ["t_sec", "mem_used_pct"],
        )
        write_csv(out_dir / "gpu_summary.csv", summarize_gpu(gpu_rows), ["metric", "count", "min", "mean", "max"])

        x = [float(row["t_sec"]) / 60.0 for row in gpu_rows]
        save_chart(
            out_dir / "gpu_util_power.svg",
            line_chart(
                "GPU Utilization and Power",
                "minutes from first GPU sample",
                "percent / watts",
                [
                    Series("GPU util %", list(zip(x, [float(row["gpu_util_pct"]) for row in gpu_rows])), COLOR["blue"]),
                    Series("Memory util %", list(zip(x, [float(row["mem_util_pct"]) for row in gpu_rows])), COLOR["green"]),
                    Series("Power W", list(zip(x, [float(row["power_w"]) for row in gpu_rows])), COLOR["orange"]),
                ],
                y_min=0,
            ),
        )
        save_chart(
            out_dir / "gpu_memory.svg",
            line_chart(
                "GPU Memory Usage",
                "minutes from first GPU sample",
                "MB / percent",
                [
                    Series("Memory used MB", list(zip(x, [float(row["mem_used_mb"]) for row in gpu_rows])), COLOR["purple"]),
                    Series("Memory used %", list(zip(x, [float(row["mem_used_pct"]) for row in gpu_rows])), COLOR["cyan"]),
                ],
                y_min=0,
            ),
        )

    write_csv(
        out_dir / "stdout_metric_summary.csv",
        metric_summary(metrics),
        ["metric", "count", "first", "last", "min", "mean", "max"],
    )
    write_text_summary(out_dir / "summary.txt", log_dir, env, gpu_rows, final_from_stdout)

    loss_keys = [
        ("sky_loss", "Sky", COLOR["blue"]),
        ("depth_rendering_loss_m", "Depth render", COLOR["green"]),
        ("normal_depth_consistency_loss", "Normal-depth", COLOR["orange"]),
        ("sdf_cons_loss", "SDF cons", COLOR["red"]),
        ("sdf_normal_cons_loss", "SDF normal", COLOR["purple"]),
        ("sdf_bce_loss", "SDF BCE", COLOR["slate"]),
        ("sdf_eikonal_loss", "SDF eikonal", COLOR["cyan"]),
    ]
    loss_series = []
    for key, label, color in loss_keys:
        vals = moving_average(metrics.get(key, []), smooth)
        if vals:
            loss_series.append(Series(label, list(enumerate(vals, start=1)), color))
    save_chart(
        out_dir / "loss_curves.svg",
        line_chart("Training Loss Curves", "logged loss occurrence", "loss", loss_series, y_min=0),
    )

    map_series = []
    for key, label, color in [
        ("global_valid_neural_points", "Global valid points", COLOR["blue"]),
        ("local_valid_neural_points", "Local valid points", COLOR["green"]),
        ("map_memory_mb", "Map memory MB", COLOR["orange"]),
    ]:
        vals = paired.get(key, [])
        if vals:
            map_series.append(Series(label, list(enumerate(vals, start=1)), color))
    save_chart(
        out_dir / "map_growth.svg",
        line_chart("Map Growth", "logged map-stat occurrence", "points / MB", map_series, y_min=0),
    )

    if metrics.get("sdf_valid_gaussian_pct"):
        vals = moving_average(metrics["sdf_valid_gaussian_pct"], smooth)
        save_chart(
            out_dir / "sdf_valid_gaussian_pct.svg",
            line_chart(
                "SDF Valid Gaussian Ratio",
                "logged SDF occurrence",
                "valid gaussian %",
                [Series("valid %", list(enumerate(vals, start=1)), COLOR["blue"])],
                y_min=0,
                y_max=105,
            ),
        )

    time_items = []
    for key, value in final_from_stdout.items():
        if key.startswith("time_") and key.endswith("_ms"):
            label = strip_prefix_suffix(key, "time_", "_ms").replace("_", " ")
            time_items.append((label, float(value)))
    if time_items:
        labels, values = zip(*sorted(time_items, key=lambda item: item[1], reverse=True))
        save_chart(out_dir / "frame_timing_ms.svg", bar_chart("Frame Timing Breakdown", list(labels), list(values), " ms"))

    rate_series = []
    for stage, color in zip(sorted(rates), [COLOR["blue"], COLOR["green"], COLOR["orange"], COLOR["red"], COLOR["purple"], COLOR["cyan"]]):
        vals = moving_average(rates[stage], smooth)
        if vals:
            rate_series.append(Series(stage, list(enumerate(vals, start=1)), color))
    if rate_series:
        save_chart(out_dir / "stderr_progress_rates.svg", line_chart("Progress Rates from stderr", "logged progress occurrence", "it/s", rate_series, y_min=0))

    report_files = [
        "gpu_util_power.svg",
        "gpu_memory.svg",
        "loss_curves.svg",
        "map_growth.svg",
        "sdf_valid_gaussian_pct.svg",
        "frame_timing_ms.svg",
        "stderr_progress_rates.svg",
    ]
    title_parts = [env.get("TAG", log_dir.name), env.get("DATASET", ""), env.get("SEQ", "")]
    title = " / ".join(part for part in title_parts if part)
    cards = []
    for filename in report_files:
        if (out_dir / filename).exists():
            cards.append(f'<section><img src="{filename}" alt="{html.escape(filename)}"></section>')
    summary_lines = html.escape((out_dir / "summary.txt").read_text()).splitlines()
    summary_html = "\n".join(f"<tr><td>{line.split('=', 1)[0]}</td><td>{line.split('=', 1)[1] if '=' in line else ''}</td></tr>" for line in summary_lines)
    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)} MPINGS report</title>
  <style>
    body {{ margin: 0; font-family: system-ui, -apple-system, Segoe UI, sans-serif; color: #0f172a; background: #f8fafc; }}
    header, main {{ max-width: 1120px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 6px; font-size: 28px; }}
    p {{ margin: 0; color: #475569; }}
    section {{ margin: 18px 0; background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; }}
    img {{ display: block; width: 100%; height: auto; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; }}
    td {{ padding: 8px 10px; border-bottom: 1px solid #e2e8f0; font-size: 13px; vertical-align: top; }}
    td:first-child {{ width: 280px; color: #475569; font-weight: 650; }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)} MPINGS report</h1>
    <p>Generated from {html.escape(str(log_dir))}</p>
  </header>
  <main>
    <table>{summary_html}</table>
    {''.join(cards)}
  </main>
</body>
</html>
"""
    (out_dir / "report.html").write_text(report)

    return out_dir


def main() -> int:
    args = parse_args()
    if args.log_dir and args.input_dir:
        raise SystemExit("Use either positional log_dir or --input-dir, not both.")
    if args.all and args.out_dir:
        raise SystemExit("--out-dir can only be used when processing one input folder.")

    if args.all:
        log_dirs = list_experiment_dirs(args.all)
        if not log_dirs:
            raise SystemExit(f"No experiment log folders found under: {resolve_log_dir(args.all)}")
        for log_dir in log_dirs:
            out_dir = build_report(log_dir, smooth=args.smooth)
            print(f"Wrote charts to: {out_dir}")
            print(f"Open report: {out_dir / 'report.html'}")
        return 0

    log_dir = resolve_log_dir(args.input_dir or args.log_dir)
    out_dir = build_report(log_dir, args.out_dir, args.smooth)
    print(f"Wrote charts to: {out_dir}")
    print(f"Open report: {out_dir / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
