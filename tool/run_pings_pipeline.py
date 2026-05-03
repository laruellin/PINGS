#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PINGS + profiling + chart + analysis in one pipeline."
    )

    parser.add_argument("--python-bin", default="python3")
    parser.add_argument("--pings-main", default="./pings.py")
    parser.add_argument("--chart-script", default="./tool/make_exp_log_charts.py")
    parser.add_argument("--analyze-script", default="./tool/analyze_pings.py")

    parser.add_argument("--config", default="./config/run_kitti_gs.yaml")
    parser.add_argument("--dataset", default="kitti")
    parser.add_argument("--seq", default="00")
    parser.add_argument("--input-path", default="./data/")
    parser.add_argument(
        "--output-path",
        default="",
        help="Optional output root for pings.py",
    )
    parser.add_argument("--tag", default="4090_gs_on_mid_s06")

    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=1101)
    parser.add_argument("--step-frame", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", default="0")

    parser.add_argument("--gs-mode", choices=["on", "off"], default="on")
    parser.add_argument("--vis-mode", choices=["on", "off"], default="off")
    parser.add_argument("--save-map", choices=["on", "off"], default="off")
    parser.add_argument("--save-mesh", choices=["on", "off"], default="off")
    parser.add_argument("--save-merged-pc", choices=["on", "off"], default="off")

    parser.add_argument("--profile-root", default="./exp_logs")
    parser.add_argument("--chart-smooth", type=int, default=9)

    parser.add_argument("--run-charts", choices=["on", "off"], default="on")
    parser.add_argument("--run-analysis", choices=["on", "off"], default="on")
    parser.add_argument(
        "--transfer-mode",
        choices=["off", "copy", "move"],
        default="copy",
    )

    parser.add_argument("--zip-profile", choices=["on", "off"], default="on")
    parser.add_argument("--zip-run", choices=["on", "off"], default="off")

    parser.add_argument("--analysis-dir-name", default="analysis")
    parser.add_argument("--analysis-output-name", default="pings_analysis_summary")

    parser.add_argument(
        "--extra-args",
        default="",
        help='Extra args passed to pings.py, example: "--deskew --tracker-off"',
    )

    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def stop_process(proc: Optional[subprocess.Popen]) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def zip_directory(source_dir: Path, archive_path: Path) -> None:
    if archive_path.exists():
        archive_path.unlink()
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in source_dir.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(source_dir.parent)))


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    idx = 1
    while True:
        candidate = path.parent / f"{path.name}_{idx}"
        if not candidate.exists():
            return candidate
        idx += 1


def start_gpu_logger(output_csv: Path, gpu_id: str) -> subprocess.Popen:
    output_file = output_csv.open("w", encoding="utf-8")
    cmd = [
        "nvidia-smi",
        f"--id={gpu_id}",
        "--query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
        "-l",
        "1",
    ]
    return subprocess.Popen(cmd, stdout=output_file, stderr=subprocess.DEVNULL, text=True)


def start_ram_logger(output_csv: Path) -> subprocess.Popen:
    output_file = output_csv.open("w", encoding="utf-8")
    code = r"""
import subprocess
import time
from datetime import datetime

print("timestamp,mem_used_mb", flush=True)
while True:
    out = subprocess.check_output(["free", "-m"], text=True)
    used = "0"
    for line in out.splitlines():
        if line.startswith("Mem:"):
            parts = line.split()
            if len(parts) >= 3:
                used = parts[2]
            break
    ts = datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
    print(f"{ts},{used}", flush=True)
    time.sleep(1)
"""
    return subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=output_file,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def tee_process_output(
    process: subprocess.Popen,
    stdout_path: Path,
    stderr_path: Path,
) -> int:
    with stdout_path.open("w", encoding="utf-8") as fout, stderr_path.open(
        "w", encoding="utf-8"
    ) as ferr:
        try:
            while True:
                out_line = process.stdout.readline() if process.stdout else ""
                err_line = process.stderr.readline() if process.stderr else ""

                if out_line:
                    sys.stdout.write(out_line)
                    fout.write(out_line)
                    fout.flush()

                if err_line:
                    sys.stderr.write(err_line)
                    ferr.write(err_line)
                    ferr.flush()

                if process.poll() is not None:
                    remaining_out = process.stdout.read() if process.stdout else ""
                    remaining_err = process.stderr.read() if process.stderr else ""

                    if remaining_out:
                        sys.stdout.write(remaining_out)
                        fout.write(remaining_out)
                    if remaining_err:
                        sys.stderr.write(remaining_err)
                        ferr.write(remaining_err)

                    fout.flush()
                    ferr.flush()
                    return process.returncode
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT)
            process.wait()
            raise


def build_pings_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        args.python_bin,
        args.pings_main,
        args.config,
        args.dataset,
        args.seq,
        "-i",
        args.input_path,
        "--range",
        str(args.start_frame),
        str(args.end_frame),
        str(args.step_frame),
        "--seed",
        str(args.seed),
        "--log-on",
        "--tag",
        args.tag,
    ]

    if args.output_path:
        cmd += ["-o", args.output_path]
    if args.gs_mode == "off":
        cmd.append("--gs-off")
    if args.vis_mode == "on":
        cmd.append("--visualize")
    if args.save_map == "on":
        cmd.append("--save-map")
    if args.save_mesh == "on":
        cmd.append("--save-mesh")
    if args.save_merged_pc == "on":
        cmd.append("--save-merged-pc")
    if args.extra_args.strip():
        cmd += shlex.split(args.extra_args)

    return cmd


def parse_run_path_from_stdout(stdout_log: Path) -> Optional[Path]:
    if not stdout_log.exists():
        return None
    text = stdout_log.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"^Start\s+(.+)$", text, re.MULTILINE)
    if not match:
        return None
    run_path = Path(match.group(1).strip())
    return run_path if run_path.exists() else None


def write_meta(
    profile_dir: Path,
    args: argparse.Namespace,
    pings_cmd: list[str],
    run_start_ms: int,
) -> None:
    meta = [
        f"RUN_START_MS={run_start_ms}",
        f"TAG={args.tag}",
        f"GPU_ID={args.gpu_id}",
        f"CONFIG={args.config}",
        f"DATASET={args.dataset}",
        f"SEQ={args.seq}",
        f"INPUT_PATH={args.input_path}",
        f"OUTPUT_PATH={args.output_path}",
        f"START_FRAME={args.start_frame}",
        f"END_FRAME={args.end_frame}",
        f"STEP_FRAME={args.step_frame}",
        f"SEED={args.seed}",
        f"GS_MODE={args.gs_mode}",
        f"VIS_MODE={args.vis_mode}",
        f"SAVE_MAP={args.save_map}",
        f"SAVE_MESH={args.save_mesh}",
        f"SAVE_MERGED_PC={args.save_merged_pc}",
        f"RUN_CHARTS={args.run_charts}",
        f"RUN_ANALYSIS={args.run_analysis}",
        f"TRANSFER_MODE={args.transfer_mode}",
        f"ZIP_PROFILE={args.zip_profile}",
        f"ZIP_RUN={args.zip_run}",
        f"ANALYSIS_DIR_NAME={args.analysis_dir_name}",
        f"ANALYSIS_OUTPUT_NAME={args.analysis_output_name}",
        f"EXTRA_ARGS={args.extra_args}",
    ]
    write_text(profile_dir / "run_meta.env", "\n".join(meta) + "\n")
    write_text(
        profile_dir / "command.sh",
        " ".join(shlex.quote(item) for item in pings_cmd) + "\n",
    )


def run_subprocess(cmd: list[str], name: str) -> None:
    print(f"[pipeline] running {name}: {' '.join(shlex.quote(x) for x in cmd)}")
    subprocess.run(cmd, check=True)


def maybe_run_charts(args: argparse.Namespace, profile_dir: Path) -> None:
    if args.run_charts != "on":
        return
    cmd = [
        args.python_bin,
        args.chart_script,
        "--input-dir",
        str(profile_dir),
        "--out-dir",
        str(profile_dir / "charts"),
        "--smooth",
        str(args.chart_smooth),
    ]
    run_subprocess(cmd, "make_exp_log_charts.py")


def sync_profile_into_run(
    profile_dir: Path,
    run_path: Path,
    tag: str,
    mode: str,
) -> Optional[Path]:
    if mode == "off":
        return None
    target = unique_path(run_path / "log" / tag)
    ensure_dir(target.parent)
    if mode == "copy":
        shutil.copytree(profile_dir, target)
    elif mode == "move":
        shutil.move(str(profile_dir), str(target))
    return target


def maybe_run_analysis(args: argparse.Namespace, run_path: Path) -> None:
    if args.run_analysis != "on":
        return
    analysis_dir = run_path / args.analysis_dir_name
    ensure_dir(analysis_dir)
    cmd = [
        args.python_bin,
        args.analyze_script,
        str(run_path),
        "--mode",
        "single",
        "--output-dir",
        str(analysis_dir),
        "--output-name",
        args.analysis_output_name,
        "--archive",
    ]
    run_subprocess(cmd, "analyze_pings.py")


def main() -> int:
    args = parse_args()

    profile_dir = Path(args.profile_root).resolve() / args.tag
    ensure_dir(profile_dir)

    stdout_log = profile_dir / "stdout.log"
    stderr_log = profile_dir / "stderr.log"
    run_start_ms = int(time.time() * 1000)
    write_text(profile_dir / "run_start_ms.txt", f"{run_start_ms}\n")

    pings_cmd = build_pings_command(args)
    write_meta(profile_dir, args, pings_cmd, run_start_ms)

    print(f"[pipeline] profile dir: {profile_dir}")
    print("[pipeline] starting pings")

    gpu_proc = None
    ram_proc = None

    try:
        gpu_proc = start_gpu_logger(profile_dir / "gpu_log_raw.csv", args.gpu_id)
        ram_proc = start_ram_logger(profile_dir / "ram_log_raw.csv")

        pings_proc = subprocess.Popen(
            pings_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        return_code = tee_process_output(pings_proc, stdout_log, stderr_log)
    finally:
        stop_process(gpu_proc)
        stop_process(ram_proc)
        run_end_ms = int(time.time() * 1000)
        write_text(profile_dir / "run_end_ms.txt", f"{run_end_ms}\n")

    if return_code != 0:
        print(f"[pipeline] pings failed with code {return_code}", file=sys.stderr)
        return return_code

    maybe_run_charts(args, profile_dir)

    run_path = parse_run_path_from_stdout(stdout_log)
    if run_path is None:
        print(
            "[pipeline] warning: could not detect pings output dir from stdout.log",
            file=sys.stderr,
        )
    else:
        print(f"[pipeline] detected run path: {run_path}")

    profile_zip = profile_dir.with_suffix(".zip")
    if args.zip_profile == "on":
        print("[pipeline] zipping profile dir")
        zip_directory(profile_dir, profile_zip)

    synced_profile = None
    if run_path is not None:
        synced_profile = sync_profile_into_run(
            profile_dir, run_path, args.tag, args.transfer_mode
        )

        if args.zip_profile == "on" and profile_zip.exists():
            shutil.copy2(profile_zip, run_path / "log" / profile_zip.name)

        maybe_run_analysis(args, run_path)

        if args.zip_run == "on":
            print("[pipeline] zipping run dir")
            zip_directory(run_path, run_path.with_suffix(".zip"))

    print("[pipeline] finished")
    if synced_profile is not None:
        print(f"[pipeline] synced profile: {synced_profile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
