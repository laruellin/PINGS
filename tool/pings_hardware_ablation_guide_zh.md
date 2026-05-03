# MPINGS 原版基準與診斷實驗指南

這份指南只針對原版 `MPINGS` 規劃，目的是把 baseline 與後續 staged 改版完全拆開。

這次的核心原則：

- `MPINGS` 是本身模型基準
- 先建立原版 non-GS 與 GS 的乾淨硬體基準
- loop closure / PGO 問題獨立成診斷線，不和硬體主表混在一起
- 所有公平硬體比較，主表先以 `--pgo-off` 為準

## 0. 執行方式

從 workspace root 執行：

```bash
cd /home/li0/lio_ws
```

統一入口：

```bash
python3 MPINGS/tool/run_pings.py --yaml <yaml-path>
```

這套 pipeline 會自動：

- 收 `gpu_log_raw.csv`
- 收 `ram_log_raw.csv`
- 保存 `stdout.log` / `stderr.log`
- 產出 charts
- 在每次 run 的 `analysis/` 產出摘要

分析摘要目前可抓：

- `ATE`
- `RPE_trans`
- `RPE_rot`
- `latency_avg_ms`
- `latency_median_ms`
- `latency_p95_ms`
- `latency_total_s`
- `wall_clock_total_s`
- `gpu_vram_peak_mb`
- `system_ram_peak_mb`
- `point_count_final`
- `point_count_peak`
- `mesh_exported`
- `loop_log_written`
- `pose_graph_written`
- `traj_plot_2d_written`
- `traj_plot_3d_written`
- `first_loop_from_frame`
- `first_loop_to_frame`

## 1. A-01 原版 non-GS 硬體基準組

目的：

- 建立 4090 與 2080 在原版非 GS 路徑下的最乾淨比較
- 回答原版 PINGS / PIN-SLAM 路徑在不同 GPU 下，速度、記憶體、ATE/RPE 差多少

設定原則：

- 用 `MPINGS/config/run_pin_slam.yaml`
- 固定 `--pgo-off`
- 不開 GS
- 先短跑 `0-200`
- 之後再長跑完整序列

短跑：

```bash
python3 MPINGS/tool/run_pings.py \
  --yaml MPINGS/config/analysis/a01_mpings_non_gs_kitti00_short.yaml
```

完整序列範例：

```bash
python3 MPINGS/tool/run_pings.py \
  --yaml MPINGS/config/analysis/a01_mpings_non_gs_kitti00_short.yaml \
  --tag 2080_a01_mpings_non_gs_kitti00_full \
  --end-frame 4540
```

主收欄位：

- `ATE`
- `RPE_trans`
- `RPE_rot`
- `latency_avg_ms`
- `latency_median_ms`
- `latency_p95_ms`
- `latency_total_s`
- `gpu_vram_peak_mb`
- `system_ram_peak_mb`
- `point_count_final`
- `point_count_peak`

## 2. A-02 原版 non-GS 閉環診斷組

目的：

- 驗證 `slam` 變形是否主要來自 loop closure / PGO

設定原則：

- 用 `MPINGS/config/run_pin_slam.yaml`
- 資料固定 `KITTI 06` 與 `KITTI 10`
- 每條序列做一組 `--pgo-off`
- 再做一組 `PGO` 維持原設定

KITTI 06:

```bash
python3 MPINGS/tool/run_pings.py \
  --yaml MPINGS/config/analysis/a02_mpings_non_gs_kitti06_pgo_off.yaml

python3 MPINGS/tool/run_pings.py \
  --yaml MPINGS/config/analysis/a02_mpings_non_gs_kitti06_pgo_on.yaml
```

KITTI 10:

```bash
python3 MPINGS/tool/run_pings.py \
  --yaml MPINGS/config/analysis/a02_mpings_non_gs_kitti10_pgo_off.yaml

python3 MPINGS/tool/run_pings.py \
  --yaml MPINGS/config/analysis/a02_mpings_non_gs_kitti10_pgo_on.yaml
```

判讀重點：

- 如果 `pgo-off` 下 `odom` 與 `slam` 都穩
- 而 `pgo-on` 下只有 `slam` 扭曲
- 問題就幾乎可鎖定在 loop acceptance、PGO 權重、或 loop 後 correction propagation

## 3. A-03 原版 GS safe 硬體基準組

目的：

- 量測原版 `MPINGS` 的 GS 路徑在 4090 與 2080 上到底有多重
- 先排除已知環境問題對硬體比較的污染

設定原則：

- 用 `MPINGS/config/run_kitti_gs.yaml`
- 固定 `--pgo-off --monodepth-off`
- 先做短跑 `0-200`
- 不拿這組判斷 loop closure

執行：

```bash
python3 MPINGS/tool/run_pings.py \
  --yaml MPINGS/config/analysis/a03_mpings_gs_safe_kitti00_short.yaml
```

主收欄位：

- `ATE`
- `RPE_trans`
- `RPE_rot`
- `latency_avg_ms`
- `latency_median_ms`
- `latency_p95_ms`
- `gpu_vram_peak_mb`
- `system_ram_peak_mb`
- `mesh_exported`

## 4. A-04 原版 GS full 壓力組

目的：

- 測試 full GS + monodepth + PGO 的完整功能上限與代價

定位：

- 比較適合 4090 或已修好環境時跑
- 不建議當 2080 主要基準
- 如果 2080 跑這組不穩，先不要直接解讀成模型不好

執行：

```bash
python3 MPINGS/tool/run_pings.py \
  --yaml MPINGS/config/analysis/a04_mpings_gs_full_kitti00_short.yaml
```

## 5. A-05 KITTI06 閉環事件分析組

目的：

- 專門分析你觀察到的「閉環後 slam 變形」

做法：

- 固定同一 GPU
- 固定同一 seed
- 固定同一 config
- 只改 `PGO on/off`
- 保留 `odom_poses`、`slam_poses`、`final_pose_graph.g2o`、`traj_plot_2d/3d`、`loop_log.txt`

執行：

```bash
python3 MPINGS/tool/run_pings.py \
  --yaml MPINGS/config/analysis/a05_mpings_kitti06_loop_event_pgo_off.yaml

python3 MPINGS/tool/run_pings.py \
  --yaml MPINGS/config/analysis/a05_mpings_kitti06_loop_event_pgo_on.yaml
```

這組重點不是 FPS，而是：

- `first_loop_from_frame`
- `first_loop_to_frame`
- 軌跡開始明顯折彎的 frame
- 兩者是否時間上重合

## 6. A-06 KITTI10 閉環事件複現組

目的：

- 驗證 A-05 是否為可複現現象，而不是單一序列偶發

執行：

```bash
python3 MPINGS/tool/run_pings.py \
  --yaml MPINGS/config/analysis/a06_mpings_kitti10_loop_event_pgo_off.yaml

python3 MPINGS/tool/run_pings.py \
  --yaml MPINGS/config/analysis/a06_mpings_kitti10_loop_event_pgo_on.yaml
```

如果 06 與 10 都出現：

- loop 後 `slam` 變形
- 但 `odom` 正常

那就代表這比較像原版 `MPINGS / PIN-SLAM` 在目前設定下的 back-end 穩定性問題，而不是單一資料序列偶發。

## 7. 如何在 4090 上做同樣測試

原則是流程不變，只換硬體與標籤。

請沿用同一份 YAML，然後覆寫：

- `--tag`
- `--profile-root ./MPINGS/exp_logs/4090`
- `--gpu-id`

範例，4090 跑 A-01：

```bash
python3 MPINGS/tool/run_pings.py \
  --yaml MPINGS/config/analysis/a01_mpings_non_gs_kitti00_short.yaml \
  --tag 4090_a01_mpings_non_gs_kitti00_short \
  --profile-root ./MPINGS/exp_logs/4090 \
  --gpu-id 0
```

範例，4090 跑 A-03：

```bash
python3 MPINGS/tool/run_pings.py \
  --yaml MPINGS/config/analysis/a03_mpings_gs_safe_kitti00_short.yaml \
  --tag 4090_a03_mpings_gs_safe_kitti00_short \
  --profile-root ./MPINGS/exp_logs/4090 \
  --gpu-id 0
```

範例，4090 跑 A-05：

```bash
python3 MPINGS/tool/run_pings.py \
  --yaml MPINGS/config/analysis/a05_mpings_kitti06_loop_event_pgo_on.yaml \
  --tag 4090_a05_mpings_kitti06_loop_event_pgo_on \
  --profile-root ./MPINGS/exp_logs/4090 \
  --gpu-id 0
```

## 8. 跑完後先看哪裡

每次 run 完成後先看：

- `analysis/pings_analysis_summary.csv`
- `analysis/pings_analysis_summary.json`
- `analysis/pings_analysis_summary.md`

如果是 A-05 / A-06，再補看：

- `loop_log.txt`
- `final_pose_graph.g2o`
- `traj_plot_2d.png`
- `traj_plot_3d.png`
- `slam_poses_kitti.txt`
- `odom_poses_kitti.txt`

## 9. 建議你先跑的順序

如果你要最快建立論文主表與診斷結論，我建議順序是：

1. A-01 2080 short
2. A-01 4090 short
3. A-03 2080 short
4. A-03 4090 short
5. A-05 KITTI06 on/off
6. A-06 KITTI10 on/off

這樣你會先得到：

- 原版 non-GS 硬體基準
- 原版 GS safe 硬體基準
- loop closure 是否是變形主因

## 10. 目前假設

- `MPINGS` 目前沒有獨立 `run_kitti.yaml`，所以 non-GS 主線以 `run_pin_slam.yaml` 為主
- `KITTI 00` 完整序列示例先用 `--end-frame 4540`
- `KITTI 06` 與 `KITTI 10` 的 YAML 目前分別採 `1101` 與 `1201` 作為完整序列上限

如果你本機資料實際 frame 上限不同，直接覆寫 `--end-frame` 即可。
