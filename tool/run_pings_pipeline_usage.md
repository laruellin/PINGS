# `run_pings.py` 使用教學

## 1. 這支程式做什麼

`tool/run_pings.py` 是把你目前手動流程整合起來的工具。

它會依序做這些事：

1. 執行 `pings.py`
2. 同步記錄 GPU 使用量到 `gpu_log_raw.csv`
3. 同步記錄 RAM 使用量到 `ram_log_raw.csv`
4. 將 `stdout` 與 `stderr` 存成 log
5. 執行 `make_exp_log_charts.py`
6. 可選擇執行 `analyze_pings.py`
7. 自動找到本次 `pings.py` 的輸出資料夾
8. 將 profiling 資料複製或移動到該次 run 的 `log/` 裡
9. 可選擇壓縮 profiling 資料夾或整個 run 資料夾

---

## 2. 目前相關檔案

主要程式：

`MPINGS/tool/run_pings.py`

YAML 範例：

`MPINGS/tool/run_pings.yaml`

教學：

`MPINGS/tool/run_pings_pipeline_usage.md`

---

## 3. 最推薦的使用方式

建議你之後用 YAML 管理設定，執行時只要：

```bash
python3 tool/run_pings.py --yaml tool/run_pings.yaml
```

這樣之後你只需要改 `tool/run_pings.yaml`，不用每次打一長串參數。

---

## 4. YAML 怎麼改

目前範例 YAML 長這樣：

```yaml
workdir: "./MPINGS"
config: "./config/run_kitti_gs.yaml"
dataset: "kitti"
seq: "06"
input_path: "./data/"
tag: "4090_gs_on_mid_s06"

start_frame: 0
end_frame: 1101
step_frame: 1
gpu_id: "0"

gs_mode: "on"
vis_mode: "off"

run_charts: "on"
run_analysis: "on"
transfer_mode: "copy"
zip_profile: "on"
zip_run: "off"
```

你最常會改的通常是：

- `workdir`
- `config`
- `dataset`
- `seq`
- `input_path`
- `tag`
- `start_frame`
- `end_frame`
- `step_frame`
- `gpu_id`

---

## 5. 你目前 KITTI 流程對應設定

你原本的命令大致是：

```bash
TAG=4090_gs_on_mid_s06 \
GPU_ID=0 \
CONFIG=./config/run_kitti_gs.yaml \
INPUT_PATH=./data/ \
START_FRAME=0 \
END_FRAME=1101 \
STEP_FRAME=1 \
GS_MODE=on \
VIS_MODE=off \
SEQ=06 \
./scripts/run_pings_profile.sh
```

改成 YAML 後，對應可以寫成：

```yaml
workdir: "./MPINGS"
config: "./config/run_kitti_gs.yaml"
dataset: "kitti"
seq: "06"
input_path: "./data/"
tag: "4090_gs_on_mid_s06"

start_frame: 0
end_frame: 1101
step_frame: 1
gpu_id: "0"

gs_mode: "on"
vis_mode: "off"

run_charts: "on"
run_analysis: "on"
transfer_mode: "copy"
zip_profile: "on"
```

然後直接執行：

```bash
python3 tool/run_pings.py --yaml tool/run_pings.yaml
```

---

## 6. 也可以直接用命令列，不靠 YAML

如果你臨時想測試，也可以直接下：

```bash
python3 tool/run_pings.py \
  --tag 4090_gs_on_mid_s06 \
  --gpu-id 0 \
  --config ./config/run_kitti_gs.yaml \
  --dataset kitti \
  --seq 06 \
  --input-path ./data/ \
  --start-frame 0 \
  --end-frame 1101 \
  --step-frame 1 \
  --gs-mode on \
  --vis-mode off \
  --run-charts on \
  --run-analysis on \
  --transfer-mode copy \
  --zip-profile on
```

---

## 7. YAML 與命令列可以混用

這支程式支援：

1. 先讀 YAML
2. 再用命令列覆蓋部分欄位

例如：

```bash
python3 tool/run_pings.py \
  --yaml tool/run_pings.yaml \
  --tag debug_seq06 \
  --vis-mode on
```

這代表：

- 其他設定沿用 YAML
- 只有 `tag` 改成 `debug_seq06`
- `vis_mode` 改成 `on`

---

## 8. rosbag / topic 的用法

目前這支流程可以支援 `rosbag`，也可以指定單一 point cloud topic。

### rosbag 的重點

- `dataset` 要設成 `rosbag`
- `seq` 要填 point cloud topic
- `input_path` 要填 bag 檔案路徑或 bag 資料夾

例如：

```yaml
config: "./config/your_rosbag_config.yaml"
dataset: "rosbag"
seq: "/velodyne_points"
input_path: "/path/to/your.bag"
tag: "rosbag_test"

gs_mode: "on"
vis_mode: "off"
run_charts: "on"
run_analysis: "on"
transfer_mode: "copy"
zip_profile: "on"
```

然後執行：

```bash
python3 tool/run_pings.py --yaml tool/run_pings.yaml
```

### rosbag 注意事項

1. `rosbag` dataloader 目前是讀 `sensor_msgs/msg/PointCloud2`
2. 目前不支援從 rosbag 直接讀影像
3. 如果 bag 裡有多個 point cloud topic，必須明確填 `seq`
4. 如果環境沒有安裝 `rosbags`，需要先安裝：

```bash
pip install -U rosbags
```

---

## 9. 常用欄位說明

### 執行主設定

- `config`
  `pings.py` 的 YAML 設定檔

- `workdir`
  實際執行 `pings.py` / chart / analysis 的工作目錄。當你要從同一個 pipeline 同時跑 `MPINGS` 與 `PINGS-2080` 時，這個欄位很重要，因為 config 內像 `./cad/...` 這種相對路徑會依這個目錄解析。

- `dataset`
  資料集類型，例如 `kitti`、`rosbag`

- `seq`
  序列名稱；如果是 `rosbag`，這裡就是 topic

- `input_path`
  輸入資料路徑

- `output_path`
  可選，指定 `pings.py` 的輸出根目錄

- `tag`
  本次實驗名稱

### Frame 控制

- `start_frame`
- `end_frame`
- `step_frame`

### 模式控制

- `gs_mode: "on" | "off"`
- `vis_mode: "on" | "off"`
- `save_map: "on" | "off"`
- `save_mesh: "on" | "off"`
- `save_merged_pc: "on" | "off"`

### 後處理控制

- `run_charts: "on" | "off"`
- `run_analysis: "on" | "off"`
- `transfer_mode: "off" | "copy" | "move"`

說明：

- `off`：不整理過去
- `copy`：複製到 `run_path/log/<tag>/`
- `move`：直接搬移到 `run_path/log/<tag>/`

### 壓縮控制

- `zip_profile: "on" | "off"`
- `zip_run: "on" | "off"`

---

## 10. 額外參數

如果還想把其他參數直接傳給 `pings.py`，可以用：

```yaml
extra_args: "--deskew --tracker-off"
```

或命令列：

```bash
python3 tool/run_pings.py \
  --yaml tool/run_pings.yaml \
  --extra-args "--deskew --tracker-off"
```

---

## 11. 常見輸出位置

### Profiling 資料夾

預設會放在：

```bash
./exp_logs/<tag>/
```

例如：

```bash
./exp_logs/4090_gs_on_mid_s06/
```

裡面通常會有：

- `gpu_log_raw.csv`
- `ram_log_raw.csv`
- `stdout.log`
- `stderr.log`
- `run_meta.env`
- `command.sh`
- `charts/`

### PINGS 輸出資料夾

這是 `pings.py` 本身的輸出資料夾。

如果你設定：

```yaml
transfer_mode: "copy"
```

profiling 資料會再同步到：

```bash
<run_path>/log/<tag>/
```

---

## 12. 最推薦的實際指令

如果你現在就想直接開始用，先做兩件事：

1. 修改 `tool/run_pings.yaml`
2. 執行：

```bash
python3 tool/run_pings.py --yaml tool/run_pings.yaml
```

如果你只是臨時想覆蓋某個值，例如改 tag：

```bash
python3 tool/run_pings.py --yaml tool/run_pings.yaml --tag test_run
```

---

## 13. 補充

如果你想看預設 YAML 範本，也可以直接輸出：

```bash
python3 tool/run_pings.py --dump-default-yaml
```
