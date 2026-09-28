# Version History

## v0.0.1-dev — 2026-09-28

### 專案初始化

- 建立 Ai-Movie-Studio repository 開發規則。
- 明確禁止 CI / QA / AR / GitHub Actions workflow。
- MVP 第一階段聚焦 Image-to-Video 本機推論。
- 基準硬體：NVIDIA GeForce RTX 2060 6GB。
- 建立 `docs/DEVELOPMENT.md`。
- 建立 `docs/PITFALLS.md`。

### 初始開發環境

```text
OS: Windows（WDDM）
GPU: NVIDIA GeForce RTX 2060 6GB
Driver: 616.92
CUDA UMD: 13.4
Python: 3.12.4
Git: 2.55.0.windows.5
```

### 初始風險

- 啟動推論前 GPU 已使用約 2457 MiB / 6144 MiB。
- RTX 2060 為 Turing，需避免假設 bf16 / 新式 attention kernel 可用。
- FramePack 官方未將 RTX 20XX 列為已測平台。
- LTX-Video 宣告 Python >=3.10，但官方已測環境較接近 Python 3.10；若 Python 3.12 依賴出問題，改用獨立 Python 3.10/3.11 venv。

### 下一步

建立本機 preflight / benchmark tooling，先驗證：

1. PyTorch CUDA 是否正確辨識 RTX 2060。
2. 實際 free VRAM。
3. fp16 能力。
4. ffmpeg 是否可用。
5. 再決定第一個 I2V backend。


### 2026-09-28 Preflight checkpoint

- 專案 venv 建立成功，Python executable 指向 `.venv\\Scripts\\python.exe`。
- `psutil 7.2.2` 安裝成功。
- 系統 RAM 實測 63.94 GiB total / 35.48 GiB available。
- RTX 2060 實測 6144 MiB total / 3475 MiB free（測試當下）。
- FFmpeg 尚未安裝。
- PyTorch 尚未安裝；下一 checkpoint 為 CUDA/FP16 smoke test。


### 2026-09-28 CUDA checkpoint PASS

```text
PyTorch: 2.9.0+cu128
CUDA runtime: 12.8
CUDA available: True
GPU: RTX 2060
Compute capability: 7.5
VRAM: 6.00 GiB
PyTorch free VRAM at test: 4.99 GiB
FP16 smoke test: PASS
STATUS: PASS
```

- 發現 NumPy 尚未安裝。
- PyTorch 回報 BF16 supported=True，但 RTX 2060/Turing 的 backend policy 仍固定優先 FP16，不把該 API 當作原生 BF16 Tensor Core 能力判定。
- CUDA 基礎環境驗證完成，下一階段進入 I2V backend feasibility benchmark。


### 2026-09-28 engine bootstrap checkpoint PASS

實機輸出：

```text
GPU: NVIDIA GeForce RTX 2060
Compute capability: 7.5
VRAM: ~6.00 GiB
PyTorch free VRAM: ~4.99 GiB
System RAM: ~63.94 GiB
Available RAM: ~32.55 GiB
Profile: low_vram
dtype: float16
cpu_offload: true
batch_size: 1
benchmark target: 512x288 / 2.0s / <=49 frames
STATUS: READY FOR BACKEND BENCHMARK
```

下一階段：實測第一個 I2V backend。首選候選為 LTX-Video 2B distilled，但必須先做相容性/VRAM feasibility 驗證；RTX 2060 不使用 Ada+ 專用 FP8 kernels，且專案仍以 FP16 為 Turing 預設。


### 2026-09-28 LTX base compatibility PASS

```text
Python: 3.12.4
Torch: 2.9.0+cu128
CUDA: 12.8
GPU: RTX 2060 / SM 7.5
LTX package: not installed
STATUS: BASE COMPATIBLE
```

Source review 後調整 benchmark 計畫：不直接以 2B 0.9.8 multi-scale/BF16 config 作 RTX 2060 首測；先以 2B 0.9.6 distilled 8-step base pipeline 驗證 feasibility。


### 2026-09-28 LTX checkpoint downloaded

```text
ltxv-2b-0.9.6-distilled-04-25.safetensors
size: ~6.34 GB
location: models/ltx/
```

Added the first real `--backend ltx` benchmark path. Initial defaults are intentionally conservative: 512x288, 25 frames, seed 12345, CPU offload enabled by the low-VRAM profile.
