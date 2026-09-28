# Pitfalls / 踩坑紀錄

此文件集中記錄開發過程中的實際問題、限制、錯誤訊息、原因與 workaround。

## P-001：不要把「模型最小 VRAM」理解成「能順暢使用」

**情境**：RTX 2060 6GB。

**風險**：官方或社群標示 6GB 可執行，可能仍依賴特定 GPU 架構、dtype、offload、RAM 容量或極低解析度。

**處理原則**：

- 一律以實測 benchmark 為準。
- 記錄模型載入是否成功。
- 記錄 peak VRAM。
- 記錄系統 RAM 使用量。
- 記錄生成秒數與輸出品質。

## P-002：RTX 2060 為 Turing，需留意 dtype / kernel 相容性

**風險**：部分新模型或最佳化路徑偏向 Ampere 以上 GPU，可能預設使用 bfloat16、Flash Attention 或特定 CUDA kernel。

**處理原則**：

- 不假設官方推薦設定能直接用於 RTX 2060。
- 若 backend 支援，優先測試 fp16。
- 遇到 kernel / dtype error 時先降級到通用 PyTorch 路徑。
- 不為了單一模型大量魔改環境；若維護成本高，直接換 backend。

## P-003：先驗證 inference，不先做完整 UI

**原因**：MVP 最大風險是 6GB VRAM 能否穩定生成，而不是 UI。

**處理原則**：

先完成：

```text
image + prompt -> CLI -> MP4
```

成功後才封裝為 engine，再做 UI。

## P-004：OOM fallback 必須可預期

遇到 CUDA Out Of Memory 時，不應只讓程式崩潰。

預計 fallback 順序：

```text
第一次 OOM
→ 清理 CUDA cache
→ 降低解析度
→ 重試

仍 OOM
→ 減少 frames / duration
→ 重試

仍失敗
→ 回報可讀錯誤與實際參數
```

## P-005：Windows WDDM 會吃掉可用 VRAM

RTX 2060 6GB 在 Windows 桌面環境中，顯示系統、瀏覽器、Discord、NVIDIA Overlay、Photos 等程式會共同佔用顯示記憶體。

2026-09-28 初始環境實測：

```text
GPU: NVIDIA GeForce RTX 2060
Total VRAM: 6144 MiB
Desktop idle-ish usage: 約 2457 MiB
Driver: 616.92
CUDA UMD: 13.4
Driver model: WDDM
```

這表示在啟動推論前，實際可用 VRAM 可能只剩約 3.5GB，而不是完整 6GB。

**處理原則**：

- benchmark 前關閉不必要的 Chrome/Edge、Discord、Photos、NVIDIA Overlay 等 GPU 程式。
- 不把 `6144 MiB` 當作模型可以完整使用的 VRAM。
- 每次 benchmark 同時記錄推論前的 idle VRAM。
- 必要時設計 `preflight` 檢查，在可用 VRAM 過低時警告使用者。

## P-006：系統 CUDA 版本 ≠ PyTorch 必須使用相同 CUDA Toolkit

`nvidia-smi` 顯示的 CUDA UMD 版本代表驅動可支援的 CUDA 上限之一，不代表專案必須安裝相同版本 CUDA Toolkit。

目前機器顯示：

```text
CUDA UMD Version: 13.4
```

專案應優先使用 PyTorch 官方預編譯 CUDA wheel 自帶的 runtime，避免為了對齊 `nvidia-smi` 顯示版本去另外安裝整套 CUDA Toolkit。

## P-007：Python 3.12 可用，但模型專案要以實際依賴相容性為準

目前系統：

```text
Python 3.12.4
Git 2.55.0.windows.5
```

LTX-Video 專案宣告 Python `>=3.10`，但官方 README 的已測環境是 Python 3.10.5 / CUDA 12.2。

**MVP 原則**：

- 不污染系統 Python。
- 使用專案獨立 venv。
- 若 3.12 遇到依賴或編譯問題，優先建立 Python 3.10/3.11 專案環境，而不是修改系統環境。

## P-008：FramePack 對 RTX 20XX 不屬官方已測範圍

官方 FramePack 要求至少 6GB VRAM，但明確指出 RTX 20XX 未測試；因此不能只因為「6GB」就把它視為 RTX 2060 的確定解法。

**策略**：先把 FramePack 視為 benchmark 候選，不作為第一個必須成功的 backend。


## P-009：首次 preflight 實測（Python 3.12 venv）

2026-09-28 本機手動執行 `python tools\\preflight.py`：

```text
Python: 3.12.4
OS: Windows 11 10.0.22631
System RAM: 63.94 GiB total / 35.48 GiB available
GPU: NVIDIA GeForce RTX 2060
VRAM: 6144 MiB total / 2480 MiB used / 3475 MiB free
Driver: 616.92
FFmpeg: NOT FOUND
PyTorch: NOT INSTALLED
```

**結論**：

- 64GB system RAM 足夠支援後續 CPU offload 實驗。
- Windows WDDM 桌面狀態下僅約 3.4GB GPU VRAM 可直接使用；正式 benchmark 前應關閉非必要 GPU 應用。
- FFmpeg 尚未安裝，列為本機 runtime dependency，不使用 GitHub Actions 自動安裝。
- 下一關先固定 PyTorch CUDA wheel 並執行 FP16 smoke test，再安裝任何 I2V backend。


## P-010：PyTorch CUDA preflight 通過，但 BF16 API 回報不可直接當硬體能力

2026-09-28 使用：

```text
torch: 2.9.0+cu128
PyTorch CUDA runtime: 12.8
GPU: RTX 2060
Compute capability: 7.5
FP16 smoke test: PASS
torch.cuda.is_bf16_supported(): True
```

同次測試中 PyTorch `mem_get_info` 回報約 4.99 GiB free / 6.00 GiB total，與 `nvidia-smi` 的 WDDM process/accounting 顯示不同，因此兩者不可直接互相比較。

**重要**：RTX 2060 / Turing (SM 7.5) 不具 Ampere 等級的原生 BF16 Tensor Core 路徑。新版 PyTorch 的 `is_bf16_supported()` 回傳值可能代表框架可執行/模擬相關 dtype，而不是證明此 GPU 有原生 BF16 Tensor Core 加速。

**專案策略**：

- RTX 2060 backend 預設使用 FP16。
- 不因 `is_bf16_supported() == True` 自動選 BF16。
- dtype policy 同時考慮 GPU architecture / compute capability。
- 發現缺少 NumPy 的 warning；在模型依賴安裝前補上 NumPy。


## P-011：直接執行子目錄 script 時找不到 repository package

**錯誤**：

```text
python benchmarks\\i2v.py
ModuleNotFoundError: No module named 'engine'
```

**原因**：Python 直接執行 `benchmarks/i2v.py` 時，`sys.path[0]` 是 `benchmarks/`，repository root 不一定在 module search path，因此 sibling package `engine/` 無法 import。

**修正**：benchmark entrypoint 明確將 repository root 加入 `sys.path`。MVP 後續若正式 package 化，改用 `pyproject.toml` + editable install，避免各 entrypoint 重複處理 import path。
