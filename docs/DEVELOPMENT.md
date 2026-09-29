# Development Guide

## 目標

建立可自行部署的 AI Movie Studio。第一階段聚焦於 Image-to-Video MVP，並以 RTX 2060 6GB 作為最低開發基準之一。

## MVP 架構

```text
UI / CLI
  ↓
Python application layer
  ↓
I2V engine abstraction
  ↓
Model backend
  ↓
FFmpeg encode
  ↓
MP4
```

## 開發順序

1. 建立最小 CLI benchmark。
2. 驗證單張圖片可產生 MP4。
3. 記錄 VRAM、耗時、解析度與 frame 數。
4. 建立 engine abstraction。
5. 加入 low-VRAM profile 與 OOM fallback。
6. 再建立 Gradio 或其他簡易 UI。
7. 後續才考慮正式 Web frontend/API queue。

## Engine 介面方向

```python
class VideoEngine:
    def generate(self, image_path, prompt, output_path, **options):
        raise NotImplementedError
```

候選 backend 可實作同一介面：

- LTXVideoEngine
- FramePackEngine
- SVDLowVRAMEngine

## RTX 2060 6GB Low-VRAM 原則

預設策略：

- batch size = 1
- 短片優先
- 低解析度先驗證
- CPU offload（若 backend 支援）
- VAE tiling / slicing（若 backend 支援）
- 必要時降低 frame 數與解析度重試
- 避免模型常駐多份

## 手動驗證流程

本專案不使用 GitHub Actions workflow。每次重要版本需手動執行：

```text
1. 啟動環境
2. 執行 benchmark
3. 使用固定測試圖片與 prompt
4. 記錄生成成功/失敗
5. 記錄 peak VRAM / elapsed time
6. 播放輸出 MP4 檢查可用性
7. 將結果寫入 docs/PITFALLS.md 或版本記錄
```

## Repository 規則

- 不新增 `.github/workflows/`。
- 不使用 CI / QA / AR action workflow。
- 開發決策與環境差異寫入 docs。
- 發現可重現問題時，需記錄觸發條件與 workaround。
- 每次架構或模型版本切換需更新 VERSION_HISTORY。


## LTX-Video backend integration checkpoint

第一個實際 backend 選用官方 LTX-Video 2B 0.9.8 distilled 作 feasibility benchmark。

目前官方 metadata：
- Python requirement: >= 3.10；官方 README 的實測環境為 Python 3.10.5 / CUDA 12.2。
- PyTorch requirement: >= 2.1.x。
- transformers: >=4.47.2,<4.52.0。
- 官方提供 `ltxv-2b-0.9.8-distilled`，描述為較小、較輕 VRAM 的 checkpoint。
- 官方 FP8 kernels 針對 Ada 或更新 GPU；RTX 2060/Turing 不啟用。

整合原則：
1. 保留現有 Torch 2.9.0+cu128，不讓 backend 安裝步驟主動覆蓋 CUDA Torch。
2. LTX package lazy import，未安裝時主程式仍可啟動。
3. RTX 2060 預設 FP16。
4. 第一輪只驗證 load / I2V / OOM 行為，不把成功視為最終品質 profile。
5. benchmark 記錄 elapsed time 與 PyTorch peak allocated VRAM。


## LTX native-crash isolation

After the first real LTX run exited with Windows `0xC0000005`, full generation is paused. `tools/ltx_load_diagnostic.py` runs checkpoint stages in child processes and reports both signed and hexadecimal exit codes.

Stage order:
1. safetensors metadata only
2. VAE `from_pretrained`
3. Transformer `from_pretrained`
4. Transformer BF16 cast
5. Transformer FP16 cast

The diagnostic stops on the first failing stage so a native crash does not obscure the boundary. Each child prints available system RAM and CUDA free/total memory at stage markers.


## RTX 2060 LTX residency decision

Measured cumulative BF16 CUDA residency makes the upstream eager-placement strategy unsuitable for the 6GB target: Transformer alone allocates about 3.58 GiB; Transformer + VAE reaches about 5.93 GiB allocated with no CUDA memory reported free before T5 is loaded.

The low-VRAM backend must therefore alter model residency, not merely sampler dimensions or the FP16/BF16 label. The next implementation target is a CPU-resident pipeline with sequential/model offload hooks so Transformer, VAE and T5 are not simultaneously resident on CUDA during initialization.


## LTX CPU-resident construction prototype

Upstream `create_ltx_video_pipeline()` eagerly moves Transformer, VAE and T5 to the selected device before `LTXVideoPipeline` is returned. The later `offload_to_cpu` argument belongs to the pipeline invocation and therefore cannot prevent the observed 6GB initialization overflow. cite source: upstream `ltx_video/inference.py` inspected 2026-09-29.

`tools/ltx_cpu_pipeline_diagnostic.py` reproduces the base 2B pipeline construction while deliberately omitting all eager `.to("cuda")` calls. Acceptance criteria: pipeline construction completes, Transformer/VAE/T5 all report `cpu`, and CUDA allocated/reserved memory remains near zero. Only after this passes should generation-time offload be tested.
