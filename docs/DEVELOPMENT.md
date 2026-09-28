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
