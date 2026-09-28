# Ai-Movie-Studio

本專案目標：建立一套可自行部署、可在本機或自管環境執行的 AI Movie Studio MVP，先從 Image-to-Video 開始，逐步擴展到腳本、分鏡、配音、Lip Sync、剪輯與輸出。

## 開發原則

1. 開發文件、架構決策、已知問題、踩坑紀錄與版本記錄都集中保存在此 repository。
2. **不使用 CI、QA、AR、GitHub Actions workflow 自動化。**
3. 優先支援低 VRAM 環境；目前基準硬體為 **NVIDIA RTX 2060 6GB**。
4. MVP 優先驗證可運作性，再處理 UI 與產品化。
5. 推論核心與 UI 解耦，避免綁死 ComfyUI。

## MVP 第一階段

目標：

```text
input image + prompt
        ↓
local I2V engine
        ↓
output MP4
```

預計結構：

```text
Ai-Movie-Studio/
├── docs/
│   ├── DEVELOPMENT.md
│   ├── PITFALLS.md
│   └── VERSION_HISTORY.md
├── src/
│   ├── engine/
│   └── app.py
├── inputs/
├── outputs/
├── requirements.txt
└── README.md
```

## 當前技術方向

第一輪先做本機 Image-to-Video feasibility test，不先建立複雜前端。

候選 inference engine：

- LTX-Video 2B / distilled
- FramePack
- Stable Video Diffusion low-VRAM baseline

選型重點：

- RTX 2060 6GB 是否可執行
- 生成時間
- Peak VRAM
- 影像品質
- 是否能以純 Python API 封裝
- 是否容易加入 CPU offload / fallback

## 禁止事項

本 repository 不加入以下自動化：

- `.github/workflows/*`
- GitHub Actions CI
- 自動 QA workflow
- AR / auto-review workflow
- 自動 merge workflow

測試與驗證在 MVP 階段採本機手動執行並記錄結果。
