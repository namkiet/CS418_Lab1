# Evaluating Lightweight VLM and OCR Models on Occluded Documents

Zero-shot benchmark of 5 lightweight Vision-Language Models (VLMs) and dedicated OCR pipelines on the **Occluded Pages Restoration Benchmark (OPRB)** 3,000-page test subset (2,000 text-region + 1,000 non-text-region occlusions).

## Repository Structure

- `report/` — LaTeX source (`main.tex`, `content/*.tex`, `figures/`) and compiled PDF (`report/main.pdf`).
- `notebooks/` — Evaluation notebooks for all 5 models:
  - `OPRB_Florence2_Test_Evaluation.ipynb` (`microsoft/Florence-2-base`)
  - `OPRB_PaddleOCR_VL_0_9B_Test_Evaluation.ipynb` (`PaddlePaddle/PaddleOCR-VL`)
  - `OPRB_RapidOCR_Test_Evaluation.ipynb` (RapidOCR / PP-OCRv6)
  - `OPRB_Tesseract_Test_Evaluation.ipynb` (Tesseract v5.5)
  - `OPRB_EasyOCR_Test_Evaluation.ipynb` (EasyOCR 1.7.2)

## Summary Results (3,000 OPRB Test Images)

| Model | Category | CER ↓ | WER ↓ | Token F1 ↑ | Time (s/img) ↓ |
| --- | --- | :---: | :---: | :---: | :---: |
| Florence-2-base | VLM | 1.0415 | 0.8008 | 0.3519 | **0.82** |
| PaddleOCR-VL | VLM | 0.4194 | 0.5886 | 0.6765 | 9.13 |
| RapidOCR (PP-OCRv6) | OCR | **0.3317** | **0.4283** | **0.7678** | 0.84 |
| Tesseract v5.5 | OCR | 0.4768 | 0.6451 | 0.5194 | 0.95 |
| EasyOCR 1.7.2 | OCR | 0.5930 | 0.8232 | 0.3556 | 2.57 |
