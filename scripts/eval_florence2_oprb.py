#!/usr/bin/env python3
"""Evaluate microsoft/Florence-2-base (or compatible) on the OPRB test set."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import tarfile
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from jiwer import cer, wer
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def _apply_transformers_compat_shim() -> None:
    """Restore attributes expected by Florence-2 remote code on transformers >= 5.0."""
    if not hasattr(PretrainedConfig, "forced_bos_token_id"):
        PretrainedConfig.forced_bos_token_id = None
    if not hasattr(PreTrainedTokenizerBase, "additional_special_tokens"):
        PreTrainedTokenizerBase.additional_special_tokens = property(
            lambda self: getattr(self, "all_special_tokens", [])
        )
    if not hasattr(PreTrainedModel, "_supports_sdpa"):
        PreTrainedModel._supports_sdpa = False

REPO_ID = "kpurkayastha/OPRB"
CATEGORIES = ["text", "nontext"]
EXPECTED_COUNTS = {
    ("text", "img"): 6000,
    ("text", "gt"): 6000,
    ("nontext", "img"): 3000,
    ("nontext", "gt"): 3000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Florence-2 OCR evaluation on the OPRB test dataset."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="microsoft/Florence-2-base",
        help="HuggingFace model ID (default: microsoft/Florence-2-base).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./oprb_test"),
        help="Local directory to store/load OPRB test data (default: ./oprb_test).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("./oprb_results/per_image_results.csv"),
        help="Path to output CSV file with per-image OCR text and metrics.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("./oprb_results/summary_results.json"),
        help="Path to output JSON file with aggregated evaluation results.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Inference batch size (default: 4).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=-1,
        help="Number of samples to evaluate (-1 for full dataset, default: -1).",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=25,
        help="Save intermediate CSV every N processed samples (default: 25).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output CSV by skipping samples with status=='ok'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed when subsampling (default: 42).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum new tokens to generate per image (default: 1024).",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=3,
        help="Number of beams for generation (default: 3).",
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Disable case folding during metric computation.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.save_every < 1:
        parser.error("--save-every must be >= 1")
    if args.num_samples == 0 or args.num_samples < -1:
        parser.error("--num-samples must be -1 (full) or a positive integer")
    return args


def _extract_tar_flat(archive_path: str | Path, destination: Path, suffix: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if member.isfile() and member.name.lower().endswith(suffix):
                target = destination / Path(member.name).name
                if not target.exists():
                    src = archive.extractfile(member)
                    if src is not None:
                        with src, target.open("wb") as dst:
                            shutil.copyfileobj(src, dst)


def download_data(data_dir: Path) -> None:
    """Download OPRB test archives into data_dir only if not already present."""
    for category in CATEGORIES:
        for kind, suffix in [("img", ".jpg"), ("gt", ".txt")]:
            expected = EXPECTED_COUNTS[(category, kind)]
            folder = data_dir / category / kind
            existing_count = len(list(folder.glob(f"*{suffix}"))) if folder.exists() else 0
            if existing_count >= expected:
                print(
                    f"[data] Found {existing_count}/{expected} {suffix} files in {folder}, skipping download."
                )
                continue

            print(
                f"[data] Downloading test/{category}/{kind}.tar.gz from {REPO_ID} into {folder}..."
            )
            archive = hf_hub_download(
                REPO_ID,
                f"test/{category}/{kind}.tar.gz",
                repo_type="dataset",
            )
            _extract_tar_flat(archive, folder, suffix)


def collect_pairs(data_dir: Path, num_samples: int, seed: int) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    for category in CATEGORIES:
        image_dir = data_dir / category / "img"
        gt_dir = data_dir / category / "gt"
        names = sorted(
            {p.stem for p in image_dir.glob("*.jpg")} & {p.stem for p in gt_dir.glob("*.txt")}
        )
        for stem in names:
            parts = stem.split("__", 2)
            pairs.append(
                {
                    "category": category,
                    "sample_id": stem,
                    "image_path": str(image_dir / f"{stem}.jpg"),
                    "gt_path": str(gt_dir / f"{stem}.txt"),
                    "occlusion_type": parts[0],
                    "occlusion_density": parts[1] if len(parts) > 1 else "unknown",
                }
            )
    if not pairs:
        raise RuntimeError(f"No paired OPRB files found in {data_dir}")

    if num_samples > 0 and num_samples < len(pairs):
        pairs = random.Random(seed).sample(pairs, num_samples)
        pairs.sort(key=lambda p: (p["category"], p["sample_id"]))
    return pairs


def gt_text(path: str | Path) -> str:
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        return " ".join(
            line.rstrip("\r\n").split("\t")[0].strip()
            for line in f
            if "\t" in line and line.split("\t", 1)[0].strip()
        )


def norm_text(s: str, normalize_case: bool) -> str:
    s = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip()
    return s.casefold() if normalize_case else s


def compute_metrics(ref: str, hyp: str, normalize_case: bool) -> dict[str, float]:
    ref_n = norm_text(ref, normalize_case)
    hyp_n = norm_text(hyp, normalize_case)
    if not ref_n:
        raise ValueError("empty ground truth")
    a, b = Counter(ref_n.split()), Counter(hyp_n.split())
    matched = sum((a & b).values())
    precision = matched / sum(b.values()) if b else 0.0
    recall = matched / sum(a.values()) if a else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "cer": float(cer(ref_n, hyp_n)),
        "wer": float(wer(ref_n, hyp_n)),
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": f1,
    }


def transcribe_batch(
    paths: list[str],
    processor: Any,
    model: Any,
    device: str,
    model_dtype: torch.dtype,
    max_new_tokens: int,
    num_beams: int,
) -> list[tuple[str, float, int, bool]]:
    start = time.perf_counter()
    images = [Image.open(p).convert("RGB") for p in paths]
    prompts = ["<OCR>"] * len(images)
    inputs = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True,
        do_resize=True,
        resample=processor.image_processor.resample,
    )
    inputs = {
        k: v.to(device=device, dtype=model_dtype) if v.is_floating_point() else v.to(device)
        for k, v in inputs.items()
    }
    with torch.inference_mode():
        ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
    elapsed = time.perf_counter() - start
    per_sample_latency = elapsed / len(images)
    raw_texts = processor.batch_decode(ids, skip_special_tokens=False)
    pad_token = processor.tokenizer.pad_token or "<pad>"
    pad_token_id = processor.tokenizer.pad_token_id

    outputs: list[tuple[str, float, int, bool]] = []
    for seq_ids, raw, image in zip(ids, raw_texts, images):
        generated_tokens = (
            int((seq_ids != pad_token_id).sum().item())
            if pad_token_id is not None
            else int(seq_ids.shape[-1])
        )
        truncated = generated_tokens >= max_new_tokens
        parsed = processor.post_process_generation(
            raw.replace(pad_token, ""),
            task="<OCR>",
            image_size=image.size,
        )
        text = parsed.get("<OCR>", parsed) if isinstance(parsed, dict) else str(parsed)
        outputs.append((str(text).strip(), per_sample_latency, generated_tokens, truncated))
    return outputs


def build_summary(
    df: pd.DataFrame,
    args: argparse.Namespace,
    total_pairs: int,
) -> dict[str, Any]:
    ok_df = df[df["status"] == "ok"] if not df.empty else pd.DataFrame()
    err_df = df[df["status"] != "ok"] if not df.empty else pd.DataFrame()

    metric_cols = [
        "cer",
        "wer",
        "token_precision",
        "token_recall",
        "token_f1",
        "latency_sec",
        "generated_tokens",
    ]

    def _agg(sub: pd.DataFrame) -> dict[str, Any]:
        if sub.empty:
            return {"count": 0}
        res: dict[str, Any] = {"count": int(len(sub))}
        for col in metric_cols:
            if col in sub.columns:
                res[f"mean_{col}"] = float(sub[col].mean())
        if "latency_sec" in sub.columns:
            res["total_latency_sec"] = float(sub["latency_sec"].sum())
        return res

    by_category = (
        {str(cat): _agg(grp) for cat, grp in ok_df.groupby("category")}
        if not ok_df.empty
        else {}
    )
    by_occlusion = (
        {str(occ): _agg(grp) for occ, grp in ok_df.groupby("occlusion_type")}
        if not ok_df.empty
        else {}
    )

    return {
        "model_id": args.model_id,
        "data_dir": str(args.data_dir),
        "output_csv": str(args.output_csv),
        "config": {
            "batch_size": args.batch_size,
            "num_samples": args.num_samples,
            "save_every": args.save_every,
            "resume": args.resume,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "num_beams": args.num_beams,
            "normalize_case": not args.case_sensitive,
        },
        "counts": {
            "total_pairs": total_pairs,
            "evaluated": int(len(df)),
            "ok": int(len(ok_df)),
            "error": int(len(err_df)),
        },
        "overall": _agg(ok_df),
        "by_category": by_category,
        "by_occlusion_type": by_occlusion,
    }


def _build_row(
    sample: dict[str, str],
    model_id: str,
    ocr_text: str,
    latency: float,
    generated: int,
    truncated: bool,
    normalize_case: bool,
) -> dict[str, Any]:
    row: dict[str, Any] = {**sample, "model_id": model_id}
    try:
        gt = gt_text(sample["gt_path"])
        row.update(
            {
                "status": "ok",
                "ocr_text": ocr_text,
                "ground_truth": gt,
                "latency_sec": latency,
                "generated_tokens": generated,
                "possibly_truncated": truncated,
            }
        )
        row.update(compute_metrics(gt, ocr_text, normalize_case))
    except Exception as exc:
        row.update({"status": "error", "error": repr(exc)})
    return row


def main() -> None:
    args = parse_args()
    normalize_case = not args.case_sensitive

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    download_data(args.data_dir)
    pairs = collect_pairs(args.data_dir, args.num_samples, args.seed)
    pair_ids = {p["sample_id"] for p in pairs}

    rows: list[dict[str, Any]] = []
    finished_ids: set[str] = set()
    if args.resume and args.output_csv.exists():
        old_rows = pd.read_csv(args.output_csv).to_dict("records")
        rows = [
            r
            for r in old_rows
            if r.get("sample_id") in pair_ids and r.get("status") == "ok"
        ]
        finished_ids = {str(r["sample_id"]) for r in rows}
        print(f"[resume] Loaded {len(finished_ids)} completed samples from {args.output_csv}")

    pending = [p for p in pairs if p["sample_id"] not in finished_ids]
    print(
        f"[eval] Total pairs: {len(pairs)} | Pending: {len(pending)} | Batch size: {args.batch_size}"
    )

    if pending:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_dtype = torch.float16 if device == "cuda" else torch.float32
        print(f"[model] Loading {args.model_id} on {device} ({model_dtype})...")
        _apply_transformers_compat_shim()
        processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
        model = (
            AutoModelForCausalLM.from_pretrained(
                args.model_id,
                trust_remote_code=True,
                torch_dtype=model_dtype,
            )
            .to(device)
            .eval()
        )

        processed_since_save = 0
        done_samples = 0
        ok_count = sum(1 for r in rows if r.get("status") == "ok")
        err_count = sum(1 for r in rows if r.get("status") != "ok")
        cer_sum = sum(float(r["cer"]) for r in rows if r.get("status") == "ok" and "cer" in r)

        batches = [
            pending[i : i + args.batch_size]
            for i in range(0, len(pending), args.batch_size)
        ]
        pbar = tqdm(
            batches,
            desc="Evaluating",
            unit="batch",
            miniters=1,
            dynamic_ncols=True,
        )

        for batch in pbar:
            new_rows: list[dict[str, Any]] = []
            try:
                batch_outputs = transcribe_batch(
                    [s["image_path"] for s in batch],
                    processor=processor,
                    model=model,
                    device=device,
                    model_dtype=model_dtype,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                )
                for sample, (ocr_text, latency, generated, truncated) in zip(
                    batch, batch_outputs
                ):
                    new_rows.append(
                        _build_row(
                            sample,
                            args.model_id,
                            ocr_text,
                            latency,
                            generated,
                            truncated,
                            normalize_case,
                        )
                    )
            except Exception as batch_exc:
                tqdm.write(
                    f"[warn] Batch of {len(batch)} failed ({batch_exc!r}); falling back to single-sample inference."
                )
                if device == "cuda":
                    torch.cuda.empty_cache()
                # Fallback to single-sample inference if batch fails (e.g. OOM or bad image)
                for sample in batch:
                    try:
                        ocr_text, latency, generated, truncated = transcribe_batch(
                            [sample["image_path"]],
                            processor=processor,
                            model=model,
                            device=device,
                            model_dtype=model_dtype,
                            max_new_tokens=args.max_new_tokens,
                            num_beams=args.num_beams,
                        )[0]
                        new_rows.append(
                            _build_row(
                                sample,
                                args.model_id,
                                ocr_text,
                                latency,
                                generated,
                                truncated,
                                normalize_case,
                            )
                        )
                    except Exception as exc:
                        new_rows.append(
                            {**sample, "model_id": args.model_id, "status": "error", "error": repr(exc)}
                        )

            rows.extend(new_rows)
            for r in new_rows:
                if r.get("status") == "ok":
                    ok_count += 1
                    cer_sum += float(r["cer"])
                else:
                    err_count += 1

            done_samples += len(batch)
            processed_since_save += len(batch)
            pbar.set_postfix(
                imgs=f"{done_samples}/{len(pending)}",
                ok=ok_count,
                err=err_count,
                cer=f"{(cer_sum / ok_count):.3f}" if ok_count else "n/a",
            )
            if processed_since_save >= args.save_every:
                pd.DataFrame(rows).to_csv(args.output_csv, index=False)
                processed_since_save = 0

    results_df = pd.DataFrame(rows)
    results_df.to_csv(args.output_csv, index=False)
    summary = build_summary(results_df, args, total_pairs=len(pairs))
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[done] Saved per-image CSV to {args.output_csv}")
    print(f"[done] Saved summary JSON to {args.output_json}")
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
