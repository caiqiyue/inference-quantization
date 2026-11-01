import json
from pathlib import Path

import torch
from datasets import load_dataset


def get_calib_dataset(data="pileval", tokenizer=None, n_samples=512, block_size=512):
    if data == "pileval":
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    elif Path(data).is_file():
        rows = []
        with Path(data).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                text = item.get("prompt") if isinstance(item, dict) else item
                if isinstance(text, str) and text.strip():
                    rows.append({"text": text})
        dataset = rows
    else:
        raise FileNotFoundError(f"calibration dataset does not exist: {data}")
    if hasattr(dataset, "shuffle"):
        dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > block_size:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    if n_split == 0:
        raise ValueError(
            f"calibration dataset produced fewer than {block_size} tokens; "
            "provide more calibration text"
        )
    return [
        cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
    ]
