import torch


def _summarize_tensor(t: torch.Tensor) -> dict:
    t = t.to(torch.float32)
    total = t.numel()
    nan_count = torch.isnan(t).sum().item()
    posinf_count = torch.isposinf(t).sum().item()
    neginf_count = torch.isneginf(t).sum().item()
    finite_mask = torch.isfinite(t)
    finite_vals = t[finite_mask]

    stats = {
        "total": int(total),
        "nan": int(nan_count),
        "posinf": int(posinf_count),
        "neginf": int(neginf_count),
    }
    if finite_vals.numel() > 0:
        stats.update(
            {
                "min": float(finite_vals.min().item()),
                "max": float(finite_vals.max().item()),
                "mean": float(finite_vals.mean().item()),
                "p95": float(torch.quantile(finite_vals, torch.tensor(0.95)).item()),
                "p99": float(torch.quantile(finite_vals, torch.tensor(0.99)).item()),
            }
        )
    return stats


def analyze_data_quality(dataloader, max_batches: int = 2):
    keys = ["history", "target"]
    aggregated = {k: None for k in keys}
    batches_checked = 0

    for batch in dataloader:
        batches_checked += 1
        for k in keys:
            if k not in batch:
                continue
            t = batch[k]
            # flatten batch dims
            t_flat = t.reshape(-1)
            stats = _summarize_tensor(t_flat)
            if aggregated[k] is None:
                aggregated[k] = stats
            else:
                # merge by summing counts and taking min/max extremes
                for field in ["total", "nan", "posinf", "neginf"]:
                    aggregated[k][field] += stats[field]
                if "min" in stats:
                    aggregated[k]["min"] = min(aggregated[k]["min"], stats["min"])
                    aggregated[k]["max"] = max(aggregated[k]["max"], stats["max"])
                    # simple running average not precise; keep last mean as indicative
                    aggregated[k]["mean"] = stats["mean"]
                    aggregated[k]["p95"] = stats["p95"]
                    aggregated[k]["p99"] = stats["p99"]
        if batches_checked >= max_batches:
            break

    print(f"Checked batches: {batches_checked}")
    for k, v in aggregated.items():
        if v is None:
            continue
        print(f"[{k}] stats: {v}")
