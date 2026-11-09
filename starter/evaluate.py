from __future__ import annotations

import json
from pathlib import Path

from openai import OpenAI

from classify import API_KEY, MODEL, classify, cost_usd
from prompts import PROMPTS

ROOT = Path(__file__).resolve().parent.parent
LABELED = ROOT / "data" / "labeled_sample.json"
RESULTS_DIR = ROOT / "results"

# (name, base_url, api_key, model). base_url=None uses the OpenAI default endpoint.
MODELS = [
    ("cloud", None, API_KEY, MODEL),
    ("ollama-local", "http://localhost:11434/v1", "ollama", "llama3.2:3b"),
]


def exact_match(pred: set[str], gold: set[str]) -> bool:
    return pred == gold


def jaccard(pred: set[str], gold: set[str]) -> float:
    union = pred | gold
    if not union:
        return 1.0
    return len(pred & gold) / len(union)


def make_client(api_key: str | None, base_url: str | None) -> OpenAI | None:
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url=base_url, timeout=30, max_retries=1)


def probe(client: OpenAI, model: str) -> str | None:
    """Returns an error message if the model is unreachable, None if it's fine."""
    res = classify(client, "Nice hotel, friendly staff.", PROMPTS["v1"], model)
    return None if res["ok"] else res.get("error", "unknown error")


def run_combo(client: OpenAI, model: str, prompt_name: str, reviews: list[dict]) -> dict:
    prompt = PROMPTS[prompt_name]
    raw: list[dict] = []
    total_seconds = total_cost = 0.0
    valid = exact = jaccard_sum = 0

    for r in reviews:
        res = classify(client, r["text"], prompt, model)
        total_seconds += res["seconds"]
        gold = set(r["gold_categories"])

        entry = {"id": r["id"], "gold": sorted(gold), "seconds": res["seconds"]}
        if res["ok"]:
            pred = set(res["review"].categories)
            valid += 1
            total_cost += cost_usd(res["input_tokens"], res["output_tokens"], model)
            exact += exact_match(pred, gold)
            jaccard_sum += jaccard(pred, gold)
            entry["pred"] = sorted(pred)
        else:
            entry["error"] = res["error"]
        raw.append(entry)

    n = len(reviews)
    return {
        "model": model,
        "prompt": prompt_name,
        "n": n,
        "valid_json_rate": valid / n,
        "exact_match_rate": exact / n,
        "avg_jaccard": jaccard_sum / n,
        "avg_seconds": total_seconds / n,
        "total_cost_usd": total_cost,
        "raw": raw,
    }


def main() -> None:
    reviews = json.loads(LABELED.read_text(encoding="utf-8"))
    RESULTS_DIR.mkdir(exist_ok=True)

    summary = []
    for name, base_url, api_key, model in MODELS:
        client = make_client(api_key, base_url)
        if client is None:
            summary.append({"model": name, "skipped": True, "reason": "missing API key"})
            print(f"{name}: skipped (missing API key)")
            continue

        error = probe(client, model)
        if error is not None:
            summary.append({"model": name, "skipped": True, "reason": error})
            print(f"{name}: skipped ({error})")
            continue

        for prompt_name in PROMPTS:
            run = run_combo(client, model, prompt_name, reviews)
            (RESULTS_DIR / f"{name}_{prompt_name}.json").write_text(
                json.dumps(run["raw"], indent=2, ensure_ascii=False), encoding="utf-8"
            )
            metrics = {k: v for k, v in run.items() if k != "raw"}
            metrics["model_label"] = name
            summary.append(metrics)
            print(
                f"{name}/{prompt_name}: exact={metrics['exact_match_rate']:.2f} "
                f"jaccard={metrics['avg_jaccard']:.2f} valid={metrics['valid_json_rate']:.2f} "
                f"avg_s={metrics['avg_seconds']:.2f} cost=${metrics['total_cost_usd']:.6f}"
            )

    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved {RESULTS_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
