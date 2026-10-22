"""Seeded request traces and Llumnix benchmark request generation."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class TraceRequest:
    request_id: str
    scheduled_arrival_sec: float
    request_type: str
    prompt: str
    input_tokens: int
    output_tokens: int


def load_local_tokenizer(path: str | None):
    if not path:
        return None
    local_path = Path(path)
    try:
        from tokenizers import Tokenizer

        tokenizer_json = local_path / "tokenizer.json" if local_path.is_dir() else local_path
        if tokenizer_json.is_file() and tokenizer_json.name == "tokenizer.json":
            return Tokenizer.from_file(str(tokenizer_json))
    except Exception:
        pass
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    except Exception:
        return None


def _encode(tokenizer, text: str) -> list[int]:
    if tokenizer is None:
        return text.split()
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return encoded.ids if hasattr(encoded, "ids") else encoded


def _decode(tokenizer, tokens) -> str:
    if tokenizer is None:
        return " ".join(tokens)
    return tokenizer.decode(tokens, skip_special_tokens=True)


def _truncate(text: str, max_tokens: int, tokenizer) -> tuple[str, int]:
    encoded = _encode(tokenizer, text)
    if len(encoded) > max_tokens:
        encoded = encoded[:max_tokens]
        text = _decode(tokenizer, encoded)
        encoded = _encode(tokenizer, text)
    return text, len(encoded)


def _iter_sharegpt(path: Path, limit: int) -> Iterable[str]:
    """Read the first user turn from ShareGPT JSON/JSONL or Parquet files."""
    files = [path] if path.is_file() else sorted(
        p for p in path.rglob("*") if p.suffix.lower() in {".jsonl", ".json", ".parquet"}
    )
    if path.is_dir() and len(files) > 1:
        preferred = [p for p in files if "sharegpt" in str(p).lower()]
        if preferred:
            files = preferred
    yielded = 0
    for file_path in files:
        try:
            if file_path.suffix.lower() == ".parquet":
                import pyarrow.parquet as pq

                table = pq.read_table(file_path, columns=["conversations"])
                rows = table.to_pylist()
            elif file_path.suffix.lower() == ".jsonl":
                rows = []
                with file_path.open(encoding="utf-8") as stream:
                    for line in stream:
                        if line.strip():
                            rows.append(json.loads(line))
                        if yielded + len(rows) >= limit:
                            break
            else:
                data = json.loads(file_path.read_text(encoding="utf-8"))
                rows = data if isinstance(data, list) else data.get("data", [data])
        except Exception:
            continue
        for row in rows:
            conversations = row.get("conversations", []) if isinstance(row, dict) else []
            if conversations and isinstance(conversations[0], dict):
                prompt = conversations[0].get("value") or conversations[0].get("content")
                if isinstance(prompt, str) and prompt.strip():
                    yield prompt.strip()
                    yielded += 1
                    if yielded >= limit:
                        return


def _make_prompt(tokenizer, target_tokens: int, seed: int) -> tuple[str, int]:
    if tokenizer is None:
        text = ("reliable service experiment " * (target_tokens // 3 + 2)).strip()
        words = text.split()[:target_tokens]
        return " ".join(words), len(words)
    vocab = tokenizer.get_vocab()
    vocab_size = tokenizer.get_vocab_size() if hasattr(tokenizer, "get_vocab_size") else len(tokenizer)
    token_ids = list(vocab.values())
    rng = random.Random(seed)
    safe_ids = [x for x in token_ids if 0 <= x < vocab_size]
    if not safe_ids:
        return _truncate("Summarize this short service experiment request.", target_tokens, tokenizer)
    ids = [rng.choice(safe_ids) for _ in range(target_tokens)]
    text = tokenizer.decode(ids, skip_special_tokens=True)
    return _truncate(text, target_tokens, tokenizer)


def _rate_changes(events: list[dict], base_qps: float):
    return sorted(
        (float(event["at_sec"]), float(event["value"]))
        for event in events
        if event.get("type") == "workload" and event.get("action") == "set_qps"
    )


def generate_trace(cfg: dict, tokenizer=None) -> list[TraceRequest]:
    exp, work, limits = cfg["experiment"], cfg["workload"], cfg["limits"]
    rng = random.Random(int(exp["seed"]))
    duration = float(exp["duration_sec"])
    max_requests = int(limits["max_requests"])
    output_default = int(work["output_tokens"])
    model_limit = int(work["max_model_len"])
    profile_mix = work.get("profile_mix")
    profiles = profile_mix or [{"name": "default", "input_tokens": int(work["input_tokens"]), "output_tokens": output_default, "weight": 1}]
    weights = [float(p.get("weight", 1)) for p in profiles]
    profile_cycle = bool(work.get("profile_cycle", False))
    profile_cycle_order = []
    configured_prompt = work.get("prompt")
    configured_prompt_cache: dict[tuple[str, int], tuple[str, int]] = {}

    dataset_prompts: list[str] = []
    dataset = work.get("dataset")
    if dataset and Path(dataset).exists():
        dataset_prompts = list(_iter_sharegpt(Path(dataset), min(max_requests * 5, 5000)))

    qps_changes = _rate_changes(cfg.get("events", []), float(work["base_qps"]))
    qps_index, qps = 0, float(work["base_qps"])
    t, trace = 0.0, []
    for index in range(max_requests):
        if profile_cycle:
            if index % len(profiles) == 0:
                profile_cycle_order = list(profiles)
                rng.shuffle(profile_cycle_order)
            profile = profile_cycle_order[index % len(profiles)]
        else:
            profile = rng.choices(profiles, weights=weights, k=1)[0]
        output_tokens = min(int(profile.get("output_tokens", output_default)), model_limit - 1)
        input_budget = min(int(profile.get("input_tokens", work["input_tokens"])), model_limit - output_tokens)
        if dataset_prompts:
            prompt = dataset_prompts[rng.randrange(len(dataset_prompts))]
            prompt, input_tokens = _truncate(prompt, input_budget, tokenizer)
            request_type = str(profile.get("name", "sharegpt"))
        elif configured_prompt:
            cache_key = (str(configured_prompt), input_budget)
            if cache_key not in configured_prompt_cache:
                configured_prompt_cache[cache_key] = _truncate(
                    str(configured_prompt), input_budget, tokenizer
                )
            prompt, input_tokens = configured_prompt_cache[cache_key]
            request_type = str(profile.get("name", "configured_prompt"))
        else:
            prompt, input_tokens = _make_prompt(tokenizer, input_budget, int(exp["seed"]) + index)
            request_type = str(profile.get("name", "default"))
        if input_tokens + output_tokens > model_limit:
            prompt, input_tokens = _truncate(prompt, model_limit - output_tokens, tokenizer)

        if index:
            if work["arrival"] == "burst":
                wait = 0.0
            elif work["arrival"] == "fixed":
                wait = 1.0 / qps
            else:
                wait = rng.expovariate(qps)
            next_change = qps_changes[qps_index] if qps_index < len(qps_changes) else None
            while next_change and t + wait >= next_change[0] and next_change[0] < duration:
                t = next_change[0]
                qps = next_change[1]
                qps_index += 1
                next_change = qps_changes[qps_index] if qps_index < len(qps_changes) else None
                wait = 1.0 / qps if work["arrival"] == "fixed" else rng.expovariate(qps)
            t += wait
        if t >= duration:
            break
        trace.append(TraceRequest(
            request_id=f"r{index + 1:05d}",
            scheduled_arrival_sec=round(t, 6),
            request_type=request_type,
            prompt=prompt,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ))
    return trace


def write_trace(path: Path, trace: list[TraceRequest]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(trace[0]).keys()) if trace else [
            "request_id", "scheduled_arrival_sec", "request_type", "prompt", "input_tokens", "output_tokens"
        ])
        writer.writeheader()
        for item in trace:
            writer.writerow(asdict(item))
