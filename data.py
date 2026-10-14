"""Hugging Face data, image encoding, API configuration and JSONL helpers."""
from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
HF_DATASET = "physelite/PhysElite"


def add_dataset_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--dataset", default=HF_DATASET, help="Hugging Face dataset ID")
    parser.add_argument("--split", default="default", help="Hub split (PhysElite uses default)")
    parser.add_argument("--config", default=None, help="Optional Hub configuration")
    parser.add_argument("--revision", default=None, help="Hub commit or tag for reproducibility")
    parser.add_argument("--cache-dir", default=None, help="Hugging Face download cache")


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def add_api_arguments(parser: argparse.ArgumentParser, concurrency: int = 10):
    parser.add_argument("--api-base", default=os.getenv("OPENAI_BASE_URL"),
                        help="OpenAI-compatible URL (or OPENAI_BASE_URL)")
    parser.add_argument("--api-key", default=None,
                        help="API key (or OPENAI_API_KEY)")
    parser.add_argument("--max-concurrent", type=positive_int, default=concurrency)


def create_client(args: argparse.Namespace, timeout: float):
    # Keep --help, data inspection and offline summaries usable without the SDK.
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError("Install evaluation dependencies: pip install -r requirements.txt") from exc
    kwargs = {"timeout": timeout, "max_retries": 2}
    if args.api_base:
        kwargs["base_url"] = args.api_base
    if args.api_key:
        kwargs["api_key"] = args.api_key
    return AsyncOpenAI(**kwargs)


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path, tolerate_invalid: bool = False) -> list[dict]:
    if tolerate_invalid and not Path(path).exists():
        return []
    records = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("expected a JSON object")
            except (ValueError, TypeError) as exc:
                if tolerate_invalid:
                    continue
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            records.append(record)
    return records


def completed_ids(path) -> set[str]:
    """Preserve resume semantics: every written problem ID counts as completed."""
    return {str(record["problem_id"]) for record in read_jsonl(path, tolerate_invalid=True)
            if record.get("problem_id") is not None}


def read_subset(path) -> list[str]:
    with open(path, encoding="utf-8") as handle:
        return list(dict.fromkeys(line.strip() for line in handle if line.strip()))


async def append_jsonl(path, record: dict, lock):
    async with lock:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_summary(path, summary: dict):
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def model_filename(model: str) -> str:
    return model.replace("/", "__").replace("\\", "__")


REQUIRED_COLUMNS = {"problem_id", "language", "problem_text", "problem_solution",
                    "problem_answer", "difficulty", "problem_tags", "images", "solution_images"}


def normalize_row(row: dict, dataset=HF_DATASET) -> dict:
    language = row.get("language")
    if language not in ("cn", "en"):
        raise ValueError(f"Unknown language for problem {row.get('problem_id')}: {language!r}")
    if row.get("problem_id") is None:
        raise ValueError("Missing problem_id in Hub row")
    return {
        "problem_id": row["problem_id"],
        "language": language,
        "problem_text": row.get("problem_text") or "",
        "problem_solution": row.get("problem_solution") or "",
        f"problem_text_{language}": row.get("problem_text") or "",
        f"problem_solution_{language}": row.get("problem_solution") or "",
        "problem_answer": row.get("problem_answer") or "",
        "difficulty": str(row.get("difficulty") or ""),
        "problem_tags": row.get("problem_tags") or [],
        "img_descriptions": row.get("img_descriptions") or [],
        "_collection": dataset,
        "_hf_images": row.get("images") or [],
        "_hf_solution_images": row.get("solution_images") or [],
    }


def load_hf_problems(dataset=HF_DATASET, *, split="default", config=None,
                     revision=None, cache_dir=None, language=None, include_images=True):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install Hub dependencies: pip install -r requirements.txt") from exc
    kwargs = {"split": split}
    if config:
        kwargs["name"] = config
    if revision:
        kwargs["revision"] = revision
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    ds = load_dataset(dataset, **kwargs)
    missing = REQUIRED_COLUMNS - set(ds.column_names)
    if missing:
        raise ValueError(f"{dataset}/{split} is missing columns: {', '.join(sorted(missing))}")
    # Iterate metadata separately so loading all problems does not decode images
    # or retain the entire image dataset in Python memory. Arrow remains on disk.
    image_dataset = ds.with_format("arrow") if include_images else None
    metadata = ds.remove_columns(["images", "solution_images"])
    problems = []
    for index, row in enumerate(metadata):
        if language and row["language"] != language:
            continue
        problem = normalize_row(row, dataset)
        if include_images:
            problem["_hf_dataset"] = image_dataset
            problem["_hf_row_index"] = index
        problems.append(problem)
    return problems


def image_data_url(value) -> str:
    """Encode HF bytes/path dictionaries or decoded PIL images for the API."""
    mime = None
    if isinstance(value, dict):
        data = value.get("bytes")
        path = value.get("path")
        if data is None and path:
            data = Path(path).read_bytes()
        if path:
            mime = mimetypes.guess_type(path)[0]
        if data is None:
            raise ValueError("Image has neither embedded bytes nor a readable path")
    elif isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
    elif hasattr(value, "save"):
        buffer = io.BytesIO()
        value.save(buffer, format="PNG")
        data, mime = buffer.getvalue(), "image/png"
    else:
        raise TypeError(f"Unsupported Hub image value: {type(value).__name__}")
    # Embedded Image values commonly have path=None; derive MIME from bytes.
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif data.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif data.startswith((b"GIF87a", b"GIF89a")):
        mime = "image/gif"
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        mime = "image/webp"
    return f"data:{mime or 'image/png'};base64,{base64.b64encode(data).decode('ascii')}"


def image_urls(problem: dict, solution=False) -> list[str]:
    if "_hf_dataset" in problem:
        table = problem["_hf_dataset"][problem["_hf_row_index"]]
        values = table.column("solution_images" if solution else "images")[0].as_py()
    else:
        values = problem.get("_hf_solution_images" if solution else "_hf_images", [])
    return [image_data_url(value) for value in values or [] if value is not None]


def load_problems(args, language=None, include_images=True):
    return load_hf_problems(args.dataset, split=args.split, config=args.config,
                            revision=args.revision, cache_dir=args.cache_dir,
                            language=language, include_images=include_images)
