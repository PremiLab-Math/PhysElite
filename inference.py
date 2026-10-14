"""PhysElite Benchmark: send problems + diagrams to LLM and save responses."""
from __future__ import annotations

import argparse
import asyncio
import time

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import AsyncOpenAI
from data import image_urls as get_image_urls
from data import (
    RESULTS_DIR, add_api_arguments, add_dataset_arguments, append_jsonl,
    completed_ids, create_client, ensure_parent, load_problems, model_filename,
    positive_int, read_subset,
)


SYSTEM_PROMPT_EN = (
    "You are an expert in physics. Solve the following Olympiad-level physics problem step by step. "
    "Show the full derivation, including the governing equations, simplifications, and any "
    "approximations you invoke. Conclude your response with a line that begins with "
    '"FINAL ANSWER:" and contains only the final result. '
)
SYSTEM_PROMPT_CN = (
    "你是一位物理专家。请逐步解答以下奥赛级物理题，"
    "完整给出推导过程，包括所用的支配方程、化简步骤以及所采用的近似条件。"
    '解答末尾另起一行，以"最终答案："开头给出最终结果，且该行仅包含最终答案。'
)


def build_messages(problem: dict, language: str, no_images: bool = False) -> list[dict]:
    text_key = f"problem_text_{language}"
    text = problem.get(text_key, problem.get("problem_text_en", ""))
    system_prompt = SYSTEM_PROMPT_CN if language == "cn" else SYSTEM_PROMPT_EN

    # Build user content: text + images (optional)
    content = [{"type": "text", "text": text}]
    if not no_images:
        for image_url in get_image_urls(problem):
            content.append({
                "type": "image_url",
                "image_url": {"url": image_url},
            })

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


async def run_one(
    client: AsyncOpenAI,
    problem: dict,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
    output_file: str,
    file_lock: asyncio.Lock,
    progress: dict,
):
    async with semaphore:
        pid = problem.get("problem_id", "unknown")
        messages = build_messages(problem, args.language, no_images=args.no_images)
        t0 = time.time()
        prompt_tokens = 0
        completion_tokens = 0
        try:
            if args.stream:
                stream = await client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    seed=args.seed,
                    stream=True,
                    stream_options={"include_usage": True},
                )
                chunks = []
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        chunks.append(chunk.choices[0].delta.content)
                    if chunk.usage:
                        prompt_tokens = chunk.usage.prompt_tokens
                        completion_tokens = chunk.usage.completion_tokens
                answer = "".join(chunks)
            else:
                resp = await client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    seed=args.seed,
                )
                answer = resp.choices[0].message.content
                if resp.usage:
                    prompt_tokens = resp.usage.prompt_tokens
                    completion_tokens = resp.usage.completion_tokens
        except Exception as e:
            answer = None
            progress["failed"] += 1
            print(f"  [FAIL] {pid}: {e}")
        elapsed = round(time.time() - t0, 2)

        result = {
            "problem_id": pid,
            "model": args.model,
            "language": args.language,
            "response": answer,
            "problem_answer": problem.get("problem_answer", ""),
            "difficulty": problem.get("difficulty", ""),
            "problem_tags": problem.get("problem_tags", []),
            "collection": problem.get("_collection", ""),
            "time_seconds": elapsed,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

        await append_jsonl(output_file, result, file_lock)

        progress["done"] += 1
        total = progress["total"]
        done = progress["done"]
        failed = progress["failed"]
        print(f"  [{done}/{total}] {pid} {'OK' if answer else 'FAIL'}", end="")
        if failed:
            print(f" (failed so far: {failed})", end="")
        print()


async def main(argv=None):
    parser = argparse.ArgumentParser(description="PhysElite Benchmark Inference")
    add_dataset_arguments(parser)
    parser.add_argument("--subset", default=None, help="One public problem ID per line")
    parser.add_argument("--limit", type=positive_int, default=None, help="Run only the first N selected problems")
    parser.add_argument("--language", default="en", choices=["cn", "en"])
    parser.add_argument("--model", default="gpt-4o")
    add_api_arguments(parser, concurrency=10)
    parser.add_argument("--output", default=None, help="Output JSONL path")
    parser.add_argument("--max-tokens", type=positive_int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-images", action="store_true", help="Text-only mode: skip sending images to model")
    parser.add_argument("--stream", action="store_true", help="Use streaming API (required for some models like qwq)")
    args = parser.parse_args(argv)

    # Default output path
    if args.output is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        suffix = "_textonly" if args.no_images else ""
        args.output = f"{RESULTS_DIR}/{model_filename(args.model)}_physelite_{args.language}{suffix}.jsonl"

    # Load problems
    print("Loading problems...")
    problems = load_problems(args, language=args.language, include_images=not args.no_images)
    if args.subset:
        keep = set(read_subset(args.subset))
        missing = keep - {str(p["problem_id"]) for p in problems}
        if missing:
            parser.error(f"{len(missing)} subset IDs missing from selected language/dataset")
        problems = [p for p in problems if str(p["problem_id"]) in keep]
    if args.limit:
        problems = problems[:args.limit]
    print(f"Loaded {len(problems)} problems")

    # Resume: skip already-done problem_ids
    done_ids = completed_ids(args.output)
    if done_ids:
        print(f"Resuming: {len(done_ids)} already completed, skipping")

    remaining = [p for p in problems if str(p.get("problem_id")) not in done_ids]
    if not remaining:
        print("All problems already completed!")
        return

    print(f"Running inference on {len(remaining)} problems with model={args.model}, language={args.language}")

    # Setup client (5 min request timeout to avoid hanging)
    client = create_client(args, timeout=300.0)

    semaphore = asyncio.Semaphore(args.max_concurrent)
    file_lock = asyncio.Lock()
    progress = {"done": 0, "failed": 0, "total": len(remaining)}

    ensure_parent(args.output)
    t0 = time.time()
    try:
        tasks = [
            run_one(client, p, args, semaphore, args.output, file_lock, progress)
            for p in remaining
        ]
        await asyncio.gather(*tasks)
    finally:
        await client.close()
    elapsed = time.time() - t0

    print(f"\nDone! {progress['done']}/{progress['total']} completed, "
          f"{progress['failed']} failed, {elapsed:.1f}s elapsed")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
