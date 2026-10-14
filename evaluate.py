"""Answer judging, process scoring and offline aggregation for PhysElite."""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import AsyncOpenAI

from data import (
    RESULTS_DIR, add_api_arguments, add_dataset_arguments, append_jsonl,
    completed_ids, create_client, ensure_parent, load_problems, positive_int,
    read_jsonl, write_summary,
)


JUDGE_PROMPT_EN = """You are an expert physics solution judge. Given a student's response and the reference materials for a physics problem, determine whether the student's final answer is correct.

## Reference Answer
{reference}

## Reference Solution
{solution}

## Student's Response
{response}

## Judging Protocol
Correctness is decided on the student's final result(s). When the reference answer is empty, extract the correct result from the reference solution.

1. Numerical answers are accepted within a 5% relative tolerance, unless the problem requires an exact integer or rational value.
2. Symbolic and algebraic answers are accepted in any mathematically equivalent form (e.g., reordered variables, with or without simplification).
3. For multi-part problems, return "correct" only if every sub-part is correct; if some sub-parts are correct and others are not, return "partially_correct".
4. Disregard differences in units notation, formatting, and intermediate steps. Only the final result is evaluated.

Return a single JSON object (no markdown fences) in exactly this schema:
{{"verdict": "correct" | "partially_correct" | "incorrect", "reasoning": "<one- to three-sentence explanation>"}}"""


JUDGE_PROMPT_CN = """你是一位物理判分专家。给定学生回答和该物理题目的参考材料，请判断学生的最终答案是否正确。

## 参考答案
{reference}

## 参考解答
{solution}

## 学生回答
{response}

## 评判规则
正误判定以学生的最终结果为依据。若参考答案为空，请从参考解答中提取正确结果。

1. 数值类答案在 5% 相对误差范围内视为正确，除非题目明确要求精确的整数或有理数。
2. 符号或代数类答案，任何数学等价形式（如变量顺序不同、化简与否等）均视为正确。
3. 多小问题目，所有小问均正确才返回 "correct"；部分正确则返回 "partially_correct"。
4. 忽略单位写法、排版格式、推导步骤的差异，仅评判最终结果。

仅返回一个 JSON 对象（不要使用 markdown 代码块），格式严格如下：
{{"verdict": "correct" | "partially_correct" | "incorrect", "reasoning": "<一至三句说明>"}}"""


def parse_verdict(text: str) -> tuple[str, str]:
    """Parse judge response JSON. Returns (verdict, reasoning). Falls back gracefully."""
    if not text:
        return "parse_error", "empty response"

    # Strip markdown fences if present
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Try direct JSON parse
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            return "parse_error", "expected a JSON object"
        v = obj.get("verdict", "parse_error")
        r = obj.get("reasoning", "")
        if v not in ("correct", "partially_correct", "incorrect"):
            return "parse_error", f"bad verdict: {v}"
        return v, r
    except json.JSONDecodeError:
        pass

    # Try extracting a JSON object from text
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if not isinstance(obj, dict):
                return "parse_error", "expected a JSON object"
            v = obj.get("verdict", "parse_error")
            r = obj.get("reasoning", "")
            if v in ("correct", "partially_correct", "incorrect"):
                return v, r
        except json.JSONDecodeError:
            pass

    # Salvage verdict from truncated JSON (long reasoning can hit max_tokens)
    m = re.search(r'"verdict"\s*:\s*"(correct|partially_correct|incorrect)"', text)
    if m:
        r = re.search(r'"reasoning"\s*:\s*"(.*)', text, re.DOTALL)
        reasoning = r.group(1).rstrip('"}').strip() + " [truncated]" if r else "(reasoning truncated)"
        return m.group(1), reasoning

    return "parse_error", text[:200]


async def judge_answer(
    client: AsyncOpenAI,
    entry: dict,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
    output_file: str,
    file_lock: asyncio.Lock,
    progress: dict,
):
    async with semaphore:
        pid = entry.get("problem_id", "unknown")
        reference = entry.get("problem_answer", "") or ""
        response = entry.get("response", "") or ""
        language = entry.get("language", "en")

        # Look up reference solution from the dataset
        problem = args._problem_map.get(str(pid), {})
        reference = reference or problem.get("problem_answer", "") or ""
        solution = problem.get(f"problem_solution_{language}", "") or ""

        template = JUDGE_PROMPT_CN if language == "cn" else JUDGE_PROMPT_EN
        prompt = template.format(reference=reference or "(not provided)",
                                 solution=solution or "(not provided)",
                                 response=response)

        t0 = time.time()
        verdict = "skipped"
        reasoning = ""
        prompt_tokens = 0
        completion_tokens = 0

        if not response:
            # No student response to judge — count as incorrect
            verdict = "no_response"
            reasoning = "student response is empty/null"
        else:
            try:
                resp = await client.chat.completions.create(
                    model=args.judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=args.max_tokens,
                    temperature=0.0,
                    seed=args.seed,
                )
                judge_text = resp.choices[0].message.content
                verdict, reasoning = parse_verdict(judge_text)
                if resp.usage:
                    prompt_tokens = resp.usage.prompt_tokens
                    completion_tokens = resp.usage.completion_tokens
            except Exception as e:
                verdict = "error"
                reasoning = f"{type(e).__name__}: {str(e)[:200]}"
                progress["errors"] += 1

        elapsed = round(time.time() - t0, 2)

        result = {
            "problem_id": pid,
            "model": entry.get("model", ""),
            "language": language,
            "verdict": verdict,
            "reasoning": reasoning,
            "problem_answer": reference,
            "response": response,
            "difficulty": entry.get("difficulty", ""),
            "problem_tags": entry.get("problem_tags", []),
            "collection": entry.get("collection", ""),
            "judge_time": elapsed,
            "judge_prompt_tokens": prompt_tokens,
            "judge_completion_tokens": completion_tokens,
        }

        await append_jsonl(output_file, result, file_lock)

        progress["done"] += 1
        progress["counts"][verdict] += 1
        done = progress["done"]
        total = progress["total"]
        c = progress["counts"]
        print(f"  [{done}/{total}] {pid}: {verdict}"
              f"  (✓{c['correct']} ~{c['partially_correct']} ✗{c['incorrect']})")


def summarize_answers(judged_file: str) -> dict:
    """Aggregate judged.jsonl into summary statistics."""
    counts = defaultdict(int)
    by_difficulty = defaultdict(lambda: defaultdict(int))
    by_tag = defaultdict(lambda: defaultdict(int))
    by_collection = defaultdict(lambda: defaultdict(int))
    total = 0

    with open(judged_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            v = d.get("verdict", "unknown")
            total += 1
            counts[v] += 1

            diff = d.get("difficulty") or "unknown"
            by_difficulty[diff]["total"] += 1
            by_difficulty[diff][v] += 1

            for tag in d.get("problem_tags") or []:
                by_tag[tag]["total"] += 1
                by_tag[tag][v] += 1

            coll = d.get("collection") or "unknown"
            by_collection[coll]["total"] += 1
            by_collection[coll][v] += 1

    def acc(d, key_set=("correct",)):
        t = d.get("total", 0)
        if t == 0:
            return 0.0
        return round(sum(d.get(k, 0) for k in key_set) / t, 4)

    summary = {
        "total": total,
        "counts": dict(counts),
        "accuracy": acc({"total": total, **counts}),
        "accuracy_with_partial": acc({"total": total, **counts}, ("correct", "partially_correct")),
        "by_difficulty": {
            k: {
                **dict(v),
                "accuracy": acc(v),
                "accuracy_with_partial": acc(v, ("correct", "partially_correct")),
            }
            for k, v in sorted(by_difficulty.items())
        },
        "by_tag": {
            k: {
                **dict(v),
                "accuracy": acc(v),
                "accuracy_with_partial": acc(v, ("correct", "partially_correct")),
            }
            for k, v in sorted(by_tag.items())
        },
        "by_collection": {
            k: {
                **dict(v),
                "accuracy": acc(v),
                "accuracy_with_partial": acc(v, ("correct", "partially_correct")),
            }
            for k, v in sorted(by_collection.items())
        },
    }
    return summary


STEP_PROMPT_EN = """You are an expert physics solution judge. Given a student's response and the reference solution to a physics problem, identify the key derivation steps in the reference solution and grade the student's work on each step.

## Problem
{problem}

## Reference Solution
{solution}

## Student's Response
{response}

## Judging Protocol
1. Identify between three and six **key derivation steps** from the reference solution. Each step should represent a distinct logical or computational milestone, such as setting up a governing equation, applying a conservation law, deriving a closed-form quantity, or performing a unit conversion.
2. Grade the student's work on each step on a three-point scale:
   - 1.0 — the step is completed with correct reasoning and a correct result.
   - 0.5 — the approach is essentially correct, but execution slips (e.g., a sign error, a missing factor, or a correct result reached via flawed reasoning).
   - 0.0 — the step is wrong or missing.
3. Disregard differences in notation, formatting, and ordering. Evaluate whether the student accomplished the goal of each step.

Return a single JSON object (no markdown fences) in exactly this schema:
{{"steps": [{{"name": "<short step name>", "score": 1.0 | 0.5 | 0.0, "comment": "<brief justification>"}}, ...]}}"""


STEP_PROMPT_CN = """你是一位物理判分专家。给定学生回答与该物理题目的参考解答，请识别参考解答中的关键推导步骤，并逐步对学生的表现进行评分。

## 题目
{problem}

## 参考解答
{solution}

## 学生回答
{response}

## 评判规则
1. 从参考解答中识别 **3 至 6 个关键推导步骤**，每个步骤对应一个独立的逻辑或计算环节，例如建立支配方程、应用守恒律、推出某量的解析形式、进行单位换算等。
2. 对学生在每一步的表现按三档制评分：
   - 1.0：步骤完成，推理与结果均正确。
   - 0.5：思路基本正确但执行存在失误（如符号错误、漏因子，或结果对而推理有误）。
   - 0.0：步骤错误或缺失。
3. 忽略记号、排版格式、步骤顺序的差异，仅评判学生是否达成了该步骤的目标。

仅返回一个 JSON 对象（不要使用 markdown 代码块），格式严格如下：
{{"steps": [{{"name": "<步骤简称>", "score": 1.0 | 0.5 | 0.0, "comment": "<简述>"}}, ...]}}"""


VERDICT_TO_SCORE = {
    "correct": 1.0,
    "partially_correct": 0.5,
    "incorrect": 0.0,
    "no_response": 0.0,
    "error": 0.0,
    "parse_error": 0.0,
    "skipped": 0.0,
}


def parse_steps(text: str) -> list[dict]:
    """Parse judge response. Returns list of {name, score, comment}."""
    if not text:
        return []
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return []
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []

    if not isinstance(obj, dict) or not isinstance(obj.get("steps"), list):
        return []
    steps = obj.get("steps", [])
    cleaned = []
    for s in steps:
        if not isinstance(s, dict):
            continue
        try:
            score = float(s.get("score", 0))
            if not math.isfinite(score):
                score = 0.0
            score = max(0.0, min(1.0, score))
        except (TypeError, ValueError):
            score = 0.0
        cleaned.append({
            "name": str(s.get("name", ""))[:200],
            "score": score,
            "comment": str(s.get("comment", ""))[:500],
        })
    return cleaned


async def judge_process(
    client: AsyncOpenAI,
    entry: dict,
    verdict: str,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
    output_file: str,
    file_lock: asyncio.Lock,
    progress: dict,
):
    async with semaphore:
        pid = entry.get("problem_id", "unknown")
        response = entry.get("response", "") or ""
        language = entry.get("language", "en")

        problem = args._problem_map.get(str(pid), {})
        problem_text = problem.get(f"problem_text_{language}", "") or ""
        solution = problem.get(f"problem_solution_{language}", "") or ""

        answer_score = VERDICT_TO_SCORE.get(verdict, 0.0)

        t0 = time.time()
        steps = []
        step_raw = ""
        err = ""
        prompt_tokens = 0
        completion_tokens = 0

        if not response or not solution:
            # No response or no reference solution — skip step judging
            step_raw = "skipped (missing response or solution)"
        else:
            template = STEP_PROMPT_CN if language == "cn" else STEP_PROMPT_EN
            # Hard caps on prompt fields to keep the request under the judge's
            # context window. The limits are well above the 99th-percentile
            # length on PhysElite, so truncation should be a no-op in practice;
            # if it does fire, a warning is emitted to stderr.
            P_MAX, S_MAX, R_MAX = 12000, 20000, 20000
            if len(problem_text) > P_MAX or len(solution) > S_MAX or len(response) > R_MAX:
                import sys as _sys
                print(
                    f"  [WARN] truncating fields for pid={pid}: "
                    f"problem={len(problem_text)}>{P_MAX}? solution={len(solution)}>{S_MAX}? "
                    f"response={len(response)}>{R_MAX}?",
                    file=_sys.stderr,
                )
            prompt = template.format(
                problem=problem_text[:P_MAX],
                solution=solution[:S_MAX],
                response=response[:R_MAX],
            )
            try:
                resp = await client.chat.completions.create(
                    model=args.judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=args.max_tokens,
                    temperature=0.0,
                    seed=args.seed,
                )
                step_raw = resp.choices[0].message.content or ""
                steps = parse_steps(step_raw)
                if resp.usage:
                    prompt_tokens = resp.usage.prompt_tokens
                    completion_tokens = resp.usage.completion_tokens
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:200]}"
                progress["errors"] += 1

        elapsed = round(time.time() - t0, 2)

        # Compute step_score
        if steps:
            step_score = sum(s["score"] for s in steps) / len(steps)
        else:
            step_score = 0.0

        total_score = args.answer_weight * answer_score + args.step_weight * step_score

        result = {
            "problem_id": pid,
            "model": entry.get("model", ""),
            "language": language,
            "verdict": verdict,
            "answer_score": round(answer_score, 4),
            "step_score": round(step_score, 4),
            "total_score": round(total_score, 4),
            "num_steps": len(steps),
            "steps": steps,
            "error": err,
            "difficulty": entry.get("difficulty", ""),
            "problem_tags": entry.get("problem_tags", []),
            "collection": entry.get("collection", ""),
            "judge_time": elapsed,
            "judge_prompt_tokens": prompt_tokens,
            "judge_completion_tokens": completion_tokens,
        }

        await append_jsonl(output_file, result, file_lock)

        progress["done"] += 1
        done = progress["done"]
        total = progress["total"]
        print(f"  [{done}/{total}] {pid}: answer={answer_score:.1f}, "
              f"step={step_score:.2f} ({len(steps)}st), total={total_score:.2f}")


def summarize_process(judged_file: str) -> dict:
    """Aggregate process-judged file into summary statistics."""
    totals = []
    answers = []
    steps = []
    by_difficulty = defaultdict(list)
    by_tag = defaultdict(list)
    by_collection = defaultdict(list)

    with open(judged_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            totals.append(d["total_score"])
            answers.append(d["answer_score"])
            steps.append(d["step_score"])

            diff = d.get("difficulty") or "unknown"
            by_difficulty[diff].append(d["total_score"])
            for tag in d.get("problem_tags") or []:
                by_tag[tag].append(d["total_score"])
            coll = d.get("collection") or "unknown"
            by_collection[coll].append(d["total_score"])

    def mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    return {
        "total": len(totals),
        "mean_total_score": mean(totals),
        "mean_answer_score": mean(answers),
        "mean_step_score": mean(steps),
        "percent_score": round(mean(totals) * 100, 2),
        "by_difficulty": {k: {"n": len(v), "mean": mean(v)} for k, v in sorted(by_difficulty.items())},
        "by_tag": {k: {"n": len(v), "mean": mean(v)} for k, v in sorted(by_tag.items())},
        "by_collection": {k: {"n": len(v), "mean": mean(v)} for k, v in sorted(by_collection.items())},
    }


async def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["answer", "process", "summary"])
    parser.add_argument("--input", required=True, help="Inference JSONL, or scored JSONL for summary")
    parser.add_argument("--output", default=None)
    parser.add_argument("--summary", default=None, help="Summary JSON path after judging")
    parser.add_argument("--kind", choices=["answer", "process"], default="answer", help="Offline summary kind")
    parser.add_argument("--judged-input", default=None, help="Answer judgments required for process scoring")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--max-tokens", type=positive_int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--answer-weight", type=float, default=0.3)
    parser.add_argument("--step-weight", type=float, default=0.7)
    add_api_arguments(parser)
    add_dataset_arguments(parser)
    args = parser.parse_args(argv)
    input_path = Path(args.input)
    if not input_path.is_file():
        parser.error(f"Input file not found: {args.input}")
    if args.stage == "summary":
        if not args.output:
            parser.error("summary requires --output")
        if Path(args.output).resolve() == input_path.resolve():
            parser.error("--output must differ from --input")
        summarize = summarize_answers if args.kind == "answer" else summarize_process
        result = summarize(args.input)
        write_summary(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if not args.judge_model:
        parser.error("judging requires --judge-model")
    if args.stage == "process" and not args.judged_input:
        parser.error("process scoring requires --judged-input")
    args.max_tokens = args.max_tokens or (1024 if args.stage == "answer" else 4096)
    folder, suffix = ("judged", "judged") if args.stage == "answer" else ("process_judged", "proc")
    args.output = args.output or str(RESULTS_DIR / folder / f"{input_path.stem}__{suffix}.jsonl")
    args.summary = args.summary or str(Path(args.output).with_suffix(".summary.json"))
    if Path(args.output).resolve() == input_path.resolve():
        parser.error("--output must differ from --input")
    entries = read_jsonl(args.input)
    done = completed_ids(args.output)
    remaining = [entry for entry in entries if str(entry.get("problem_id")) not in done]
    verdicts = {}
    if args.stage == "process":
        verdicts = {str(entry["problem_id"]): entry["verdict"] for entry in read_jsonl(args.judged_input)}
        if any(str(entry.get("problem_id")) not in verdicts for entry in remaining):
            parser.error("Some input IDs have no answer judgment in --judged-input")
    if remaining:
        args._problem_map = {str(problem["problem_id"]): problem
                             for problem in load_problems(args, include_images=False)}
        if any(str(entry.get("problem_id")) not in args._problem_map for entry in remaining):
            parser.error("Some input IDs have no reference in the selected Hub dataset")
    ensure_parent(args.output)
    Path(args.output).touch(exist_ok=True)
    if remaining:
        client = create_client(args, timeout=120.0)
        semaphore, lock = asyncio.Semaphore(args.max_concurrent), asyncio.Lock()
        progress = {"done": 0, "total": len(remaining), "errors": 0, "counts": defaultdict(int)}
        try:
            if args.stage == "answer":
                tasks = [judge_answer(client, entry, args, semaphore, args.output, lock, progress)
                         for entry in remaining]
            else:
                tasks = [judge_process(client, entry, verdicts[str(entry["problem_id"])], args,
                                       semaphore, args.output, lock, progress) for entry in remaining]
            await asyncio.gather(*tasks)
        finally:
            await client.close()
    summarize = summarize_answers if args.stage == "answer" else summarize_process
    result = summarize(args.output)
    write_summary(args.summary, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Results: {args.output}\nSummary: {args.summary}")


if __name__ == "__main__":
    asyncio.run(main())
