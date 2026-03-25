import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset_runtime import (  # noqa: E402
    ConfigError,
    RegistryError,
    build_fewshot_prompt_prefix,
    compute_accuracy,
    format_query,
    is_prediction_correct,
    load_dataset_config,
    prepare_eval_examples,
    resolve_prompt_spec,
    run_vllm_inference,
)


LOGPROB_FAIL = -1e9


def _to_float_logprob(value: Any) -> float | None:
    """兼容不同 vLLM 版本的 logprob 返回结构。"""
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    if hasattr(value, "logprob"):
        try:
            return float(value.logprob)
        except Exception:
            return None
    if isinstance(value, dict) and "logprob" in value:
        try:
            return float(value["logprob"])
        except Exception:
            return None
    return None


def compute_option_logprob(
    llm: Any,
    tokenizer: Any,
    prompt_text: str,
    option_text: str,
) -> float:
    """
    计算在给定 prompt 后生成 option_text 的 log 概率。

    使用 vLLM 的 prompt_logprobs 功能来估计条件概率。
    """
    from vllm import SamplingParams

    # 拼接 prompt 和 option
    full_text = f"{prompt_text}{option_text}"

    # 编码文本以获取 token 数量
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_tokens = tokenizer.encode(full_text, add_special_tokens=False)

    # 如果 option 为空或者 token 数量相同，返回一个非常低的概率
    if len(full_tokens) <= len(prompt_tokens):
        return LOGPROB_FAIL

    # 使用 prompt_logprobs 来获取每个 token 的 logprob
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=len(full_tokens),
    )

    try:
        outputs = llm.generate([full_text], sampling_params)
        if not outputs or not outputs[0].prompt_logprobs:
            return LOGPROB_FAIL

        # 计算 option 部分的累积 logprob
        # prompt_logprobs[i] 表示第 i 个 token 的 logprob（基于前面的 tokens）
        option_logprobs = []
        logprobs_list = outputs[0].prompt_logprobs

        for i in range(len(prompt_tokens), len(full_tokens)):
            if i < len(logprobs_list) and logprobs_list[i]:
                # 找到当前 token 的 logprob
                token_id = full_tokens[i]
                token_entry = logprobs_list[i].get(token_id)
                token_logprob = _to_float_logprob(token_entry)
                if token_logprob is None:
                    # token 不在映射时给一个较低但有限的分数，避免整题塌缩到同分。
                    token_logprob = -50.0
                option_logprobs.append(token_logprob)

        if option_logprobs:
            # 返回平均 logprob（归一化长度）
            return sum(option_logprobs) / len(option_logprobs)
        return LOGPROB_FAIL
    except Exception:
        return LOGPROB_FAIL


def normalize_option_label(value: Any) -> str:
    """归一化选项标签，例如 '(A)'/'a'/' A ' -> 'A'。"""
    if value is None:
        return ""
    text = str(value).strip()
    # 优先匹配括号中的 token，如 (A)
    m = re.search(r"\(([A-Za-z0-9]+)\)", text)
    if m:
        return m.group(1).upper()
    # 退化到第一个字母数字 token
    m = re.search(r"[A-Za-z0-9]+", text)
    if m:
        return m.group(0).upper()
    return text.upper()


def build_label_candidates(label: str) -> List[str]:
    """构造用于打分的候选输出形式，覆盖常见解码变体。"""
    normalized = normalize_option_label(label)
    if not normalized:
        return []
    candidates = [
        f"({normalized})",
        normalized,
        f" ({normalized})",
        f" {normalized}",
    ]
    # 保持顺序去重
    seen = set()
    dedup = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            dedup.append(c)
    return dedup


def compute_acc_norm_arc(
    llm: Any,
    dataset_cfg: Dict[str, Any],
    prompt_spec: Any,
    eval_examples: List[Dict[str, Any]],
    few_shot_prompt_prefix: str = "",
) -> tuple[float, List[Dict[str, Any]]]:
    """
    使用 acc_norm (概率归一化) 计算 ARC 多选题的准确率。

    对每个问题，计算每个选项 (A/B/C/D) 的条件概率，选择概率最高的作为预测。

    Returns:
        (accuracy_percent, details_list)
    """
    tokenizer = llm.get_tokenizer()
    instruction = f"{prompt_spec.task_prompt} {prompt_spec.answer_format}".strip()

    details = []

    for ex in eval_examples:
        question = ex["input"]
        answer_key = normalize_option_label(ex.get("answerKey", ex.get("target", "")))
        choices = ex.get("choices", [])

        if not choices:
            # 如果没有选项，使用原始方法
            details.append({
                "input": question,
                "target": answer_key,
                "prediction": "",
                "is_correct": False,
                "choices": [],
            })
            continue

        # 构建基础 prompt (不含选项)
        query = format_query(question, dataset_cfg)
        parts = [instruction, few_shot_prompt_prefix, query]
        user_prompt = "\n\n".join([p for p in parts if p]).strip()

        if hasattr(tokenizer, "apply_chat_template"):
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = user_prompt

        # 计算每个选项的条件概率（acc_norm 核心）
        # 主路径: 选项文本；回退: 仅当文本打分失败时使用标签候选。
        choice_scores = []
        for choice in choices:
            label = normalize_option_label(choice.get("label", ""))
            text = str(choice.get("text", "") or "")

            best_logprob = LOGPROB_FAIL
            score_source = "text"

            # 文本主评分：优先尝试原文本，同时尝试一个前导空格变体以兼容常见分词。
            text_candidates = []
            if text.strip():
                text_candidates.append(text)
                if not text.startswith(" "):
                    text_candidates.append(f" {text}")

            if text_candidates:
                text_scores = [
                    compute_option_logprob(llm, tokenizer, prompt_text, cand)
                    for cand in text_candidates
                ]
                best_logprob = max(text_scores)

            # 仅在文本打分失败时回退到标签候选。
            if best_logprob <= LOGPROB_FAIL:
                label_candidates = build_label_candidates(label)
                if label_candidates:
                    label_scores = [
                        compute_option_logprob(llm, tokenizer, prompt_text, cand)
                        for cand in label_candidates
                    ]
                    best_logprob = max(label_scores)
                    score_source = "label_fallback"

            choice_scores.append(
                {
                    "label": label,
                    "text": text,
                    "logprob": best_logprob,
                    "score_source": score_source,
                }
            )

        # 如果所有选项都退化到失败分，说明 prompt_logprobs 链路失效。
        if choice_scores and all(c["logprob"] <= LOGPROB_FAIL for c in choice_scores):
            print("[Worker][WARN] ARC acc_norm 打分失败：当前样本所有选项 logprob 均为兜底值。")

        # 选择概率最高的选项
        if choice_scores:
            pred_label = normalize_option_label(max(choice_scores, key=lambda x: x["logprob"])["label"])
        else:
            pred_label = ""

        is_correct = normalize_option_label(pred_label) == answer_key

        details.append({
            "input": question,
            "target": answer_key,
            "prediction": pred_label,
            "is_correct": is_correct,
            "choices": choice_scores,
        })

    if not details:
        return 0.0, details

    correct = sum(1 for d in details if d["is_correct"])
    acc_percent = 100.0 * correct / len(details)
    return acc_percent, details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vLLM 评测 worker，支持 BBH/ARC/MMLU/Password 的 zero-shot 与 few-shot")
    parser.add_argument("--dataset", choices=["bbh", "arc", "password", "mmlu"], required=True, help="数据集名称")
    parser.add_argument("--dataset_dir", type=str, default=None, help="数据集目录覆盖")
    parser.add_argument("--task_registry", type=str, default=None, help="任务注册表 YAML/JSON")
    parser.add_argument("--eval_unit", type=str, required=True, help="评测单元：BBH/Password/MMLU task 名或 ARC split 名")
    parser.add_argument("--train_unit", type=str, default="", help="训练单元名称，仅用于结果记录")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--eval_mode", choices=["zero-shot", "few-shot"], default="zero-shot")
    parser.add_argument("--n_shot", type=int, default=0, help="few-shot 样本数")
    parser.add_argument("--train_size", type=int, default=5, help="支持样本池大小")
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--stage", type=str, default="eval")
    parser.add_argument("--checkpoint_label", type=str, default="final")
    parser.add_argument("--note", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        dataset_cfg = load_dataset_config(args.dataset, args.dataset_dir, args.task_registry)
        prompt_spec = resolve_prompt_spec(args.dataset, dataset_cfg, args.eval_unit)
        support_examples, eval_examples = prepare_eval_examples(
            args.dataset,
            dataset_cfg,
            args.eval_unit,
            args.train_size,
        )
    except (FileNotFoundError, RegistryError, ConfigError) as e:
        print(f"评测准备失败: {e}")
        sys.exit(1)

    effective_n_shot = 0
    few_shot_prompt_prefix = ""
    if args.eval_mode == "few-shot":
        effective_n_shot = min(args.n_shot, len(support_examples))
        few_shot_prompt_prefix = build_fewshot_prompt_prefix(
            support_examples,
            dataset_cfg,
            effective_n_shot,
        )

    eval_questions = [ex["input"] for ex in eval_examples]
    eval_targets = [ex["target"] for ex in eval_examples]

    print(
        f"[Worker] model={args.model_path} dataset={args.dataset} eval_unit={args.eval_unit} "
        f"mode={args.eval_mode} n_shot={effective_n_shot} eval_examples={len(eval_questions)}"
    )

    try:
        from vllm import LLM

        llm = LLM(
            model=args.model_path,
            trust_remote_code=True,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )
    except Exception as e:
        print(f"vLLM 初始化失败: {e}")
        sys.exit(1)

    # 全数据集统一使用标准 acc 文本匹配评测。
    preds = run_vllm_inference(
        llm=llm,
        questions=eval_questions,
        dataset_cfg=dataset_cfg,
        prompt_spec=prompt_spec,
        max_new_tokens=prompt_spec.generation_length,
        few_shot_prompt_prefix=few_shot_prompt_prefix,
    )

    acc_percent = compute_accuracy(preds, eval_targets)
    acc = acc_percent / 100.0
    print(f"Accuracy: {acc_percent:.2f}%")

    details = []
    for inp, tgt, pred in zip(eval_questions, eval_targets, preds):
        details.append(
            {
                "input": inp,
                "target": tgt,
                "prediction": pred,
                "is_correct": is_prediction_correct(pred, tgt),
            }
        )

    # Extract model name from path (e.g., /path/to/Qwen2.5-1.5B-Instruct -> Qwen2.5-1.5B-Instruct)
    model_name = os.path.basename(os.path.normpath(args.model_path))

    result_data = {
        "dataset": args.dataset,
        "task": args.eval_unit,
        "eval_task": args.eval_unit,
        "train_task": args.train_unit,
        "stage": args.stage,
        "checkpoint_label": args.checkpoint_label,
        "eval_mode": args.eval_mode,
        "n_shot": effective_n_shot,
        "train_size": args.train_size,
        "accuracy": acc,
        "num_eval_examples": len(eval_questions),
        "model_name": model_name,
        "model_path": args.model_path,
        "note": args.note,
        "details": details,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
