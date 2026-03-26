from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUPPORTED_DATASETS = {"bbh", "arc", "password", "mmlu"}


@dataclass
class PromptSpec:
    task_prompt: str = ""
    answer_format: str = ""
    generation_length: int = 128


class RegistryError(RuntimeError):
    pass


class ConfigError(RuntimeError):
    pass


def _load_json_or_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"任务注册表不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if path.endswith((".yaml", ".yml")):
        data = yaml.safe_load(content)
    else:
        data = json.loads(content)
    return data or {}


def _default_registry_path(dataset_name: str) -> str:
    return os.path.join(SCRIPT_DIR, f"registry_{dataset_name}.yaml")


def _split_task_names(task_names: Optional[str]) -> Optional[List[str]]:
    if not task_names:
        return None
    names = [x.strip() for x in task_names.split(",") if x.strip()]
    if not names or names == ["all"] or names == ["*"]:
        return None
    return names


def _default_config(dataset_name: str) -> Dict[str, Any]:
    if dataset_name == "bbh":
        return {
            "data_dir": os.path.join(SCRIPT_DIR, "dataset", "bbh"),
            "file_pattern": "*.json",
            "question_template": "Q: {input}\nA:",
            "default_prompt": {"task_prompt": "", "answer_format": "", "generation_length": 128},
        }
    if dataset_name == "arc":
        return {
            "data_dir": os.path.join(SCRIPT_DIR, "dataset", "arc_c"),
            "question_template": "{input}\nAnswer:",
            "eval_split_default": "test",
            "default_prompt": {"task_prompt": "", "answer_format": "", "generation_length": 128},
        }
    if dataset_name == "password":
        return {
            "data_dir": os.path.join(SCRIPT_DIR, "dataset", "password"),
            "file_pattern": "*.json",
            "question_template": "{input}",
            "default_prompt": {
                "task_prompt": "Given the user context, output the correct password.",
                "answer_format": "Answer with only the password.",
                "generation_length": 32,
            },
        }
    if dataset_name == "mmlu":
        return {
            "data_dir": os.path.join(SCRIPT_DIR, "dataset", "mmlu"),
            "question_template": "{input}",
            "eval_split_default": "test-00000-of-00001.json",
            "default_prompt": {
                "task_prompt": "The following are multiple choice questions (with options).",
                "answer_format": "Answer with only the corresponding letter (e.g. (A)).",
                "generation_length": 3,
            },
        }
    raise ConfigError(f"暂不支持的数据集: {dataset_name}")


def load_dataset_config(
    dataset_name: str,
    dataset_dir: Optional[str] = None,
    task_registry_path: Optional[str] = None,
) -> Dict[str, Any]:
    dataset_name = dataset_name.strip().lower()
    if dataset_name not in SUPPORTED_DATASETS:
        raise ConfigError(f"当前仅支持数据集: {sorted(SUPPORTED_DATASETS)}")

    registry_path = task_registry_path or _default_registry_path(dataset_name)
    registry_data = _load_json_or_yaml(registry_path)
    datasets = registry_data.get("datasets", registry_data)
    if not isinstance(datasets, dict):
        raise RegistryError("任务注册表格式错误: 顶层应为 datasets 字典")

    cfg = _default_config(dataset_name)
    cfg.update(dict(datasets.get(dataset_name, {})))

    if dataset_dir:
        cfg["data_dir"] = dataset_dir

    if not cfg.get("data_dir"):
        raise ConfigError(f"数据集 {dataset_name} 缺少 data_dir 配置")

    cfg["data_dir"] = os.path.abspath(os.path.join(SCRIPT_DIR, cfg["data_dir"]))
    if not os.path.isdir(cfg["data_dir"]):
        fallback_dir = _default_config(dataset_name)["data_dir"]
        fallback_dir = os.path.abspath(os.path.join(SCRIPT_DIR, fallback_dir))
        if os.path.isdir(fallback_dir):
            cfg["data_dir"] = fallback_dir
        else:
            raise ConfigError(f"数据集目录不存在: {cfg['data_dir']}")

    return cfg


def list_training_units(dataset_name: str, dataset_cfg: Dict[str, Any], task_names: Optional[str]) -> List[str]:
    requested = _split_task_names(task_names)
    if requested is not None:
        return requested

    if dataset_name == "arc":
        return [str(dataset_cfg.get("eval_split_default", "test"))]

    if dataset_name == "mmlu":
        subject_dirs = sorted(
            [
                name
                for name in os.listdir(dataset_cfg["data_dir"])
                if os.path.isdir(os.path.join(dataset_cfg["data_dir"], name))
            ]
        )
        if not subject_dirs:
            raise FileNotFoundError(f"未在 {dataset_cfg['data_dir']} 下找到 MMLU subject 目录")
        return subject_dirs

    # BBH 和 password 使用相同的文件扫描逻辑
    pattern = os.path.join(dataset_cfg["data_dir"], dataset_cfg.get("file_pattern", "*.json"))
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"未在 {pattern} 下找到任务文件")

    return [os.path.splitext(os.path.basename(path))[0] for path in files]


def list_eval_units(
    dataset_name: str,
    dataset_cfg: Dict[str, Any],
    selected_units: List[str],
    eval_scope: str,
    eval_split: str,
) -> List[str]:
    if dataset_name == "arc":
        return [eval_split or dataset_cfg.get("eval_split_default", "validation")]

    if dataset_name == "mmlu":
        # MMLU 按 subject 评测，每个 subject 内部固定使用 test split。
        return list(selected_units)

    if eval_scope == "all_dataset_tasks":
        return list_training_units(dataset_name, dataset_cfg, task_names=None)
    return list(selected_units)


def _load_bbh_examples(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both formats: list directly or dict with "examples" key
    if isinstance(data, list):
        raw_examples = data
    else:
        raw_examples = data.get("examples", [])
    if not isinstance(raw_examples, list):
        raise ConfigError(f"BBH 文件格式错误: {path}")

    examples: List[Dict[str, str]] = []
    for item in raw_examples:
        if not isinstance(item, dict) or "input" not in item or "target" not in item:
            raise ConfigError(f"BBH 样本缺少 input/target: {path}")
        examples.append({"input": str(item["input"]), "target": str(item["target"])})
    return examples


def _load_arc_jsonl(path: str) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "input" not in item or "target" not in item:
                raise ConfigError(f"ARC 样本缺少 input/target: {path}")
            # 保留 choices 和 answerKey 用于 acc_norm 计算
            example = {
                "input": str(item["input"]),
                "target": str(item["target"]),
            }
            if "choices" in item:
                example["choices"] = item["choices"]
            if "answerKey" in item:
                example["answerKey"] = item["answerKey"]
            examples.append(example)
    return examples


def _load_mmlu_subject_examples(subject_dir: str, split_file: str) -> List[Dict[str, str]]:
    file_path = os.path.join(subject_dir, split_file)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"MMLU split 文件不存在: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ConfigError(f"MMLU 文件格式错误（期望 list）: {file_path}")

    examples: List[Dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict) or "input" not in item or "target" not in item:
            raise ConfigError(f"MMLU 样本缺少 input/target: {file_path}")
        examples.append({"input": str(item["input"]), "target": str(item["target"])})
    return examples


def load_unit_examples(dataset_name: str, dataset_cfg: Dict[str, Any], unit_name: str) -> List[Dict[str, str]]:
    if dataset_name == "bbh":
        file_path = os.path.join(dataset_cfg["data_dir"], f"{unit_name}.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"BBH 任务文件不存在: {file_path}")
        return _load_bbh_examples(file_path)

    if dataset_name == "password":
        file_path = os.path.join(dataset_cfg["data_dir"], f"{unit_name}.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Password 任务文件不存在: {file_path}")
        return _load_bbh_examples(file_path)  # 格式兼容，直接复用

    if dataset_name == "mmlu":
        subject_dir = os.path.join(dataset_cfg["data_dir"], unit_name)
        if not os.path.isdir(subject_dir):
            raise FileNotFoundError(f"MMLU subject 目录不存在: {subject_dir}")
        split_file = str(dataset_cfg.get("eval_split_default", "test-00000-of-00001.json"))
        return _load_mmlu_subject_examples(subject_dir, split_file)

    file_path = os.path.join(dataset_cfg["data_dir"], f"{unit_name}.jsonl")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ARC split 文件不存在: {file_path}")
    return _load_arc_jsonl(file_path)


def load_training_examples(dataset_name: str, dataset_cfg: Dict[str, Any], train_unit: str) -> List[Dict[str, str]]:
    return load_unit_examples(dataset_name, dataset_cfg, train_unit)


def prepare_eval_examples(
    dataset_name: str,
    dataset_cfg: Dict[str, Any],
    eval_unit: str,
    train_size: int,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    if train_size <= 0:
        raise ConfigError("train_size 必须大于 0")

    # 统一 TTT-style split: support 与 eval 都来自同一 unit。
    # 例如 ARC test.jsonl: 前 train_size 条用于训练/示例，剩余用于评测。
    examples = load_unit_examples(dataset_name, dataset_cfg, eval_unit)
    if len(examples) <= train_size:
        raise ConfigError(
            f"{dataset_name} unit {eval_unit} 样本不足，train_size={train_size}，实际={len(examples)}"
        )
    return examples[:train_size], examples[train_size:]


def resolve_prompt_spec(dataset_name: str, dataset_cfg: Dict[str, Any], unit_name: str) -> PromptSpec:
    task_prompts = dataset_cfg.get("task_prompts", {})
    if dataset_name in {"bbh", "mmlu"} and unit_name in task_prompts:
        prompt_data = task_prompts[unit_name]
        return PromptSpec(
            task_prompt=str(prompt_data.get("task_prompt", "")),
            answer_format=str(prompt_data.get("answer_format", "")),
            generation_length=int(prompt_data.get("generation_length", 128)),
        )

    default_prompt = dataset_cfg.get("default_prompt", {})
    if default_prompt:
        return PromptSpec(
            task_prompt=str(default_prompt.get("task_prompt", "")),
            answer_format=str(default_prompt.get("answer_format", "")),
            generation_length=int(default_prompt.get("generation_length", 128)),
        )

    raise ConfigError(f"任务 {unit_name} 缺少 prompt 配置")


def build_instruction(prompt_spec: PromptSpec) -> str:
    return f"{prompt_spec.task_prompt} {prompt_spec.answer_format}".strip()


def format_query(question: str, dataset_cfg: Dict[str, Any]) -> str:
    template = dataset_cfg.get("question_template", "{input}")
    return template.format(input=question)


def build_fewshot_prompt_prefix(
    train_examples: Iterable[Dict[str, str]],
    dataset_cfg: Dict[str, Any],
    n_shot: int,
) -> str:
    if n_shot <= 0:
        return ""

    rendered: List[str] = []
    for ex in list(train_examples)[:n_shot]:
        query = format_query(ex["input"], dataset_cfg)
        rendered.append(f"{query} {ex['target']}".strip())

    intro = str(dataset_cfg.get("few_shot_intro", "")).strip()
    body = "\n\n".join(rendered).strip()
    if intro and body:
        return f"{intro}\n\n{body}"
    return intro or body


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def is_prediction_correct(pred: str, target: str) -> bool:
    return normalize_text(pred) == normalize_text(target)


def compute_accuracy(preds: List[str], targets: List[str]) -> float:
    if not targets:
        return 0.0
    correct = sum(1 for pred, target in zip(preds, targets) if is_prediction_correct(pred, target))
    return 100.0 * correct / len(targets)


def run_vllm_inference(
    llm: Any,
    questions: List[str],
    dataset_cfg: Dict[str, Any],
    prompt_spec: PromptSpec,
    max_new_tokens: int,
    few_shot_prompt_prefix: str = "",
    temperature: float = 0.0,
    top_p: float = 1.0,
    stop: Optional[List[str]] = None,
    lora_request: Optional[Any] = None,
) -> List[str]:
    from vllm import SamplingParams

    tokenizer = llm.get_tokenizer()
    instruction = build_instruction(prompt_spec)
    prompts: List[str] = []

    for question in questions:
        query = format_query(question, dataset_cfg)
        parts = [instruction, few_shot_prompt_prefix, query]
        user_prompt = "\n\n".join([part for part in parts if part]).strip()

        if hasattr(tokenizer, "apply_chat_template"):
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = user_prompt
        prompts.append(prompt_text)

    params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        stop=stop,
    )
    if lora_request is not None:
        outputs = llm.generate(prompts, params, lora_request=lora_request)
    else:
        outputs = llm.generate(prompts, params)

    preds: List[str] = []
    for out in outputs:
        if not out.outputs:
            preds.append("")
            continue
        preds.append(out.outputs[0].text.strip())
    return preds
