import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Dict, List

import yaml

from dataset_runtime import (
    ConfigError,
    RegistryError,
    build_instruction,
    format_query,
    list_eval_units,
    list_training_units,
    load_dataset_config,
    load_training_examples,
    resolve_prompt_spec,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TEMPLATE_YAML = os.path.join(SCRIPT_DIR, "train_config_template.yaml")
DEFAULT_RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
DEFAULT_LOGS_DIR = os.path.join(SCRIPT_DIR, "training_logs")
DEFAULT_WORKSPACE_DIR = os.path.join(SCRIPT_DIR, "workspace_fewshot")
DEFAULT_EVAL_WORKER = os.path.join(SCRIPT_DIR, "eval_worker_impl.py")


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    value = str(v).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {v}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified few-shot pipeline for BBH, ARC, and task-style datasets like Password."
    )
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3", help="Training GPUs, for example 0,1,2,3")
    parser.add_argument("--dataset", choices=["bbh", "arc", "password"], required=True, help="Dataset name")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Override dataset directory")
    parser.add_argument("--task_registry", type=str, default=None, help="Registry YAML/JSON, default registry_<dataset>.yaml")
    parser.add_argument("--task_names", type=str, default="", help="Task-style datasets only: comma-separated task list; empty means all")
    parser.add_argument("--eval_split", choices=["validation", "test"], default="test", help="ARC only")

    parser.add_argument("--model_path", type=str, required=True, help="Base model path")
    parser.add_argument("--template_yaml", type=str, default=DEFAULT_TEMPLATE_YAML, help="LlamaFactory training template")
    parser.add_argument("--eval_worker_script", type=str, default=DEFAULT_EVAL_WORKER, help="Evaluation worker script")

    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR, help="Results output directory")
    parser.add_argument("--logs_dir", type=str, default=DEFAULT_LOGS_DIR, help="Training logs directory")
    parser.add_argument("--workspace_dir", type=str, default=DEFAULT_WORKSPACE_DIR, help="Workspace directory")

    parser.add_argument("--few_shot_k", type=int, default=5, help="Few-shot inference example count")
    parser.add_argument("--train_size", type=int, default=None, help="Few-shot fine-tuning sample count; defaults to few_shot_k")
    parser.add_argument("--repeat_times", type=int, default=200, help="Repeat count for the constructed fine-tuning set")
    parser.add_argument("--epochs", type=float, default=3.0, help="Number of training epochs")

    parser.add_argument("--run_pre_ft_eval", type=str2bool, default=True, help="Run pre-fine-tuning evaluation")
    parser.add_argument("--run_post_ft_eval", type=str2bool, default=True, help="Run post-fine-tuning evaluation")
    parser.add_argument("--pre_eval_modes", type=str, default="few-shot", help="zero-shot / few-shot / both / off")
    parser.add_argument("--post_eval_modes", type=str, default="zero-shot", help="zero-shot / few-shot / both / off")
    parser.add_argument("--run_checkpoint_eval", type=str2bool, default=True, help="Evaluate intermediate checkpoints")

    parser.add_argument("--max_model_len", type=int, default=4096, help="vLLM max context length")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90, help="vLLM GPU memory utilization")
    parser.add_argument("--eval_batch_note", type=str, default="", help="Extra note written into results")

    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--cutoff_buffer", type=int, default=100)
    parser.add_argument("--min_cutoff_len", type=int, default=256)
    parser.add_argument("--save_training_logs", type=str2bool, default=True)
    parser.add_argument("--overwrite_output_dir", type=str2bool, default=True)
    parser.add_argument("--skip_training", type=str2bool, default=False, help="Only evaluate; skip fine-tuning")

    return parser.parse_args()


def normalize_modes(mode_value: str) -> List[str]:
    mode_value = (mode_value or "").strip().lower()
    if not mode_value or mode_value == "off":
        return []
    if mode_value == "both":
        return ["zero-shot", "few-shot"]
    if mode_value in {"zero-shot", "few-shot"}:
        return [mode_value]
    modes = [x.strip() for x in mode_value.split(",") if x.strip()]
    return [m for m in modes if m in {"zero-shot", "few-shot"}]


def calculate_dynamic_cutoff(tokenizer, texts: List[str], buffer: int, min_cutoff_len: int) -> int:
    max_len = 0
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        max_len = max(max_len, len(ids))
    return max(max_len + buffer, min_cutoff_len)


def sanitize_path_component(name: str) -> str:
    """将任意名称转换为适合目录名的安全片段。"""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "unknown"


def resolve_workspace_dir(base_workspace_dir: str, model_path: str, dataset: str) -> str:
    """基于根目录生成隔离 workspace：<base>/<model>/<dataset>。"""
    model_name = sanitize_path_component(os.path.basename(os.path.normpath(model_path)))
    dataset_name = sanitize_path_component(dataset)
    return os.path.join(os.path.abspath(base_workspace_dir), model_name, dataset_name)


def setup_workspace(work_dir: str, results_dir: str, logs_dir: str) -> Dict[str, str]:
    work_dir = os.path.abspath(work_dir)
    data_dir = os.path.join(work_dir, "data")
    output_dir = os.path.join(work_dir, "output")
    config_path = os.path.join(work_dir, "train_config.yaml")
    ds_config_path = os.path.join(work_dir, "ds_z2_config.json")

    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.abspath(results_dir), exist_ok=True)
    os.makedirs(os.path.abspath(logs_dir), exist_ok=True)

    ds_config = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
        },
        "bf16": {"enabled": "auto"},
        "fp16": {"enabled": "auto"},
    }
    with open(ds_config_path, "w", encoding="utf-8") as f:
        json.dump(ds_config, f, indent=2)

    return {
        "WORK_DIR": work_dir,
        "DATA_DIR": data_dir,
        "OUTPUT_DIR": output_dir,
        "CONFIG_PATH": config_path,
        "DS_CONFIG_PATH": ds_config_path,
    }


def checkpoint_sort_key(path: str) -> int:
    match = re.search(r"checkpoint-(\d+)$", os.path.basename(path))
    if not match:
        return 10**12
    return int(match.group(1))


def write_dataset_info(dataset_name: str, data_dir: str, data_file_name: str) -> None:
    info_content = {
        dataset_name: {
            "file_name": data_file_name,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        }
    }
    with open(os.path.join(data_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(info_content, f, indent=2, ensure_ascii=False)


def create_dataset_and_config(
    args: argparse.Namespace,
    dataset_cfg: Dict,
    train_unit: str,
    train_examples: List[Dict[str, str]],
    tokenizer,
    paths: Dict[str, str],
) -> int:
    prompt_spec = resolve_prompt_spec(args.dataset, dataset_cfg, train_unit)
    instruction = build_instruction(prompt_spec)

    train_data = []
    raw_texts: List[str] = []
    for ex in train_examples:
        input_text = format_query(ex["input"], dataset_cfg)
        target_text = ex["target"]
        raw_texts.append(f"{instruction}\n\n{input_text} {target_text}".strip())
        train_data.append(
            {
                "instruction": instruction,
                "input": input_text,
                "output": target_text,
            }
        )

    full_train_data = train_data * max(1, int(args.repeat_times))
    dataset_alias = f"train_{args.dataset}_{train_unit}"
    data_file_name = f"{dataset_alias}.json"
    data_file_path = os.path.join(paths["DATA_DIR"], data_file_name)
    with open(data_file_path, "w", encoding="utf-8") as f:
        json.dump(full_train_data, f, ensure_ascii=False, indent=2)
    write_dataset_info(dataset_alias, paths["DATA_DIR"], data_file_name)

    dynamic_len = calculate_dynamic_cutoff(
        tokenizer,
        raw_texts,
        buffer=int(args.cutoff_buffer),
        min_cutoff_len=int(args.min_cutoff_len),
    )

    with open(args.template_yaml, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["model_name_or_path"] = args.model_path
    config["dataset"] = dataset_alias
    config["dataset_dir"] = paths["DATA_DIR"]
    config["cutoff_len"] = dynamic_len
    config["output_dir"] = paths["OUTPUT_DIR"]
    config["stage"] = "sft"
    config["finetuning_type"] = "lora"
    config["lora_target"] = "all"
    config["lora_rank"] = int(args.lora_rank)
    config["lora_alpha"] = int(args.lora_alpha)
    config["lora_dropout"] = float(args.lora_dropout)
    config["learning_rate"] = float(args.learning_rate)
    config["num_train_epochs"] = float(args.epochs)
    config["per_device_train_batch_size"] = int(args.per_device_train_batch_size)
    config["gradient_accumulation_steps"] = int(args.gradient_accumulation_steps)
    config["overwrite_output_dir"] = bool(args.overwrite_output_dir)
    config["report_to"] = "tensorboard"
    config["logging_dir"] = os.path.join(paths["OUTPUT_DIR"], "runs")
    config["logging_steps"] = 5
    config["deepspeed"] = paths["DS_CONFIG_PATH"]

    if args.run_checkpoint_eval:
        config["save_strategy"] = "epoch"
        config["save_total_limit"] = max(1, int(args.epochs))
    else:
        config["save_strategy"] = "no"
        config.pop("save_total_limit", None)
        config.pop("save_steps", None)

    with open(paths["CONFIG_PATH"], "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False, allow_unicode=True)
    return dynamic_len


def train_model(args: argparse.Namespace, gpu_ids: str, main_gpu: str, paths: Dict[str, str]) -> bool:
    if os.path.exists(paths["OUTPUT_DIR"]):
        shutil.rmtree(paths["OUTPUT_DIR"])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env["MASTER_PORT"] = str(29500 + int(main_gpu) * 11)

    cmd = ["llamafactory-cli", "train", paths["CONFIG_PATH"]]
    try:
        print(f"[Train] Start training on GPUs={gpu_ids}")
        subprocess.run(cmd, check=True, env=env)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Training failed: {e}")
        return False


def merge_lora_adapter(model_path: str, main_gpu: str, adapter_path: str, output_dir: str) -> bool:
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = main_gpu

    cmd = [
        "llamafactory-cli",
        "export",
        "--model_name_or_path",
        model_path,
        "--adapter_name_or_path",
        adapter_path,
        "--export_dir",
        output_dir,
        "--export_size",
        "5",
    ]
    try:
        print(f"[Merge] {adapter_path} -> {output_dir}")
        subprocess.run(cmd, check=True, env=env)
        return True
    except subprocess.CalledProcessError as e:
        print(f"LoRA merge failed: {e}")
        return False


def save_training_logs(train_unit: str, logs_dir: str, output_dir: str) -> None:
    if not os.path.exists(output_dir):
        print(f"Skip log export because output directory does not exist: {output_dir}")
        return

    unit_log_dir = os.path.join(logs_dir, train_unit)
    os.makedirs(unit_log_dir, exist_ok=True)

    log_files = [
        "trainer_state.json",
        "training_loss.png",
        "all_results.json",
        "train_results.json",
        "training_args.bin",
    ]
    for filename in log_files:
        src_path = os.path.join(output_dir, filename)
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(unit_log_dir, filename))

    runs_dir = os.path.join(output_dir, "runs")
    if os.path.exists(runs_dir):
        for root, _, files in os.walk(runs_dir):
            for filename in files:
                if not filename.startswith("events.out.tfevents"):
                    continue
                src_path = os.path.join(root, filename)
                rel_path = os.path.relpath(root, runs_dir)
                dst_dir = unit_log_dir if rel_path == "." else os.path.join(unit_log_dir, "runs", rel_path)
                os.makedirs(dst_dir, exist_ok=True)
                shutil.copy2(src_path, os.path.join(dst_dir, filename))


def extract_model_name(model_path: str) -> str:
    """从模型路径中提取模型名称（最后一个目录名）"""
    return os.path.basename(os.path.normpath(model_path))


def build_result_file_path(
    results_root: str,
    dataset_name: str,
    train_unit: str,
    stage: str,
    eval_unit: str,
    eval_mode: str,
    n_shot: int,
    checkpoint_label: str,
    model_path: str,
) -> str:
    mode_tag = "zeroshot" if eval_mode == "zero-shot" else f"fewshot{n_shot}"
    filename = f"{eval_unit}__{stage}__{mode_tag}__{checkpoint_label}.json"
    model_name = extract_model_name(model_path)
    # 新路径结构: results/model_name/dataset_name/train_unit/stage/filename.json
    result_dir = os.path.join(results_root, model_name, dataset_name, train_unit, stage)
    os.makedirs(result_dir, exist_ok=True)
    return os.path.join(result_dir, filename)


def run_eval_worker(
    args: argparse.Namespace,
    dataset_cfg: Dict,
    train_unit: str,
    eval_unit: str,
    model_path: str,
    output_path: str,
    eval_mode: str,
    n_shot: int,
    train_size: int,
    main_gpu: str,
    stage: str,
    checkpoint_label: str,
) -> bool:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = main_gpu

    cmd = [
        sys.executable,
        args.eval_worker_script,
        "--dataset",
        args.dataset,
        "--dataset_dir",
        dataset_cfg["data_dir"],
        "--eval_unit",
        eval_unit,
        "--train_unit",
        train_unit,
        "--model_path",
        model_path,
        "--output_path",
        output_path,
        "--eval_mode",
        eval_mode,
        "--n_shot",
        str(n_shot),
        "--train_size",
        str(train_size),
        "--max_model_len",
        str(args.max_model_len),
        "--gpu_memory_utilization",
        str(args.gpu_memory_utilization),
        "--stage",
        stage,
        "--checkpoint_label",
        checkpoint_label,
    ]
    if args.task_registry:
        cmd.extend(["--task_registry", args.task_registry])
    if args.eval_batch_note:
        cmd.extend(["--note", args.eval_batch_note])

    try:
        subprocess.run(cmd, check=True, env=env)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed: eval_unit={eval_unit} stage={stage} mode={eval_mode} error={e}")
        return False


def summarize_results(results_root: str) -> None:
    summary_rows = []
    for path in sorted(glob.glob(os.path.join(results_root, "**", "*.json"), recursive=True)):
        if os.path.basename(path) in {"summary.json", "stage_summary.json", "train_task_summary.json"}:
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if not isinstance(data, dict) or "accuracy" not in data:
            continue
        summary_rows.append(
            {
                "dataset": data.get("dataset", ""),
                "train_task": data.get("train_task", ""),
                "eval_task": data.get("eval_task", data.get("task", "")),
                "stage": data.get("stage", ""),
                "checkpoint_label": data.get("checkpoint_label", ""),
                "eval_mode": data.get("eval_mode", ""),
                "n_shot": data.get("n_shot", 0),
                "accuracy": data.get("accuracy", 0.0),
                "num_eval_examples": data.get("num_eval_examples", 0),
                "file": os.path.relpath(path, results_root),
            }
        )

    summary_path = os.path.join(results_root, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)
    print(f"Summary written to: {summary_path}")


def main() -> None:
    args = parse_args()
    gpu_ids = args.gpu_ids
    gpu_list = [x.strip() for x in gpu_ids.split(",") if x.strip()]
    main_gpu = gpu_list[0]
    train_size = args.train_size if args.train_size is not None else args.few_shot_k
    if train_size <= 0:
        raise ValueError("train_size must be greater than 0")

    run_workspace_dir = resolve_workspace_dir(args.workspace_dir, args.model_path, args.dataset)
    run_logs_dir = resolve_workspace_dir(args.logs_dir, args.model_path, args.dataset)

    print(f"Training GPU(s): {gpu_ids}")
    print(f"Inference GPU: {main_gpu}")
    print(f"dataset={args.dataset} few_shot_k={args.few_shot_k} train_size={train_size} epochs={args.epochs}")
    print(f"workspace_dir={run_workspace_dir}")
    print(f"logs_dir={run_logs_dir}")

    dataset_cfg = load_dataset_config(args.dataset, args.dataset_dir, args.task_registry)
    selected_units = list_training_units(args.dataset, dataset_cfg, args.task_names)
    eval_units = list_eval_units(args.dataset, dataset_cfg, selected_units, "selected", args.eval_split)

    if args.dataset == "arc":
        selected_units = [args.eval_split]
        eval_units = [args.eval_split]

    paths = setup_workspace(run_workspace_dir, args.results_dir, run_logs_dir)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print(f"Training units: {selected_units}")
    print(f"Evaluation units: {eval_units}")

    pre_eval_modes = normalize_modes(args.pre_eval_modes) if args.run_pre_ft_eval else []
    post_eval_modes = normalize_modes(args.post_eval_modes) if args.run_post_ft_eval else []

    for unit_index, train_unit in enumerate(selected_units, start=1):
        print(f"\n{'=' * 12} unit [{unit_index}/{len(selected_units)}]: {train_unit} {'=' * 12}")

        # Task-style datasets evaluate the current task; ARC evaluates the selected split.
        current_eval_units = [train_unit] if args.dataset in {"bbh", "password"} else eval_units

        try:
            train_examples_all = load_training_examples(args.dataset, dataset_cfg, train_unit)
        except (FileNotFoundError, RegistryError, ConfigError) as e:
            print(f"Skip {train_unit}: failed to load training examples: {e}")
            continue

        if len(train_examples_all) < train_size:
            print(f"Skip {train_unit}: need at least {train_size} training examples, got {len(train_examples_all)}")
            continue

        train_examples = train_examples_all[:train_size]
        dynamic_len = create_dataset_and_config(args, dataset_cfg, train_unit, train_examples, tokenizer, paths)
        print(f"Prepared training data and config, cutoff_len={dynamic_len}")

        if pre_eval_modes:
            for eval_unit in current_eval_units:
                for eval_mode in pre_eval_modes:
                    n_shot = args.few_shot_k if eval_mode == "few-shot" else 0
                    out_path = build_result_file_path(
                        args.results_dir,
                        args.dataset,
                        train_unit,
                        "pre_ft",
                        eval_unit,
                        eval_mode,
                        n_shot,
                        "base",
                        args.model_path,
                    )
                    if os.path.exists(out_path):
                        print(f"Skip existing result: {out_path}")
                        continue
                    run_eval_worker(
                        args,
                        dataset_cfg,
                        train_unit,
                        eval_unit,
                        args.model_path,
                        out_path,
                        eval_mode,
                        n_shot,
                        train_size,
                        main_gpu,
                        "pre_ft",
                        "base",
                    )

        if args.skip_training:
            print("Skip training and run evaluation only")
            continue

        if not train_model(args, gpu_ids, main_gpu, paths):
            print(f"Training failed: {train_unit}")
            continue

        time.sleep(3)
        if args.save_training_logs:
            save_training_logs(train_unit, run_logs_dir, paths["OUTPUT_DIR"])

        if args.run_checkpoint_eval:
            checkpoint_dirs = sorted(
                glob.glob(os.path.join(paths["OUTPUT_DIR"], "checkpoint-*")),
                key=checkpoint_sort_key,
            )
            print(f"{train_unit} checkpoints found: {len(checkpoint_dirs)}")
            for ckpt_idx, ckpt_path in enumerate(checkpoint_dirs, start=1):
                checkpoint_label = f"epoch{ckpt_idx}"
                merged_dir_epoch = os.path.join(paths["WORK_DIR"], f"merged_{train_unit}_{checkpoint_label}")
                if not merge_lora_adapter(args.model_path, main_gpu, ckpt_path, merged_dir_epoch):
                    continue
                try:
                    for eval_unit in current_eval_units:
                        for eval_mode in post_eval_modes:
                            n_shot = args.few_shot_k if eval_mode == "few-shot" else 0
                            out_path = build_result_file_path(
                                args.results_dir,
                                args.dataset,
                                train_unit,
                                "checkpoint",
                                eval_unit,
                                eval_mode,
                                n_shot,
                                checkpoint_label,
                                args.model_path,
                            )
                            if os.path.exists(out_path):
                                print(f"Skip existing result: {out_path}")
                                continue
                            run_eval_worker(
                                args,
                                dataset_cfg,
                                train_unit,
                                eval_unit,
                                merged_dir_epoch,
                                out_path,
                                eval_mode,
                                n_shot,
                                train_size,
                                main_gpu,
                                "checkpoint",
                                checkpoint_label,
                            )
                finally:
                    if os.path.exists(merged_dir_epoch):
                        shutil.rmtree(merged_dir_epoch)
                time.sleep(2)
        # 独立的 final eval，确保即使执行了 checkpoint eval 也会执行
        if post_eval_modes:
            final_merged_dir = os.path.join(paths["WORK_DIR"], f"merged_{train_unit}_final")
            if merge_lora_adapter(args.model_path, main_gpu, paths["OUTPUT_DIR"], final_merged_dir):
                try:
                    for eval_unit in current_eval_units:
                        for eval_mode in post_eval_modes:
                            n_shot = args.few_shot_k if eval_mode == "few-shot" else 0
                            out_path = build_result_file_path(
                                args.results_dir,
                                args.dataset,
                                train_unit,
                                "post_ft",
                                eval_unit,
                                eval_mode,
                                n_shot,
                                "final",
                                args.model_path,
                            )
                            if os.path.exists(out_path):
                                print(f"Skip existing result: {out_path}")
                                continue
                            run_eval_worker(
                                args,
                                dataset_cfg,
                                train_unit,
                                eval_unit,
                                final_merged_dir,
                                out_path,
                                eval_mode,
                                n_shot,
                                train_size,
                                main_gpu,
                                "post_ft",
                                "final",
                            )
                finally:
                    if os.path.exists(final_merged_dir):
                        shutil.rmtree(final_merged_dir)

    summarize_results(os.path.abspath(args.results_dir))


if __name__ == "__main__":
    main()
