import argparse
import csv
import json
import os
import shutil
from collections import defaultdict
from statistics import mean
from typing import Dict, List


def parse_args():
    parser = argparse.ArgumentParser(description="汇总 few-shot 全流程评测结果")
    parser.add_argument("--results_dir", type=str, required=True, help="结果目录，支持递归读取")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录。为空时按 results_dir/summary/<model_name>/<dataset>/acc_analysis_<mode> 自动分组输出",
    )
    parser.add_argument("--mode", choices=["auto", "bbh", "arc", "password", "mmlu"], default="auto", help="统计模式")
    parser.add_argument("--stage_filter", type=str, default="", help="仅统计指定 stage，逗号分隔")
    parser.add_argument("--train_task_filter", type=str, default="", help="仅统计指定 train_task，逗号分隔")
    parser.add_argument("--eval_task_filter", type=str, default="", help="仅统计指定 eval_task，逗号分隔")
    parser.add_argument("--model_name_filter", type=str, default="", help="仅统计指定 model_name，逗号分隔")
    return parser.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def sanitize_path_component(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(name).strip())
    safe = safe.strip("._-")
    return safe or "unknown"


def split_filter(value: str):
    return {x.strip() for x in value.split(",") if x.strip()}


def load_accuracy(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "accuracy" in data:
        return data
    raise ValueError(f"文件缺少 accuracy 字段: {file_path}")


def collect_records(results_dir: str):
    records = []
    ignored_names = {"summary.json", "stage_summary.json", "train_task_summary.json"}
    for root, _, files in os.walk(results_dir):
        for name in sorted(files):
            if not name.endswith(".json") or name in ignored_names:
                continue

            file_path = os.path.join(root, name)
            try:
                data = load_accuracy(file_path)
            except Exception as e:
                print(f"跳过异常文件 {file_path}: {e}")
                continue

            records.append(
                {
                    "model_name": data.get("model_name", ""),
                    "model_path": data.get("model_path", ""),
                    "dataset": data.get("dataset", ""),
                    "train_task": data.get("train_task", data.get("task", "")),
                    "eval_task": data.get("eval_task", data.get("task", "")),
                    "stage": data.get("stage", ""),
                    "checkpoint_label": data.get("checkpoint_label", ""),
                    "eval_mode": data.get("eval_mode", ""),
                    "n_shot": int(data.get("n_shot", 0)),
                    "accuracy": float(data["accuracy"]),
                    "num_eval_examples": int(data.get("num_eval_examples", len(data.get("details", [])))),
                    "file": os.path.relpath(file_path, results_dir),
                    "path": file_path,
                }
            )
    return records


def infer_model_name_from_path(model_path: str, dataset: str) -> str:
    """从 model_path 尝试恢复基础模型名，兼容 workspace_fewshot/<model>/<dataset>/merged_* 结构。"""
    if not model_path:
        return ""
    normalized = os.path.normpath(model_path)
    parts = normalized.split(os.sep)
    if "workspace_fewshot" in parts:
        idx = parts.index("workspace_fewshot")
        if idx + 2 < len(parts):
            model_candidate = parts[idx + 1]
            dataset_candidate = parts[idx + 2]
            if dataset_candidate == dataset and model_candidate and not model_candidate.startswith("merged_"):
                return model_candidate
    return ""


def attach_model_group(records: List[Dict]) -> List[Dict]:
    """
    将 merged_* 结果归并到基础模型名，避免 summary 目录裂变。
    优先级：
    1) 同 dataset+train_task 的 pre_ft 非 merged 记录
    2) 从 model_path 反推 workspace_fewshot/<model>/<dataset>
    3) 回退原 model_name
    """
    preft_map: Dict[tuple, str] = {}
    for record in records:
        name = str(record.get("model_name", ""))
        if record.get("stage") == "pre_ft" and name and not name.startswith("merged_"):
            preft_map[(record.get("dataset", ""), record.get("train_task", ""))] = name

    for record in records:
        raw_name = str(record.get("model_name", ""))
        model_group = raw_name
        if raw_name.startswith("merged_"):
            key = (record.get("dataset", ""), record.get("train_task", ""))
            mapped = preft_map.get(key, "")
            if mapped:
                model_group = mapped
            else:
                inferred = infer_model_name_from_path(str(record.get("model_path", "")), str(record.get("dataset", "")))
                if inferred:
                    model_group = inferred
        record["model_group"] = model_group or "unknown"
    return records


def match_mode(record: Dict, mode: str) -> bool:
    if mode == "auto":
        return True
    return str(record.get("dataset", "")).strip().lower() == mode


def filter_records(records: List[Dict], args) -> List[Dict]:
    stage_filter = split_filter(args.stage_filter)
    train_task_filter = split_filter(args.train_task_filter)
    eval_task_filter = split_filter(args.eval_task_filter)
    model_name_filter = split_filter(args.model_name_filter)

    filtered = []
    for record in records:
        if not match_mode(record, args.mode):
            continue
        if stage_filter and record["stage"] not in stage_filter:
            continue
        if train_task_filter and record["train_task"] not in train_task_filter:
            continue
        if eval_task_filter and record["eval_task"] not in eval_task_filter:
            continue
        if model_name_filter and record["model_name"] not in model_name_filter:
            continue
        filtered.append(record)
    return filtered


def aggregate_stage(records: List[Dict]):
    bucket = defaultdict(list)
    for record in records:
        key = (record["model_group"], record["stage"], record["checkpoint_label"], record["eval_mode"], record["n_shot"])
        bucket[key].append(record)

    rows = []
    for key in sorted(bucket.keys()):
        items = bucket[key]
        rows.append(
            {
                "model_name": key[0],
                "stage": key[1],
                "checkpoint_label": key[2],
                "eval_mode": key[3],
                "n_shot": key[4],
                "num_records": len(items),
                "acc": mean(item["accuracy"] for item in items),
            }
        )
    return rows


def aggregate_train_task(records: List[Dict]):
    bucket = defaultdict(list)
    for record in records:
        key = (
            record["model_group"],
            record["train_task"],
            record["stage"],
            record["checkpoint_label"],
            record["eval_mode"],
            record["n_shot"],
        )
        bucket[key].append(record)

    rows = []
    for key in sorted(bucket.keys()):
        items = bucket[key]
        rows.append(
            {
                "model_name": key[0],
                "train_task": key[1],
                "stage": key[2],
                "checkpoint_label": key[3],
                "eval_mode": key[4],
                "n_shot": key[5],
                "num_eval_tasks": len(items),
                "acc": mean(item["accuracy"] for item in items),
            }
        )
    return rows


def aggregate_sft_icl(records: List[Dict]):
    """
    输出更明确的标志列：
    - sft_zero_shot_mean_acc: zero-shot 均值（对应 SFT 结果）
    - icl_few_shot_mean_acc: few-shot 均值（对应 ICL 结果）
    """
    bucket = defaultdict(list)
    for record in records:
        key = (record.get("model_group", ""), record.get("dataset", ""))
        bucket[key].append(record)

    rows = []
    for key in sorted(bucket.keys()):
        items = bucket[key]
        zero_shot_items = [x for x in items if x.get("eval_mode") == "zero-shot"]
        few_shot_items = [x for x in items if x.get("eval_mode") == "few-shot"]

        rows.append(
            {
                "model_name": key[0],
                "dataset": key[1],
                "sft_zero_shot_mean_acc": (mean(x["accuracy"] for x in zero_shot_items) if zero_shot_items else None),
                "icl_few_shot_mean_acc": (mean(x["accuracy"] for x in few_shot_items) if few_shot_items else None),
                "num_zero_shot_records": len(zero_shot_items),
                "num_few_shot_records": len(few_shot_items),
            }
        )
    return rows


def save_tables(records: List[Dict], stage_rows: List[Dict], train_rows: List[Dict], sft_icl_rows: List[Dict], output_dir: str):
    detail_csv = os.path.join(output_dir, "task_eval_accuracy.csv")
    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_name",
                "model_group",
                "dataset",
                "train_task",
                "eval_task",
                "stage",
                "checkpoint_label",
                "eval_mode",
                "n_shot",
                "accuracy",
                "num_eval_examples",
                "file",
            ],
        )
        writer.writeheader()
        for record in sorted(
            records,
            key=lambda x: (x.get("model_group", ""), x["train_task"], x["stage"], x["eval_task"], x["checkpoint_label"]),
        ):
            writer.writerow({k: record.get(k, "") for k in writer.fieldnames})

    stage_csv = os.path.join(output_dir, "stage_summary.csv")
    with open(stage_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model_name", "stage", "checkpoint_label", "eval_mode", "n_shot", "num_records", "acc"])
        writer.writeheader()
        for row in stage_rows:
            writer.writerow(row)

    train_csv = os.path.join(output_dir, "train_task_summary.csv")
    with open(train_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model_name", "train_task", "stage", "checkpoint_label", "eval_mode", "n_shot", "num_eval_tasks", "acc"],
        )
        writer.writeheader()
        for row in train_rows:
            writer.writerow(row)

    sft_icl_csv = os.path.join(output_dir, "sft_icl_summary.csv")
    with open(sft_icl_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_name",
                "dataset",
                "sft_zero_shot_mean_acc",
                "icl_few_shot_mean_acc",
                "num_zero_shot_records",
                "num_few_shot_records",
            ],
        )
        writer.writeheader()
        for row in sft_icl_rows:
            writer.writerow(row)

    with open(os.path.join(output_dir, "stage_summary.json"), "w", encoding="utf-8") as f:
        json.dump(stage_rows, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "train_task_summary.json"), "w", encoding="utf-8") as f:
        json.dump(train_rows, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "sft_icl_summary.json"), "w", encoding="utf-8") as f:
        json.dump(sft_icl_rows, f, ensure_ascii=False, indent=2)

    print(f"已输出明细: {detail_csv}")
    print(f"已输出阶段汇总: {stage_csv}")
    print(f"已输出训练单元汇总: {train_csv}")
    print(f"已输出 SFT/ICL 汇总: {sft_icl_csv}")


def group_records_by_model_dataset(records: List[Dict]) -> Dict[tuple, List[Dict]]:
    grouped: Dict[tuple, List[Dict]] = defaultdict(list)
    for record in records:
        key = (str(record.get("model_group", "")), str(record.get("dataset", "")))
        grouped[key].append(record)
    return grouped


def default_group_output_dir(results_dir: str, mode: str, model_name: str, dataset: str) -> str:
    # 与 training_logs 一样按 model+dataset 分层，再放入 acc_analysis_<mode>
    return os.path.join(
        results_dir,
        "summary",
        sanitize_path_component(model_name),
        sanitize_path_component(dataset),
        f"acc_analysis_{mode}",
    )


def cleanup_stale_summary_dirs(results_dir: str, mode: str, valid_keys: set[tuple]) -> None:
    """清理当前 mode 下不再对应任何记录的旧 summary 目录。"""
    summary_root = os.path.join(results_dir, "summary")
    if not os.path.isdir(summary_root):
        return

    for model_folder in os.listdir(summary_root):
        model_path = os.path.join(summary_root, model_folder)
        if not os.path.isdir(model_path):
            continue

        for dataset_folder in os.listdir(model_path):
            dataset_path = os.path.join(model_path, dataset_folder)
            if not os.path.isdir(dataset_path):
                continue

            analysis_dir = os.path.join(dataset_path, f"acc_analysis_{mode}")
            if not os.path.isdir(analysis_dir):
                continue

            if (model_folder, dataset_folder) not in valid_keys:
                shutil.rmtree(analysis_dir, ignore_errors=True)
                print(f"已清理过期目录: {analysis_dir}")

            # 清理空目录层级
            if os.path.isdir(dataset_path) and not os.listdir(dataset_path):
                os.rmdir(dataset_path)
            if os.path.isdir(model_path) and not os.listdir(model_path):
                os.rmdir(model_path)


def main():
    args = parse_args()
    if not os.path.isdir(args.results_dir):
        raise FileNotFoundError(f"结果目录不存在: {args.results_dir}")

    records = filter_records(collect_records(args.results_dir), args)
    if not records:
        print("未找到可统计的结果文件")
        return

    records = attach_model_group(records)

    if args.output_dir:
        output_dir = args.output_dir
        ensure_dir(output_dir)
        stage_rows = aggregate_stage(records)
        train_rows = aggregate_train_task(records)
        sft_icl_rows = aggregate_sft_icl(records)
        save_tables(records, stage_rows, train_rows, sft_icl_rows, output_dir)
        print(f"summary_output_dir={output_dir}")
    else:
        grouped = group_records_by_model_dataset(records)
        valid_keys = {
            (sanitize_path_component(model_name), sanitize_path_component(dataset))
            for (model_name, dataset) in grouped.keys()
        }
        cleanup_stale_summary_dirs(args.results_dir, args.mode, valid_keys)
        for (model_name, dataset), group_records in sorted(grouped.items()):
            output_dir = default_group_output_dir(args.results_dir, args.mode, model_name, dataset)
            ensure_dir(output_dir)
            stage_rows = aggregate_stage(group_records)
            train_rows = aggregate_train_task(group_records)
            sft_icl_rows = aggregate_sft_icl(group_records)
            save_tables(group_records, stage_rows, train_rows, sft_icl_rows, output_dir)
            print(f"summary_output_dir={output_dir}")

    print(f"结果目录: {args.results_dir}")
    print(f"mode={args.mode}")
    print(f"记录条数: {len(records)}")


if __name__ == "__main__":
    main()
