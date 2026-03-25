import argparse
import csv
import json
import os
from collections import defaultdict
from statistics import mean
from typing import Dict, List


def parse_args():
    parser = argparse.ArgumentParser(description="汇总 few-shot 全流程评测结果")
    parser.add_argument("--results_dir", type=str, required=True, help="结果目录，支持递归读取")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录，默认 results_dir/acc_analysis_<mode>")
    parser.add_argument("--mode", choices=["auto", "bbh", "arc", "password", "mmlu"], default="auto", help="统计模式")
    parser.add_argument("--stage_filter", type=str, default="", help="仅统计指定 stage，逗号分隔")
    parser.add_argument("--train_task_filter", type=str, default="", help="仅统计指定 train_task，逗号分隔")
    parser.add_argument("--eval_task_filter", type=str, default="", help="仅统计指定 eval_task，逗号分隔")
    parser.add_argument("--model_name_filter", type=str, default="", help="仅统计指定 model_name，逗号分隔")
    return parser.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


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
        key = (record["model_name"], record["stage"], record["checkpoint_label"], record["eval_mode"], record["n_shot"])
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
        key = (record["model_name"], record["train_task"], record["stage"], record["checkpoint_label"], record["eval_mode"], record["n_shot"])
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


def save_tables(records: List[Dict], stage_rows: List[Dict], train_rows: List[Dict], output_dir: str):
    detail_csv = os.path.join(output_dir, "task_eval_accuracy.csv")
    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_name",
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
        for record in sorted(records, key=lambda x: (x["model_name"], x["train_task"], x["stage"], x["eval_task"], x["checkpoint_label"])):
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

    with open(os.path.join(output_dir, "stage_summary.json"), "w", encoding="utf-8") as f:
        json.dump(stage_rows, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "train_task_summary.json"), "w", encoding="utf-8") as f:
        json.dump(train_rows, f, ensure_ascii=False, indent=2)

    print(f"已输出明细: {detail_csv}")
    print(f"已输出阶段汇总: {stage_csv}")
    print(f"已输出训练单元汇总: {train_csv}")


def main():
    args = parse_args()
    if not os.path.isdir(args.results_dir):
        raise FileNotFoundError(f"结果目录不存在: {args.results_dir}")

    output_dir = args.output_dir or os.path.join(args.results_dir, f"acc_analysis_{args.mode}")
    ensure_dir(output_dir)

    records = filter_records(collect_records(args.results_dir), args)
    if not records:
        print("未找到可统计的结果文件")
        return

    stage_rows = aggregate_stage(records)
    train_rows = aggregate_train_task(records)
    save_tables(records, stage_rows, train_rows, output_dir)

    print(f"结果目录: {args.results_dir}")
    print(f"mode={args.mode}")
    print(f"记录条数: {len(records)}")


if __name__ == "__main__":
    main()
