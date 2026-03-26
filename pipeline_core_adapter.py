import argparse
import gc
import json
import os
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

from dataset_runtime import (
    ConfigError,
    RegistryError,
    build_fewshot_prompt_prefix,
    compute_accuracy,
    is_prediction_correct,
    list_eval_units,
    list_training_units,
    load_dataset_config,
    load_training_examples,
    prepare_eval_examples,
    resolve_prompt_spec,
    run_vllm_inference,
)
from pipeline_core import (
    build_result_file_path,
    create_dataset_and_config,
    extract_model_name,
    merge_lora_adapter,
    normalize_modes,
    resolve_workspace_dir,
    save_training_logs,
    setup_workspace,
    str2bool,
    summarize_results,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adapter pipeline: keep vLLM engine in-process and evaluate post_ft via LoRA adapter without merge by default."
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="",
        help="GPU pool for auto split (e.g. 0,1,2,3). Backward-compatible fallback.",
    )
    parser.add_argument("--train_gpu_ids", type=str, default="", help="Training GPU ids, e.g. 1,2,3")
    parser.add_argument("--inference_gpu_id", type=str, default="", help="Inference GPU id, e.g. 0")
    parser.add_argument("--dataset", choices=["bbh", "arc", "password", "mmlu"], required=True, help="Dataset name")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Override dataset directory")
    parser.add_argument("--task_registry", type=str, default=None, help="Registry YAML/JSON, default registry_<dataset>.yaml")
    parser.add_argument("--task_names", type=str, default="", help="Task-style datasets only: comma-separated task list; empty means all")
    parser.add_argument("--eval_split", choices=["validation", "test"], default="test", help="ARC only")

    parser.add_argument("--model_path", type=str, required=True, help="Base model path")

    parser.add_argument("--results_dir", type=str, default="results", help="Results output directory")
    parser.add_argument("--logs_dir", type=str, default="training_logs", help="Training logs directory")
    parser.add_argument("--workspace_dir", type=str, default="workspace_fewshot", help="Workspace directory")
    parser.add_argument("--template_yaml", type=str, default="train_config_template.yaml", help="LlamaFactory training template")

    parser.add_argument("--few_shot_k", type=int, default=5, help="Few-shot inference example count")
    parser.add_argument("--train_size", type=int, default=None, help="Few-shot fine-tuning sample count; defaults to few_shot_k")
    parser.add_argument("--repeat_times", type=int, default=200, help="Repeat count for the constructed fine-tuning set")
    parser.add_argument("--epochs", type=float, default=3.0, help="Number of training epochs")

    parser.add_argument("--run_pre_ft_eval", type=str2bool, default=True, help="Run pre-fine-tuning evaluation")
    parser.add_argument("--run_post_ft_eval", type=str2bool, default=True, help="Run post-fine-tuning evaluation")
    parser.add_argument("--pre_eval_modes", type=str, default="few-shot", help="zero-shot / few-shot / both / off")
    parser.add_argument("--post_eval_modes", type=str, default="zero-shot", help="zero-shot / few-shot / both / off")
    parser.add_argument("--run_checkpoint_eval", type=str2bool, default=False, help="Ignored in adapter pipeline")

    parser.add_argument("--max_model_len", type=int, default=4096, help="vLLM max context length")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90, help="vLLM GPU memory utilization")
    parser.add_argument("--max_lora_rank", type=int, default=64, help="vLLM max LoRA rank when adapter inference is enabled")
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

    parser.add_argument(
        "--allow_merge_fallback",
        type=str2bool,
        default=True,
        help="Fallback to merge+full-model eval if adapter inference is unsupported in current vLLM",
    )
    parser.add_argument(
        "--keep_infer_engine_alive",
        type=str2bool,
        default=True,
        help="Keep the same in-process vLLM engine alive across pre_ft -> training -> post_ft when GPU pools do not overlap.",
    )
    parser.add_argument(
        "--keep_infer_engine_across_tasks",
        type=str2bool,
        default=True,
        help="Reuse the same in-process vLLM engine across different train tasks when safe.",
    )

    return parser.parse_args()


def _parse_gpu_ids(value: str) -> List[str]:
    if not value:
        return []
    result = []
    for item in value.split(","):
        gpu = item.strip()
        if not gpu:
            continue
        if gpu not in result:
            result.append(gpu)
    return result


def _detect_available_gpu_ids() -> List[str]:
    visible_env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible_env and visible_env != "-1":
        parsed = _parse_gpu_ids(visible_env)
        if parsed:
            return parsed

    try:
        import torch

        if torch.cuda.is_available():
            return [str(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass

    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        ids = []
        for line in output.splitlines():
            gpu = line.strip()
            if gpu and gpu not in ids:
                ids.append(gpu)
        return ids
    except Exception:
        return []


def resolve_gpu_assignment(args: argparse.Namespace) -> Dict[str, Any]:
    detected_pool = _detect_available_gpu_ids()
    configured_pool = _parse_gpu_ids(args.gpu_ids)

    if configured_pool and detected_pool:
        filtered = [gpu for gpu in configured_pool if gpu in detected_pool]
        if filtered:
            if len(filtered) != len(configured_pool):
                dropped = [gpu for gpu in configured_pool if gpu not in filtered]
                print(f"[Adapter][GPU] 忽略不存在的 gpu_ids: {','.join(dropped)}")
            auto_pool = filtered
        else:
            raise ValueError(
                f"gpu_ids={configured_pool} 与可用 GPU {detected_pool} 无交集，请检查配置。"
            )
    elif configured_pool:
        auto_pool = configured_pool
    else:
        auto_pool = detected_pool

    if not auto_pool:
        raise ValueError("未检测到可用 GPU；请显式传入 --train_gpu_ids/--inference_gpu_id。")

    explicit_train = _parse_gpu_ids(args.train_gpu_ids)
    explicit_infer = args.inference_gpu_id.strip() if args.inference_gpu_id else ""

    if detected_pool:
        for gpu in explicit_train:
            if gpu not in detected_pool:
                raise ValueError(f"train_gpu_ids 包含不可用 GPU: {gpu}，可用 GPU: {detected_pool}")
        if explicit_infer and explicit_infer not in detected_pool:
            raise ValueError(f"inference_gpu_id={explicit_infer} 不可用，可用 GPU: {detected_pool}")

    if explicit_train and explicit_infer:
        train_ids = explicit_train
        infer_id = explicit_infer
        strategy = "explicit(train+infer)"
    elif explicit_train:
        train_ids = explicit_train
        candidates = [gpu for gpu in auto_pool if gpu not in train_ids]
        infer_id = candidates[0] if candidates else train_ids[0]
        strategy = "explicit(train)+auto(infer)"
    elif explicit_infer:
        infer_id = explicit_infer
        train_ids = [gpu for gpu in auto_pool if gpu != infer_id]
        if not train_ids:
            train_ids = [infer_id]
        strategy = "explicit(infer)+auto(train)"
    else:
        infer_id = auto_pool[0]
        if len(auto_pool) >= 2:
            train_ids = auto_pool[1:]
        else:
            train_ids = [infer_id]
        strategy = "auto"

    if not train_ids:
        train_ids = [infer_id]

    serial_mode = infer_id in train_ids
    if serial_mode and len(train_ids) == 1:
        effective_mode = "single-gpu serial"
    else:
        effective_mode = "dual-pool"

    return {
        "train_gpu_ids": ",".join(train_ids),
        "train_main_gpu": train_ids[0],
        "inference_gpu_id": infer_id,
        "detected_pool": detected_pool,
        "auto_pool": auto_pool,
        "strategy": strategy,
        "serial_mode": serial_mode,
        "effective_mode": effective_mode,
    }


class PersistentVLLMRunner:
    def __init__(
        self,
        model_path: str,
        max_model_len: int,
        gpu_memory_utilization: float,
        inference_gpu_id: str,
        enable_lora: bool,
        max_lora_rank: int,
    ) -> None:
        from vllm import LLM

        self.model_path = model_path
        self.enable_lora = bool(enable_lora)
        self.inference_gpu_id = str(inference_gpu_id)
        self.lora_supported = False
        self._lora_request_cls = None
        self._next_lora_id = 1
        self._lora_request_cache: Dict[str, Tuple[float, Any]] = {}
        self._prev_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

        os.environ["CUDA_VISIBLE_DEVICES"] = self.inference_gpu_id
        print(f"[Adapter][vLLM] Bind inference GPU: CUDA_VISIBLE_DEVICES={self.inference_gpu_id}")

        llm_kwargs: Dict[str, Any] = {
            "model": model_path,
            "trust_remote_code": True,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
        }

        if self.enable_lora:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = int(max_lora_rank)

        try:
            self.llm = LLM(**llm_kwargs)
        except TypeError as e:
            if not self.enable_lora:
                raise
            # Older vLLM may not accept enable_lora/max_lora_rank kwargs.
            print(f"[Adapter][WARN] enable_lora 初始化失败，将退回 base-only 引擎: {e}")
            fallback_kwargs = {
                "model": model_path,
                "trust_remote_code": True,
                "gpu_memory_utilization": gpu_memory_utilization,
                "max_model_len": max_model_len,
            }
            self.llm = LLM(**fallback_kwargs)
            self.enable_lora = False

        if self.enable_lora:
            self._lora_request_cls = self._import_lora_request_class()
            self.lora_supported = self._lora_request_cls is not None

    def _import_lora_request_class(self) -> Optional[type]:
        try:
            from vllm.lora.request import LoRARequest

            return LoRARequest
        except Exception:
            pass
        try:
            from vllm import LoRARequest

            return LoRARequest
        except Exception:
            return None

    def _build_lora_request(self, adapter_path: str):
        if not self.lora_supported or self._lora_request_cls is None:
            raise RuntimeError("Current vLLM runtime does not support LoRARequest adapter inference.")
        adapter_abs_path = os.path.abspath(adapter_path)
        adapter_stamp = self._get_adapter_stamp(adapter_abs_path)
        cached = self._lora_request_cache.get(adapter_abs_path)
        if cached is not None and cached[0] == adapter_stamp:
            return cached[1]

        lora_id = self._next_lora_id
        self._next_lora_id += 1
        lora_name = f"adapter_{os.path.basename(os.path.normpath(adapter_abs_path))}_{lora_id}"
        request = self._lora_request_cls(lora_name, lora_id, adapter_abs_path)
        self._lora_request_cache[adapter_abs_path] = (adapter_stamp, request)
        return request

    def _get_adapter_stamp(self, adapter_path: str) -> float:
        # Track adapter changes and only allocate a new LoRARequest when files changed.
        candidates = [
            os.path.join(adapter_path, "adapter_model.safetensors"),
            os.path.join(adapter_path, "adapter_model.bin"),
            adapter_path,
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return os.path.getmtime(candidate)
        return time.time()

    def predict(
        self,
        questions: List[str],
        dataset_cfg: Dict[str, Any],
        prompt_spec: Any,
        max_new_tokens: int,
        few_shot_prompt_prefix: str,
        adapter_path: Optional[str] = None,
    ) -> List[str]:
        lora_request = None
        if adapter_path:
            lora_request = self._build_lora_request(adapter_path)

        return run_vllm_inference(
            llm=self.llm,
            questions=questions,
            dataset_cfg=dataset_cfg,
            prompt_spec=prompt_spec,
            max_new_tokens=max_new_tokens,
            few_shot_prompt_prefix=few_shot_prompt_prefix,
            lora_request=lora_request,
        )

    def close(self) -> None:
        try:
            del self.llm
        except Exception:
            pass

        if self._prev_cuda_visible_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = self._prev_cuda_visible_devices

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def evaluate_and_write(
    runner: PersistentVLLMRunner,
    args: argparse.Namespace,
    dataset_cfg: Dict[str, Any],
    train_unit: str,
    eval_unit: str,
    eval_mode: str,
    n_shot: int,
    train_size: int,
    output_path: str,
    stage: str,
    checkpoint_label: str,
    adapter_path: Optional[str],
) -> bool:
    try:
        prompt_spec = resolve_prompt_spec(args.dataset, dataset_cfg, eval_unit)
        support_examples, eval_examples = prepare_eval_examples(args.dataset, dataset_cfg, eval_unit, train_size)
    except (FileNotFoundError, RegistryError, ConfigError) as e:
        print(f"[Adapter][Eval] 准备失败: eval_unit={eval_unit} error={e}")
        return False

    effective_n_shot = 0
    few_shot_prompt_prefix = ""
    if eval_mode == "few-shot":
        effective_n_shot = min(n_shot, len(support_examples))
        few_shot_prompt_prefix = build_fewshot_prompt_prefix(
            support_examples,
            dataset_cfg,
            effective_n_shot,
        )

    eval_questions = [ex["input"] for ex in eval_examples]
    eval_targets = [ex["target"] for ex in eval_examples]

    try:
        preds = runner.predict(
            questions=eval_questions,
            dataset_cfg=dataset_cfg,
            prompt_spec=prompt_spec,
            max_new_tokens=prompt_spec.generation_length,
            few_shot_prompt_prefix=few_shot_prompt_prefix,
            adapter_path=adapter_path,
        )
    except Exception as e:
        print(f"[Adapter][Eval] 推理失败: eval_unit={eval_unit} mode={eval_mode} error={e}")
        return False

    acc_percent = compute_accuracy(preds, eval_targets)
    acc = acc_percent / 100.0

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

    result_data = {
        "dataset": args.dataset,
        "task": eval_unit,
        "eval_task": eval_unit,
        "train_task": train_unit,
        "stage": stage,
        "checkpoint_label": checkpoint_label,
        "eval_mode": eval_mode,
        "n_shot": effective_n_shot,
        "train_size": train_size,
        "accuracy": acc,
        "num_eval_examples": len(eval_questions),
        # Keep model_name stable so downstream summary does not split merged/adapter paths.
        "model_name": extract_model_name(args.model_path),
        "model_path": args.model_path,
        "inference_model_path": args.model_path,
        "adapter_path": adapter_path or "",
        "note": args.eval_batch_note,
        "details": details,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    print(
        f"[Adapter][Eval] stage={stage} eval_unit={eval_unit} mode={eval_mode} "
        f"acc={acc_percent:.2f}% adapter={'yes' if adapter_path else 'no'}"
    )
    return True


def main() -> None:
    args = parse_args()

    gpu_plan = resolve_gpu_assignment(args)
    train_gpu_ids = gpu_plan["train_gpu_ids"]
    train_main_gpu = gpu_plan["train_main_gpu"]
    inference_gpu_id = gpu_plan["inference_gpu_id"]

    train_size = args.train_size if args.train_size is not None else args.few_shot_k
    if train_size <= 0:
        raise ValueError("train_size must be greater than 0")

    if args.run_checkpoint_eval:
        print("[Adapter][Info] run_checkpoint_eval 在 adapter 流程中被忽略（仅 final 评测）。")
        args.run_checkpoint_eval = False

    run_workspace_dir = resolve_workspace_dir(args.workspace_dir, args.model_path, args.dataset)
    run_logs_dir = resolve_workspace_dir(args.logs_dir, args.model_path, args.dataset)

    print(f"[Adapter] GPU strategy={gpu_plan['strategy']} mode={gpu_plan['effective_mode']}")
    if gpu_plan["detected_pool"]:
        print(f"[Adapter] detected_gpu_pool={','.join(gpu_plan['detected_pool'])}")
    print(f"[Adapter] auto_gpu_pool={','.join(gpu_plan['auto_pool'])}")
    print(f"[Adapter] Training GPU(s): {train_gpu_ids}")
    print(f"[Adapter] Inference GPU: {inference_gpu_id}")
    if gpu_plan["serial_mode"]:
        print("[Adapter][WARN] 训练与推理 GPU 重叠，自动降级为单卡串行模式。")
    print(f"[Adapter] dataset={args.dataset} few_shot_k={args.few_shot_k} train_size={train_size} epochs={args.epochs}")
    print(
        f"[Adapter] train_batch(per_device={args.per_device_train_batch_size}, "
        f"grad_accum={args.gradient_accumulation_steps})"
    )
    print(f"[Adapter] workspace_dir={run_workspace_dir}")
    print(f"[Adapter] logs_dir={run_logs_dir}")

    dataset_cfg = load_dataset_config(args.dataset, args.dataset_dir, args.task_registry)
    selected_units = list_training_units(args.dataset, dataset_cfg, args.task_names)
    eval_units = list_eval_units(args.dataset, dataset_cfg, selected_units, "selected", args.eval_split)

    if args.dataset == "arc":
        selected_units = [args.eval_split]
        eval_units = [args.eval_split]

    paths = setup_workspace(run_workspace_dir, args.results_dir, run_logs_dir)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print(f"[Adapter] Training units: {selected_units}")
    print(f"[Adapter] Evaluation units: {eval_units}")

    pre_eval_modes = normalize_modes(args.pre_eval_modes) if args.run_pre_ft_eval else []
    post_eval_modes = normalize_modes(args.post_eval_modes) if args.run_post_ft_eval else []

    can_keep_runner_alive = bool(
        args.keep_infer_engine_alive
        and pre_eval_modes
        and post_eval_modes
        and (not args.skip_training)
        and (not gpu_plan["serial_mode"])
    )

    if can_keep_runner_alive:
        print("[Adapter] keep_infer_engine_alive=on（跨训练阶段复用同一个 vLLM 引擎）")
    elif args.keep_infer_engine_alive:
        print(
            "[Adapter] keep_infer_engine_alive 未生效（需要同时有 pre/post 评测、非 skip_training、且训练/推理 GPU 不重叠）。"
        )

    keep_runner_across_tasks = bool(can_keep_runner_alive and args.keep_infer_engine_across_tasks)
    global_shared_runner: Optional[PersistentVLLMRunner] = None
    if keep_runner_across_tasks:
        print("[Adapter] keep_infer_engine_across_tasks=on（跨 task 复用同一个 vLLM 引擎）")
    elif args.keep_infer_engine_across_tasks:
        print("[Adapter] keep_infer_engine_across_tasks 未生效（依赖 keep_infer_engine_alive 生效条件）。")

    for unit_index, train_unit in enumerate(selected_units, start=1):
        print(f"\n{'=' * 12} [Adapter] unit [{unit_index}/{len(selected_units)}]: {train_unit} {'=' * 12}")

        current_eval_units = [train_unit] if args.dataset in {"bbh", "password", "mmlu"} else eval_units

        try:
            train_examples_all = load_training_examples(args.dataset, dataset_cfg, train_unit)
        except (FileNotFoundError, RegistryError, ConfigError) as e:
            print(f"[Adapter] Skip {train_unit}: failed to load training examples: {e}")
            continue

        if len(train_examples_all) < train_size:
            print(f"[Adapter] Skip {train_unit}: need at least {train_size} training examples, got {len(train_examples_all)}")
            continue

        train_examples = train_examples_all[:train_size]
        dynamic_len = create_dataset_and_config(args, dataset_cfg, train_unit, train_examples, tokenizer, paths)
        print(f"[Adapter] Prepared training data and config, cutoff_len={dynamic_len}")

        shared_runner = global_shared_runner if keep_runner_across_tasks else None
        if can_keep_runner_alive and shared_runner is None:
            try:
                shared_runner = PersistentVLLMRunner(
                    model_path=args.model_path,
                    max_model_len=args.max_model_len,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    inference_gpu_id=inference_gpu_id,
                    enable_lora=True,
                    max_lora_rank=args.max_lora_rank,
                )
                if keep_runner_across_tasks:
                    global_shared_runner = shared_runner
                print("[Adapter] Shared vLLM engine created for pre_ft and post_ft.")
            except Exception as e:
                print(f"[Adapter][WARN] 创建共享引擎失败，将回退到分阶段加载: {e}")
                shared_runner = None

        try:
            # Pre-FT: one persistent base engine for all pre-ft evals in this unit.
            if pre_eval_modes:
                pre_runner = None
                own_pre_runner = False
                try:
                    if shared_runner is not None:
                        pre_runner = shared_runner
                    else:
                        pre_runner = PersistentVLLMRunner(
                            model_path=args.model_path,
                            max_model_len=args.max_model_len,
                            gpu_memory_utilization=args.gpu_memory_utilization,
                            inference_gpu_id=inference_gpu_id,
                            enable_lora=False,
                            max_lora_rank=args.max_lora_rank,
                        )
                        own_pre_runner = True
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
                                print(f"[Adapter] Skip existing result: {out_path}")
                                continue
                            evaluate_and_write(
                                runner=pre_runner,
                                args=args,
                                dataset_cfg=dataset_cfg,
                                train_unit=train_unit,
                                eval_unit=eval_unit,
                                eval_mode=eval_mode,
                                n_shot=n_shot,
                                train_size=train_size,
                                output_path=out_path,
                                stage="pre_ft",
                                checkpoint_label="base",
                                adapter_path=None,
                            )
                finally:
                    if own_pre_runner and pre_runner is not None:
                        pre_runner.close()

            if args.skip_training:
                print("[Adapter] Skip training and run evaluation only")
                continue

            # Release vLLM memory before training only when engine is not kept alive.
            if shared_runner is None:
                gc.collect()
            else:
                print("[Adapter] Keep shared vLLM engine alive during training.")

            if not train_model(args, train_gpu_ids, train_main_gpu, paths):
                print(f"[Adapter] Training failed: {train_unit}")
                continue

            time.sleep(3)
            if args.save_training_logs:
                save_training_logs(train_unit, run_logs_dir, paths["OUTPUT_DIR"])

            if post_eval_modes:
                post_done = False
                post_runner = None
                own_post_runner = False

                try:
                    if shared_runner is not None:
                        post_runner = shared_runner
                    else:
                        post_runner = PersistentVLLMRunner(
                            model_path=args.model_path,
                            max_model_len=args.max_model_len,
                            gpu_memory_utilization=args.gpu_memory_utilization,
                            inference_gpu_id=inference_gpu_id,
                            enable_lora=True,
                            max_lora_rank=args.max_lora_rank,
                        )
                        own_post_runner = True

                    if post_runner.lora_supported:
                        adapter_path = paths["OUTPUT_DIR"]
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
                                    print(f"[Adapter] Skip existing result: {out_path}")
                                    continue
                                evaluate_and_write(
                                    runner=post_runner,
                                    args=args,
                                    dataset_cfg=dataset_cfg,
                                    train_unit=train_unit,
                                    eval_unit=eval_unit,
                                    eval_mode=eval_mode,
                                    n_shot=n_shot,
                                    train_size=train_size,
                                    output_path=out_path,
                                    stage="post_ft",
                                    checkpoint_label="final",
                                    adapter_path=adapter_path,
                                )
                        post_done = True
                    else:
                        print("[Adapter][WARN] 当前 vLLM 不支持 LoRARequest，无法直接 adapter 评测。")
                finally:
                    if own_post_runner and post_runner is not None:
                        post_runner.close()

                # Optional fallback: merge then evaluate as full model if adapter inference unsupported.
                if (not post_done) and args.allow_merge_fallback:
                    final_merged_dir = os.path.join(paths["WORK_DIR"], f"merged_{train_unit}_final")
                    if merge_lora_adapter(args.model_path, inference_gpu_id, paths["OUTPUT_DIR"], final_merged_dir):
                        merged_runner = None
                        try:
                            print("[Adapter][Fallback] 使用 merge+full-model 进行 post_ft 评测。")
                            merged_runner = PersistentVLLMRunner(
                                model_path=final_merged_dir,
                                max_model_len=args.max_model_len,
                                gpu_memory_utilization=args.gpu_memory_utilization,
                                inference_gpu_id=inference_gpu_id,
                                enable_lora=False,
                                max_lora_rank=args.max_lora_rank,
                            )
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
                                        print(f"[Adapter] Skip existing result: {out_path}")
                                        continue
                                    evaluate_and_write(
                                        runner=merged_runner,
                                        args=args,
                                        dataset_cfg=dataset_cfg,
                                        train_unit=train_unit,
                                        eval_unit=eval_unit,
                                        eval_mode=eval_mode,
                                        n_shot=n_shot,
                                        train_size=train_size,
                                        output_path=out_path,
                                        stage="post_ft",
                                        checkpoint_label="final",
                                        adapter_path=None,
                                    )
                        finally:
                            if merged_runner is not None:
                                merged_runner.close()
                            if os.path.exists(final_merged_dir):
                                shutil.rmtree(final_merged_dir)
        finally:
            if (shared_runner is not None) and (not keep_runner_across_tasks):
                shared_runner.close()

    if global_shared_runner is not None:
        global_shared_runner.close()

    summarize_results(os.path.abspath(args.results_dir))


if __name__ == "__main__":
    main()
