# 通用数据集配置指南

本文档介绍新的通用数据集和提示词配置方案。

## 核心改进

- **统一配置**：所有数据集（包括 BBH）的 prompt 都配置在 `registry.yaml` 中
- **自动检测**：自动检测数据字段名（input/target/examples），减少配置工作
- **分层配置**：支持 default_prompt（默认）+ task_prompts（任务特定）
- **友好报错**：配置错误时给出清晰的解决提示

## 配置文件结构

所有配置都在 `registry.yaml`（或 `.json`）中，顶层为 `datasets`：

```yaml
datasets:
  dataset_name_1:
    data_dir: ./path/to/data
    default_prompt:
      task_prompt: "..."
      answer_format: "..."
      generation_length: 128
    task_prompts:
      task_a:
        task_prompt: "..."
        ...

  dataset_name_2:
    ...
```

## 三种配置方式

### 1. 极简配置（推荐）

适用于标准格式数据，所有任务使用相同的 prompt：

```yaml
datasets:
  my_dataset:
    data_dir: ./data/my_dataset
    default_prompt:
      task_prompt: "请根据问题给出正确答案。"
      answer_format: "只输出答案，不要解释。"
      generation_length: 128
```

**数据格式要求**：
- JSON: `{"examples": [{"input": "...", "target": "..."}]}`
- JSONL: 每行 `{"input": "...", "target": "..."}`

### 2. 任务级 Prompt

适用于同一数据集下不同任务需要不同 prompt：

```yaml
datasets:
  classification:
    data_dir: ./data/classification
    default_prompt:
      task_prompt: "请对输入进行分类。"
      answer_format: "只输出类别标签。"
    task_prompts:
      sentiment:
        task_prompt: "判断文本情感倾向。"
        answer_format: "输出 positive 或 negative。"
      topic:
        task_prompt: "判断文本主题类别。"
        answer_format: "输出 politics/tech/sports。"
```

### 3. 非标准字段名

适用于数据字段名不是 `input`/`target` 的情况：

```yaml
datasets:
  squad:
    data_dir: ./data/squad
    format:
      examples_key: "data"        # 默认为 examples
      input_key: "question"       # 默认为 input
      target_key: "answer"        # 默认为 target
    default_prompt:
      task_prompt: "根据上下文回答问题。"
      answer_format: "只输出答案。"
```

## BBH 数据集配置

### 方式 1：从 src.tasks 导入

如果你有 `src.tasks` 模块，运行工具脚本导出 prompt：

```bash
cd modified_fewshot_pipeline_bundle
python tools/import_bbh_prompts.py > bbh_prompts.yaml
```

然后将内容复制到你的 `registry.yaml` 中。

### 方式 2：手动配置

```yaml
datasets:
  bbh:
    data_dir: ./external/BIG-Bench-Hard/bbh
    default_prompt:
      task_prompt: ""
      answer_format: ""
      generation_length: 128
    task_prompts:
      boolean_expressions:
        task_prompt: "Evaluate the boolean expression..."
        answer_format: "Answer with only 'true' or 'false'."
        generation_length: 16
      # ... 其他 tasks
```

## 自动检测机制

如果未指定字段名，系统会按以下优先级自动检测：

- **examples_key**: `examples` → `data` → `items` → `questions` → ...
- **input_key**: `input` → `question` → `query` → `prompt` → ...
- **target_key**: `target` → `answer` → `output` → `label` → ...

检测失败时会报错并提示配置方式。

## 运行命令

使用 shell 脚本：

```bash
bash run_general_fewshot_pipeline.sh \
  --model_path /path/to/model \
  --dataset my_dataset \
  --dataset_dir ./data/my_dataset \
  --task_registry ./registry.yaml \
  --few_shot_k 5 \
  --epochs 3
```

或直接运行 Python：

```bash
python run_bbh_paraLearn_vllm_ds_lora.py \
  --model_path /path/to/model \
  --dataset my_dataset \
  --dataset_dir ./data/my_dataset \
  --task_registry ./registry.yaml \
  --few_shot_k 5 \
  --epochs 3
```

## 配置文件示例

见 `config_examples/` 目录：

- `minimal.yaml` - 极简配置
- `with_task_prompts.yaml` - 任务级 Prompt
- `with_format.yaml` - 非标准字段名
- `complete_example.yaml` - 完整示例（混合多种数据集）

## 从旧版本迁移

如果你之前使用带 `type: bbh` 的旧配置：

1. 运行导出工具获取 BBH prompts：
   ```bash
   python tools/import_bbh_prompts.py > bbh_prompts.yaml
   ```

2. 将导出的 `task_prompts` 复制到你的 registry.yaml

3. 删除 `type: bbh` 字段

## 故障排除

### 错误："任务 'xxx' 缺少 prompt 配置"

**原因**：该任务没有在 `task_prompts` 中定义，也没有设置 `default_prompt`。

**解决**：添加 `default_prompt` 或为该任务添加 `task_prompts` 配置。

### 错误："无法检测 input 字段"

**原因**：数据字段名不在自动检测列表中。

**解决**：在配置中显式指定字段名：

```yaml
datasets:
  my_dataset:
    data_dir: ./data/my_dataset
    format:
      input_key: "question"
      target_key: "answer"
    default_prompt:
      ...
```
## 2026-03 Current Workflow

当前版本优先支持 `bbh` 和 `arc`，统一入口是：

```bash
bash run_fewshot_pipeline.sh ...
```

关键变化：

- `BBH` 仍然按 `task` 运行，可用 `--task_names`
- `ARC` 按当前 `eval_split` 运行
- 默认比较方式改为：微调前 `few-shot`，微调后 `zero-shot`

当前流程采用 test-time training / TTT 风格切分：

- `BBH`: 对当前 task 本身，取前 `train_size` 条样本做 few-shot/微调支持集，剩余样本做评测。
- `ARC`: 对当前 split 本身，取前 `train_size` 条样本做 few-shot/微调支持集，剩余样本做评测。

常用示例：

```bash
bash run_fewshot_pipeline.sh \
  --model_path /path/to/model \
  --dataset bbh \
  --task_names boolean_expressions,date_understanding \
  --few_shot_k 5 \
  --epochs 3
```

```bash
bash run_fewshot_pipeline.sh \
  --model_path /path/to/model \
  --dataset arc \
  --eval_split validation \
  --few_shot_k 5 \
  --epochs 3
```
