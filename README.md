# memory_task
## 环境安装流程
### 1.创建conda环境
```bash
conda create -n tttenv python=3.12
conda activate tttenv
```
### 2.安装依赖环境
#### A.安装环境
```bash
pip install -r requirements.txt
```
#### B.安装llama-factory
```bash
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
pip install -e .
pip install -r requirements/metrics.txt
```

## 使用
### BBH
```bash
bash bbh_run_fewshot_pipeline_simple.sh
```
### ARC
```bash
bash arc_run_fewshot_pipeline_simple.sh
```
### MMLU
```bash
bash mmlu_run_fewshot_pipeline_simple.sh
```

### Password
```bash
bash password_run_fewshot_pipeline_simple.sh
```

### Adapter Pipeline (Persistent vLLM + LoRA Adapter Eval)
```bash
bash run_fewshot_pipeline_adapter.sh --model_path /path/to/model --dataset bbh

# dataset-specific entrypoints
bash bbh_run_fewshot_pipeline_adapter.sh --model_path /path/to/model
bash arc_run_fewshot_pipeline_adapter.sh --model_path /path/to/model
bash password_run_fewshot_pipeline_adapter.sh --model_path /path/to/model
bash mmlu_run_fewshot_pipeline_adapter.sh --model_path /path/to/model

# dual-pool GPU settings (default infer=0, train=auto from remaining GPUs)
bash run_fewshot_pipeline_adapter.sh \
	--model_path /path/to/model \
	--dataset bbh \
	--inference_gpu_id 0 \
	--train_gpu_ids 1,2,3
```

Auto split fallback policy (when `train_gpu_ids` is not provided):
- 4+ GPUs: inference=0, training=remaining GPUs
- 3 GPUs: inference=0, training=1,2
- 2 GPUs: inference=0, training=1
- 1 GPU: inference=0, training=0 (serial mode)

