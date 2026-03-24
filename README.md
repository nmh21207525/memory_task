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

### Password
```bash
bash password_run_fewshot_pipeline_simple.sh
```

