<div align="right">
  <a href="README.md">🇬🇧 English</a> &nbsp;|&nbsp;
  <a href="README_CN.md">🇨🇳 中文</a>
</div>

# Critic-R1：基于强化学习的QA自批判模型

本仓库包含训练和评估 critic 模型的代码，该模型用于验证和修正检索增强型 QA 系统的输出。

## 环境配置

### 1. Search-R1 推理与检索环境

我们的检索和推理管线基于 **Search-R1** 构建。请按照 Search-R1 仓库的说明进行安装配置：

> [https://github.com/PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1)

克隆并按照其 README 安装 Search-R1。该环境提供了检索服务（`http://127.0.0.1:8000/retrieve`）以及 `infer/` 和 `eval/` 脚本所需的基础推理工具。

### 2. Critic 模型训练环境

创建专用的 conda 环境用于 critic 训练：

```bash
conda create -n critic-r1 python=3.12.0 -y
conda activate critic-r1
pip install -r requirements.txt
```

如果提供了 `environment.yaml`，也可以使用以下方式创建环境：

```bash
conda env create -f environment.yaml -n critic-r1
conda activate critic-r1
```

我们的训练管线基于 **VERL** 框架构建。请按照 VERL 官方指南安装：

> [https://github.com/verl-project/verl](https://github.com/verl-project/verl)

VERL 应安装到同一个 `critic-r1` 环境中。

## Critic 训练管线

Critic 模型采用两阶段训练（保守判断对齐 → 诊断质量对齐）。以下是生成训练数据并执行训练的完整流程。

### 第一步：生成带上下文的推理轨迹

使用 HotpotQA 训练集生成 QA 推理轨迹，并附带结构化上下文（支持性标题、证据、候选标题）：

```bash
python supervision/hotpot_train.py \
    --suffix v1
```

生成 `hotpotqa_trainset_inference_results_v1.csv`，包含问题、模型输出、提取答案、标准答案和结构化上下文。

### 第二步：通过投票标注生成 Critic 标注

我们提供两种标注脚本——分别使用 **DeepSeek API** 和**本地 Qwen 模型**。两者均使用多轮采样投票机制生成稳健的 critic 标签。

**方案 A：DeepSeek API**

```bash
export DEEPSEEK_API_KEY="your-api-key"

python supervision/llm_as_judge_ds.py \
    --in_csv hotpotqa_trainset_inference_results_v1.csv \
    --out_csv hotpotqa_trainset_critic_labels_ds.csv \
    --model_name deepseek-chat \
    --critic_num_votes 3 \
    --judge_num_votes 3
```

**方案 B：本地 Qwen 模型**

```bash
python supervision/llm_as_judge_qwen.py \
    --in_csv hotpotqa_trainset_inference_results_v1.csv \
    --out_csv hotpotqa_trainset_critic_labels_qwen.csv \
    --model_path /path/to/Qwen2.5-14B-Instruct \
    --load_in_4bit \
    --critic_num_votes 3 \
    --judge_num_votes 3
```

两种脚本均生成包含 `critique`、`keywords` 和 `llm_judge` 列的 CSV 文件，其中包含结构化的 critic 标注。

### 第三步：构建训练 Parquet 文件

将标注后的 CSV 转换为 VERL 训练框架所需的 parquet 格式：

```bash
python train/gen_critic_train.py \
    --csv_path hotpotqa_trainset_critic_labels_ds.csv \
    --out_dir ./data/critic \
    --out_name train_critic.parquet \
    --question_col question \
    --traj_col model_output \
    --gold_col golden_answer \
    --llm_judge_col llm_judge \
    --critique_col critique \
    --keywords_col keywords \
    --content_col context
```

### 第四步：两阶段 RL 训练

使用 VERL 框架运行基于 GRPO 的两阶段 PPO 训练：

```bash
# 设置所需路径
export BASE_MODEL=/path/to/base_model
export LORA_PATH=/path/to/lora_adapter
export TRAIN_FILE=./data/critic/train_critic.parquet
export VAL_FILE=./data/critic/val_critic.parquet
export REWARD_DIR=./train
export EXPERIMENT_NAME=critic-r1-exp

# 第一阶段：保守判断对齐（默认）
# 第二阶段：诊断质量对齐
bash train/train_critic.sh
```

两个阶段通过环境变量控制：
- `STEP1=true` — 保守判断对齐（惩罚过度激进的错误判断）
- `STEP2=true` — 诊断质量对齐（奖励精确的位置、原因和修复方案预测）

默认情况下脚本仅运行第二阶段。将两者均设为 `true` 即可运行完整的二阶段课程训练。

### 第五步：提取 LoRA 适配器

训练完成后，从 FSDP checkpoint 中提取 LoRA 适配器：

```bash
python train/merge_lora.py \
    --checkpoint_path /path/to/checkpoint/global_step_xxx \
    --base_model_path /path/to/base_model \
    --target_dir ./extracted_lora
```

独立的 LoRA 适配器权重将保存到 `--target_dir`，可用于后续推理或评估。

## 评估

评估管线分为四个阶段：(1) 在 QA 基准上运行推理，(2) 为每条轨迹生成 critic 输出，(3) 应用 critic 反馈修正错误答案，(4) 对原始和修正后的输出计算最终指标。

### 阶段一：QA 基准推理

每个数据集有独立的推理脚本。关键可配参数在脚本顶部设置：

| 脚本 | 数据集 | 数据文件 |
|------|--------|----------|
| `infer/infer_nq.py` | Natural Questions (dev) | `NQ-open.dev.jsonl` |
| `infer/infer_trivia.py` | TriviaQA (val) | `trivia_qa_val.parquet` |
| `infer/infer_hotpot.py` | HotpotQA (dev) | `hotpot_dev_distractor_v1.json` |
| `infer/infer_popqa.py` | PopQA (test) | `test.tsv` |
| `infer/infer_asqa.py` | ASQA (dev) | `devset.parquet` |

**示例 — HotpotQA：**

```bash
python infer/infer_hotpot.py \
    --suffix my_experiment
```

运行前根据需要调整脚本顶部的可配变量：

```python
num_questions = 10000                  # 处理的样本数
model_id = "your-org/your-model"       # HuggingFace 模型或本地路径
max_search_calls = 1                   # 每个问题最大检索次数
max_new_tokens = 1024                  # 每步最大生成 token 数
do_sample = True                       # 采样解码（关闭则为贪心解码）
temperature = 0.7                      # 采样温度
top_p = 0.9                            # 核采样阈值
retriever_url = "http://127.0.0.1:8000/retrieve"  # Search-R1 检索服务端点
```

其他数据集遵循相同模式——配置这些变量后直接运行脚本：

```bash
python infer/infer_nq.py
python infer/infer_trivia.py
python infer/infer_hotpot.py --suffix my_exp
python infer/infer_popqa.py
python infer/infer_asqa.py
```

每个脚本输出一个 CSV（例如 `nq_dev_inference_results.csv`），包含 `question`、`model_output`、`extracted_answer`、`golden_answer` 列。

### 阶段二：生成 Critic 输出

使用训练好的 critic 模型（base + LoRA 或完整模型）审计每条轨迹并生成结构化批评：

```bash
python eval/evaluate_lora_3b.py \
    --csv_path nq_dev_inference_results.csv \
    --out_csv nq_dev_with_critique.csv \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --lora_dir ./extracted_lora \
    --use_lora \
    --output_col 3b_critique \
    --question_col question \
    --traj_col model_output \
    --extracted_answer_col extracted_answer \
    --gold_col golden_answer \
    --critique_col critique \
    --llm_judge_col llm_judge \
    --max_new_tokens 256 \
    --n 500 \
    --save_every 20 \
    --print_every 20 \
    --resume
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--csv_path` | *(必填)* | 阶段一输出的 CSV |
| `--out_csv` | *(必填)* | 追加 critic 列后的输出 CSV |
| `--base_model` | `Qwen/Qwen2.5-3B-Instruct` | 基础模型（用于 tokenizer 和权重） |
| `--lora_dir` | `""` | LoRA 适配器目录路径 |
| `--use_lora` | `False` | 启用 LoRA 适配器加载 |
| `--model_path` | `""` | 完整 HF 模型路径（覆盖 base+LoRA） |
| `--output_col` | `3b_critique` | 输出 critic 列名 |
| `--question_col` | `question` | 问题列名 |
| `--traj_col` | `trajectory` | 完整轨迹/模型输出列 |
| `--extracted_answer_col` | `extracted_answer` | 提取答案列 |
| `--gold_col` | `golden_answer` | 标准答案列 |
| `--critique_col` | `critique` | 已有 critic 列（如有） |
| `--llm_judge_col` | `llm_judge` | 已有 LLM judge 列（如有） |
| `--max_new_tokens` | `256` | 每次 critic 生成的最大 token 数 |
| `--n` | `-1`（全部） | 仅处理前 N 行 |
| `--save_every` | `20` | 每新填充 N 行自动保存 |
| `--print_every` | `20` | 每 N 行打印进度 |
| `--resume` | `False` | 跳过已有非空输出的行 |

### 阶段三：应用 Critic 反馈进行修正

运行修正循环：当 critic 判定为 `INCORRECT` 时，脚本以批评为指导触发重新生成，然后衡量答案是否得到改善：

```bash
python eval/evaluate_critic_feedback.py \
    --input_csv_path nq_dev_with_critique.csv \
    --output_csv_path nq_dev_refined.csv \
    --stats_json_path nq_dev_refined_stats.json \
    --model_path /path/to/your/base-model \
    --n 500 \
    --local_files_only
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input_csv_path` | *(必填)* | 包含 `3b_critique`（或 `critique`）列的 CSV |
| `--output_csv_path` | *(必填)* | 包含修正结果的输出 CSV |
| `--stats_json_path` | *(必填)* | 聚合统计 JSON |
| `--model_path` | *(必填)* | 用于修正生成的 QA 模型路径 |
| `--n` | `-1`（全部） | 仅处理前 N 行 |
| `--local_files_only` | `False` | 仅使用本地模型文件 |

输出 CSV 包含：
- `critique_verdict` — 提取的判定（CORRECT / INCORRECT / UNSURE）
- `old_correct` / `final_correct` — 修正前后的正确性
- `used_refine` — 是否触发了修正
- `improved_after_refine` / `harmed_after_refine` — 方向性变化标记

统计 JSON 报告错误检测精确率/召回率、修正成功率以及净提升分解。

### 阶段四：计算最终指标

对最终输出运行完整指标套件（EM、F1、SBERT 相似度、LLM judge）：

```bash
python eval/evaluate_metrics.py \
    --input_path nq_dev_refined.csv \
    --output_path nq_dev_final_eval.csv \
    --summary_path nq_dev_final_summary.json \
    --answer_col final_extracted_answer \
    --gold_col golden_answer \
    --question_col question \
    --sbert_model_name sentence-transformers/all-MiniLM-L6-v2 \
    --judge_model_path Qwen/Qwen2.5-7B-Instruct \
    --limit 1000 \
    --save_every 10 \
    --resume
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input_path` | *(必填)* | 阶段三（或阶段一）输出的 CSV |
| `--output_path` | `{input}_eval.csv` | 逐样本详细结果 |
| `--summary_path` | `{input}_summary.json` | 聚合指标 JSON |
| `--answer_col` | `extracted_answer` | 模型预测列名 |
| `--gold_col` | `golden_answer` | 标准答案列名 |
| `--question_col` | `question` | 问题列名（供 LLM judge 使用） |
| `--sbert_model_name` | `all-MiniLM-L6-v2` | 语义相似度 SBERT 模型 |
| `--judge_model_path` | `Qwen/Qwen2.5-7B-Instruct` | LLM judge 模型路径 |
| `--limit` | `None` | 仅评估前 N 行 |
| `--save_every` | `10` | 每 N 行保存中间结果 |
| `--resume` | `False` | 从已有输出 CSV 续跑 |

摘要 JSON 报告：EM、F1、Precision、Recall、SBERT 余弦相似度、LLM judge 准确率，以及（ASQA 数据集）str-EM / str-Hit。

## 项目结构

```
├── infer/                  # QA 基准推理脚本
│   ├── infer_nq.py         #   Natural Questions
│   ├── infer_trivia.py     #   TriviaQA
│   ├── infer_hotpot.py     #   HotpotQA
│   ├── infer_popqa.py      #   PopQA
│   └── infer_asqa.py       #   ASQA
├── supervision/            # 训练数据生成与标注
│   ├── hotpot_train.py     #   轨迹 + 上下文生成
│   ├── llm_as_judge_ds.py  #   基于 DeepSeek 的投票标注
│   └── llm_as_judge_qwen.py#   基于 Qwen 的投票标注
├── train/                  # 训练管线
│   ├── gen_critic_train.py #   CSV → parquet 格式转换
│   ├── critic_reward.py    #   两阶段奖励函数
│   ├── train_critic.sh     #   PPO 训练启动脚本
│   └── merge_lora.py       #   LoRA 适配器提取
└── eval/                   # 评估
    ├── evaluate_metrics.py #   EM、F1、SBERT、LLM-Judge 指标
    ├── evaluate_lora_3b.py #   带 critic 反馈的 LoRA 模型评估
    └── evaluate_critic_feedback.py  # Critic 反馈修正评估
```

## 许可证

[待添加]
