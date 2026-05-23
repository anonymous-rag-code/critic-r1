<div align="right">
  <a href="README.md">🇬🇧 English</a> &nbsp;|&nbsp;
  <a href="README_CN.md">🇨🇳 中文</a>
</div>

# CRITIC-R1: Learning Reliable Critics for Retrieval-Augmented Generation

This repository contains the code for training and evaluating a critic model that verifies and refines retrieval-augmented QA outputs.

## Environment Setup

### 1. Search-R1 Inference & Retrieval Environment

The retrieval and inference pipeline is built on top of **Search-R1**. Please follow the setup instructions in the Search-R1 repository:

> [https://github.com/PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1)

Clone and install Search-R1 following its README. This provides the retrieval server (`http://127.0.0.1:8000/retrieve`) and the base inference utilities used by the `infer/` and `eval/` scripts.

### 2. Critic Model Training Environment

Create a dedicated conda environment for critic training:

```bash
conda create -n critic-r1 python=3.12.0 -y
conda activate critic-r1
pip install -r requirements.txt
```

Alternatively, create the environment from the `environment.yml` file:

```bash
conda env create -f environment.yml -n critic-r1
conda activate critic-r1
```

The training pipeline is built on the **VERL** framework. Install VERL following its official guide:

> [https://github.com/verl-project/verl](https://github.com/verl-project/verl)

VERL should be installed into the same `critic-r1` environment.

## Critic Training Pipeline

The critic model is trained in two stages (CJA → DQA). Below is the step-by-step pipeline for generating training data and running the training.

### Step 1: Generate Trajectories with Context

Use the HotpotQA training set to produce QA trajectories paired with structured context (supporting titles, evidence, candidate titles):

```bash
python supervision/hotpot_train.py \
    --suffix v1
```

This produces `hotpotqa_trainset_inference_results_v1.csv` containing questions, model outputs, extracted answers, golden answers, and structured context.

### Step 2: Generate Critic Annotations via Majority Voting

Two annotation scripts are provided — one using the **DeepSeek API** and one using a **local Qwen model**. Both use majority voting over multiple samples to produce robust critic labels.

**Option A: DeepSeek API**

```bash
export DEEPSEEK_API_KEY="your-api-key"

python supervision/llm_as_judge_ds.py \
    --in_csv hotpotqa_trainset_inference_results_v1.csv \
    --out_csv hotpotqa_trainset_critic_labels_ds.csv \
    --model_name deepseek-chat \
    --critic_num_votes 3 \
    --judge_num_votes 3
```

**Option B: Local Qwen Model**

```bash
python supervision/llm_as_judge_qwen.py \
    --in_csv hotpotqa_trainset_inference_results_v1.csv \
    --out_csv hotpotqa_trainset_critic_labels_qwen.csv \
    --model_path /path/to/Qwen2.5-14B-Instruct \
    --load_in_4bit \
    --critic_num_votes 3 \
    --judge_num_votes 3
```

Both scripts produce a CSV with `critique`, `keywords`, and `llm_judge` columns containing structured critic annotations.

### Step 3: Build Training Parquet

Convert the annotated CSV into the parquet format required by the VERL training framework:

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

### Step 4: Two-Stage RL Training

Run the two-stage GRPO training with using the VERL framework:

```bash
# Set required paths
export BASE_MODEL=/path/to/base_model
export LORA_PATH=/path/to/lora_adapter
export TRAIN_FILE=./data/critic/train_critic.parquet
export VAL_FILE=./data/critic/val_critic.parquet
export REWARD_DIR=./train
export EXPERIMENT_NAME=critic-r1-exp

# Stage 1: CJA (default)
# Stage 2: DQA
bash train/train_critic.sh
```

The two stages are controlled by environment variables:
- `STEP1=true` — CJA (penalizes over-aggressive incorrect judgments)
- `STEP2=true` — DQA (rewards precise location, reason, and fix predictions)

By default, the script runs Stage 2 only. Set both to `true` to run the full two-stage curriculum.

### Step 5: Extract LoRA Adapter

After training completes, extract the LoRA adapter from the FSDP checkpoint:

```bash
python train/merge_lora.py \
    --checkpoint_path /path/to/checkpoint/global_step_xxx \
    --base_model_path /path/to/base_model \
    --target_dir ./extracted_lora
```

This saves the standalone LoRA adapter weights to `--target_dir`, which can then be loaded for inference or evaluation.

## Evaluation

The evaluation pipeline has four stages: (1) run inference on QA benchmarks, (2) generate critic outputs for each trajectory, (3) apply critic feedback to refine incorrect answers, and (4) compute final metrics on both the original and refined outputs.

### Stage 1: Inference on QA Benchmarks

Each dataset has its own inference script. Key configurable parameters are set at the top of each script:

| Script | Dataset | Data File |
|--------|---------|------------|
| `infer/infer_nq.py` | Natural Questions (dev) | `NQ-open.dev.jsonl` |
| `infer/infer_trivia.py` | TriviaQA (val) | `trivia_qa_val.parquet` |
| `infer/infer_hotpot.py` | HotpotQA (dev) | `hotpot_dev_distractor_v1.json` |
| `infer/infer_popqa.py` | PopQA (test) | `test.tsv` |
| `infer/infer_asqa.py` | ASQA (dev) | `devset.parquet` |

**Example — HotpotQA:**

```bash
python infer/infer_hotpot.py \
    --suffix my_experiment
```

Before running, adjust the configurable variables at the top of the script as needed:

```python
num_questions = 10000                  # number of samples to process
model_id = "your-org/your-model"       # HuggingFace model or local path
max_search_calls = 1                   # max retrieval calls per question
max_new_tokens = 1024                  # max generated tokens per step
do_sample = True                       # use sampling (vs. greedy)
temperature = 0.7                      # sampling temperature
top_p = 0.9                            # nucleus sampling threshold
retriever_url = "http://127.0.0.1:8000/retrieve"  # Search-R1 retrieval endpoint
```

Other datasets follow the same pattern — run the script directly after configuring these variables:

```bash
python infer/infer_nq.py
python infer/infer_trivia.py
python infer/infer_hotpot.py --suffix my_exp
python infer/infer_popqa.py
python infer/infer_asqa.py
```

Each script outputs a CSV (e.g., `nq_dev_inference_results.csv`) with columns: `question`, `model_output`, `extracted_answer`, `golden_answer`.

### Stage 2: Generate Critic Outputs

Use the trained critic model (base + LoRA, or a full model) to audit each trajectory and produce structured critiques:

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

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--csv_path` | *(required)* | Input CSV from Stage 1 |
| `--out_csv` | *(required)* | Output CSV with critic column appended |
| `--base_model` | `Qwen/Qwen2.5-3B-Instruct` | Base model for tokenizer/weights |
| `--lora_dir` | `""` | Path to LoRA adapter directory |
| `--use_lora` | `False` | Enable LoRA adapter loading |
| `--model_path` | `""` | Full HF model path (overrides base+LoRA) |
| `--output_col` | `3b_critique` | Name of the output critique column |
| `--question_col` | `question` | Question column name |
| `--traj_col` | `trajectory` | Full trajectory/model output column |
| `--extracted_answer_col` | `extracted_answer` | Extracted answer column |
| `--gold_col` | `golden_answer` | Golden answer column |
| `--critique_col` | `critique` | Existing critique column (if any) |
| `--llm_judge_col` | `llm_judge` | Existing LLM judge column (if any) |
| `--max_new_tokens` | `256` | Max tokens for critic generation |
| `--n` | `-1` (all) | Process only first N rows |
| `--save_every` | `20` | Auto-save every N newly filled rows |
| `--print_every` | `20` | Print progress every N rows |
| `--resume` | `False` | Skip rows with non-empty output |

### Stage 3: Apply Critic Feedback & Refine

Run the refinement loop: when the critic verdict is `INCORRECT`, the script triggers a fresh generation with the critique as guidance, then measures whether the answer improved:

```bash
python eval/evaluate_critic_feedback.py \
    --input_csv_path nq_dev_with_critique.csv \
    --output_csv_path nq_dev_refined.csv \
    --stats_json_path nq_dev_refined_stats.json \
    --model_path /path/to/your/base-model \
    --n 500 \
    --local_files_only
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input_csv_path` | *(required)* | CSV with `3b_critique` (or `critique`) column |
| `--output_csv_path` | *(required)* | Output CSV with refinement results |
| `--stats_json_path` | *(required)* | Aggregated statistics JSON |
| `--model_path` | *(required)* | QA model path for refinement generation |
| `--n` | `-1` (all) | Process only first N rows |
| `--local_files_only` | `False` | Only use local model files |

The output CSV includes:
- `critique_verdict` — extracted verdict (CORRECT / INCORRECT / UNSURE)
- `old_correct` / `final_correct` — correctness before and after refinement
- `used_refine` — whether refinement was triggered
- `improved_after_refine` / `harmed_after_refine` — directional change flags

The stats JSON reports error detection precision/recall, correction success rate, and the net improvement breakdown.

### Stage 4: Compute Final Metrics

Run the full metric suite (F1, SBERT similarity, LLM judge) on the final outputs:

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

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input_path` | *(required)* | Input CSV from Stage 3 (or Stage 1) |
| `--output_path` | `{input}_eval.csv` | Detailed per-sample results |
| `--summary_path` | `{input}_summary.json` | Aggregated metrics JSON |
| `--answer_col` | `extracted_answer` | Column with model predictions |
| `--gold_col` | `golden_answer` | Column with ground-truth answers |
| `--question_col` | `question` | Column with questions (for LLM judge) |
| `--sbert_model_name` | `all-MiniLM-L6-v2` | SBERT model for semantic similarity |
| `--judge_model_path` | `Qwen/Qwen2.5-7B-Instruct` | LLM judge model path |
| `--limit` | `None` | Only evaluate first N rows |
| `--save_every` | `10` | Save intermediate results every N rows |
| `--resume` | `False` | Resume from existing output CSV |

The summary JSON reports: F1, Precision, Recall, SBERT cosine similarity, LLM judge accuracy.

## Project Structure

```
├── infer/                  # Inference scripts for QA benchmarks
│   ├── infer_nq.py         #   Natural Questions
│   ├── infer_trivia.py     #   TriviaQA
│   ├── infer_hotpot.py     #   HotpotQA
│   ├── infer_popqa.py      #   PopQA
│   └── infer_asqa.py       #   ASQA
├── supervision/            # Training data generation & annotation
│   ├── hotpot_train.py     #   Trajectory + context generation
│   ├── llm_as_judge_ds.py  #   DeepSeek-based majority-voting annotation
│   └── llm_as_judge_qwen.py#   Qwen-based majority-voting annotation
├── train/                  # Training pipeline
│   ├── gen_critic_train.py #   CSV → parquet conversion
│   ├── critic_reward.py    #   Two-stage reward function
│   ├── train_critic.sh     #   PPO training launch script
│   └── merge_lora.py       #   LoRA adapter extraction
└── eval/                   # Evaluation
    ├── evaluate_metrics.py #   F1, SBERT, LLM-Judge metrics
    ├── evaluate_lora_3b.py #   LoRA model evaluation with critic
    └── evaluate_critic_feedback.py  # Critic feedback refinement eval
```

