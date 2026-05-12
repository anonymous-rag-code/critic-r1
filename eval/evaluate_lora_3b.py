import re
import ast
import argparse
from typing import List, Tuple, Any, Dict, Optional
import os
import pandas as pd
import transformers
import torch

from peft import PeftModel


# -----------------------------
# Critic prompting (NO compression)
# -----------------------------
SYSTEM_PROMPT = (
    "You are an external critic for a retrieval-augmented QA system. "
    "You will audit a trajectory produced by another QA model. "
    "The trajectory may contain reasoning, search steps, retrieved information, and a final answer. "

    "IMPORTANT RULES:\n"
    "1. You MUST use <information> as the ONLY source of knowledge.\n"
    "2. DO NOT use your own parametric knowledge or world knowledge.\n"
    "3. If the <information> does not contradict the answer, you should assume the answer is acceptable.\n"
    "4. Do NOT invent missing facts from your own knowledge.\n"

    "Evaluation policy:\n"
    "- If the reasoning and answer are consistent with <information>, judge CORRECT.\n"
    "- Only judge INCORRECT if there is a clear contradiction or a clear entity mismatch.\n"
    "- If the evidence is insufficient but not contradictory, judge CORRECT or UNSURE.\n"
)

CRITIC_INSTRUCTION = """
Input:
- QUESTION
- TRAJECTORY with tags: <think> <search> <information> <answer>

Output (STRICT, one line, in this exact order):
<verdict>...</verdict><location>...</location><reason>...</reason><fix>...</fix>

Allowed values:
- verdict: CORRECT | INCORRECT | UNSURE
- location: none | answer | information:DocK | search:stepK | think:stepK

Constraints:
- Output MUST start with "<verdict>" (no leading text).
- Keep <reason> <= 25 words.
- Keep <fix> <= 20 words.
- If verdict is CORRECT: location must be "none" and fix must be "keep".
- If verdict is INCORRECT or UNSURE: location must NOT be "none".
- If there is no clear contradiction with <information>, prefer CORRECT.

Example (format only):
<verdict>INCORRECT</verdict><location>information:Doc1</location><reason>Doc1 contradicts the final answer.</reason><fix>search: retrieve correct entity</fix>
""".strip()



class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        self.target_ids = [tokenizer.encode(s, add_special_tokens=False) for s in target_sequences]
        self.target_lens = [len(x) for x in self.target_ids]

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] < min(self.target_lens):
            return False
        for tid, tlen in zip(self.target_ids, self.target_lens):
            target = torch.as_tensor(tid, device=input_ids.device)
            if torch.equal(input_ids[0, -tlen:], target):
                return True
        return False


def run_critic_once(
    question: str,
    trajectory_raw: str,
    tokenizer,
    model,
    device,
    max_new_tokens: int = 128,
    return_prompt: bool = False,
) -> Tuple[str, str]:
    """
    IMPORTANT: trajectory_raw is used VERBATIM (no modification).
    """
    user_prompt = f"""{CRITIC_INSTRUCTION}

=== QUESTION ===
{question}

=== TRAJECTORY ===
{trajectory_raw}
"""

    if getattr(tokenizer, "chat_template", None):
        #messages = [
        #    {"role": "system", "content": SYSTEM_PROMPT},
        #    {"role": "user", "content": user_prompt},
        #]
        messages = [
            {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + user_prompt},
        ]
        final_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    else:
        final_prompt = SYSTEM_PROMPT + "\n\n" + user_prompt

    # stop at </fix>
    target_sequences = ["</fix>", "</fix>\n", "</fix>\n\n", "</fix>\r\n"]
    stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])

    input_ids = tokenizer.encode(final_prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        stopping_criteria=stopping_criteria,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )

    gen = outputs[0][input_ids.shape[1]:]
    out = tokenizer.decode(gen, skip_special_tokens=False).strip()

    # hard truncate after </fix>
    if "</fix>" in out:
        out = out.split("</fix>")[0] + "</fix>"
        out = out.strip()

    if return_prompt:
        return out, final_prompt
    return out, ""


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="/path/to/input.csv")
    parser.add_argument("--out_csv", type=str, default="/path/to/output.csv")

    parser.add_argument("--n", type=int, default=-1, help="use first n rows; -1 means all")
    parser.add_argument("--question_col", type=str, default="question")
    parser.add_argument("--traj_col", type=str, default="trajectory")
    parser.add_argument("--extracted_answer_col", type=str, default="extracted_answer")
    parser.add_argument("--gold_col", type=str, default="golden_answer")
    parser.add_argument("--critique_col", type=str, default="critique")
    parser.add_argument("--llm_judge_col", type=str, default="llm_judge")
    # base + lora
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument(
        "--lora_dir",
        type=str,
        default="",
        help="Path to the LoRA adapter directory. Required when --use_lora is set.",
    )
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA adapter")
    parser.add_argument("--output_col", type=str, default="3b_critique", help="Name of output critique column")

    parser.add_argument("--max_new_tokens", type=int, default=256)

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--print_every", type=int, default=20)

    parser.add_argument(
    "--model_path",
    type=str,
    default="",
    help="If set, load a full HF model directly from this path. "
         "Otherwise fallback to base_model (+ LoRA if --use_lora).",
)
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.csv_path)
    if args.n is not None and args.n > 0:
        df = df.head(args.n).copy()
    else:
        df = df.copy()

    # Check required columns.
    need_cols = [
        args.question_col,
        args.traj_col,
        args.extracted_answer_col,
        args.gold_col,
    ]
    
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Got columns: {list(df.columns)}")
    
    has_critique = args.critique_col in df.columns
    has_llm_judge = args.llm_judge_col in df.columns

    if args.resume and os.path.exists(args.out_csv):
        old = pd.read_csv(args.out_csv)
        if args.output_col in old.columns and len(old) == len(df):
            df[args.output_col] = old[args.output_col]
            print(f"[resume] loaded existing {args.out_csv}, will skip non-empty {args.output_col} rows.")
        else:
            print(f"[resume] {args.out_csv} exists but schema/len mismatch, ignore resume.")

    # Create the output column if it does not exist.
    if args.output_col not in df.columns:
        df[args.output_col] = ""

    # -----------------------------
    # Load model (backward compatible)
    # Priority:
    #   1) --model_path            -> load full HF model directly
    #   2) --use_lora             -> load base + LoRA adapter
    #   3) fallback               -> load base only
    # -----------------------------
    if args.model_path and str(args.model_path).strip():
        print(f"[model] loading full model from: {args.model_path}")

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_path,
            local_files_only=True,
            trust_remote_code=True,
        )

        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True,
        )

    else:
        print(f"[model] loading tokenizer from base model: {args.base_model}")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.base_model,
            local_files_only=True,
            trust_remote_code=True,
        )

        base = transformers.AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True,
        )

        if args.use_lora:
            print(f"[model] loading base + LoRA from: {args.lora_dir}")
            model = PeftModel.from_pretrained(base, args.lora_dir, is_trainable=False)
        else:
            print("[model] loading base model only (LoRA disabled)")
            model = base
    
    model.eval()

    done = 0
    total = len(df)

    for i in range(total):
        # Skip finished rows when resuming
        if isinstance(df.at[i, args.output_col], str) and df.at[i, args.output_col].strip():
            continue

        q = "" if pd.isna(df.at[i, args.question_col]) else str(df.at[i, args.question_col])
        traj_raw = "" if pd.isna(df.at[i, args.traj_col]) else str(df.at[i, args.traj_col])

        try:
            with torch.no_grad():
                critic_out, _ = run_critic_once(
                    q, traj_raw, tokenizer, model, device,
                    max_new_tokens=args.max_new_tokens,
                    return_prompt=False,
                )
        except Exception as e:
            # Save error messages for debugging failed rows.
            critic_out = f"[ERROR] {repr(e)}"

        df.at[i, args.output_col] = critic_out
        done += 1

        if args.print_every > 0 and done % args.print_every == 0:
            print(f"[progress] filled {done} rows (latest idx={i}) / total={total}")

        # Periodically save intermediate results.
        if args.save_every > 0 and done % args.save_every == 0:
            cols = [
                args.question_col,
                args.traj_col,
                args.extracted_answer_col,
                args.gold_col,
            ]
            
            if has_critique:
                cols.append(args.critique_col)
            
            if has_llm_judge:
                cols.append(args.llm_judge_col)
            
            cols.append(args.output_col)
            
            out_df = df[cols].copy()
            out_df.to_csv(args.out_csv, index=False, encoding="utf-8")
            print(f"[autosave] saved to {args.out_csv}")

    # Save selected output columns.
    cols = [
        args.question_col,
        args.traj_col,
        args.extracted_answer_col,
        args.gold_col,
    ]
    
    if has_critique:
        cols.append(args.critique_col)
    
    if has_llm_judge:
        cols.append(args.llm_judge_col)
    
    cols.append(args.output_col)
    
    out_df = df[cols].copy()

    rename_map = {
        args.question_col: "question",
        args.traj_col: "trajectory",
        args.extracted_answer_col: "extracted_answer",
        args.gold_col: "golden_answer",
    }
    
    if has_critique:
        rename_map[args.critique_col] = "critique"
    
    if has_llm_judge:
        rename_map[args.llm_judge_col] = "llm_judge"
    
    out_df = out_df.rename(columns=rename_map)

    out_df.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"[done] saved final csv: {args.out_csv} rows={len(out_df)}")


if __name__ == "__main__":
    main()