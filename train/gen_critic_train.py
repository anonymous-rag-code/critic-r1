import os
import re
import ast
import argparse
from typing import List, Any, Dict

import pandas as pd
import datasets


# -----------------------------
# Utilities
# -----------------------------
def extract_last_answer(traj: str) -> str:
    if traj is None:
        return ""
    m = re.findall(r"<answer>(.*?)</answer>", str(traj), flags=re.DOTALL)
    return m[-1].strip() if m else ""


def parse_content_titles_field(x: Any) -> List[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    return [t.strip() for t in str(x).split(",") if t.strip()]
    
def parse_golden_answer_field(g: Any) -> List[str]:
    """
    Compatible with:
      - "ans1 ||| ans2"
      - "['a','b']"
      - "{'target': ['a']}"
      - "{'target': array([\"Arthur's Magazine\"], dtype=object)}"
      - plain string
    """
    if g is None or (isinstance(g, float) and pd.isna(g)):
        return []
    s = str(g).strip()
    if not s:
        return []

    if "|||" in s:
        return [x.strip() for x in s.split("|||") if x.strip()]

    # python literal list/dict
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
        if isinstance(obj, dict):
            for k in ["target", "answers", "answer", "gold"]:
                if k in obj:
                    v = obj[k]
                    if isinstance(v, list):
                        return [str(x).strip() for x in v if str(x).strip()]
                    if isinstance(v, str):
                        return [v.strip()] if v.strip() else []
    except Exception:
        pass

    # numpy array repr
    m = re.search(r"array\(\[(.*?)\]\s*,\s*dtype=.*?\)", s, flags=re.DOTALL)
    if m:
        inside = m.group(1)
        items = re.findall(r"['\"](.*?)['\"]", inside)
        items = [x.strip() for x in items if x.strip()]
        if items:
            return items

    # fallback: quoted strings
    items = re.findall(r"['\"](.*?)['\"]", s)
    items = [x.strip() for x in items if x.strip()]
    if items:
        bad = {"target", "dtype", "object", "array"}
        items2 = [x for x in items if x.lower() not in bad]
        return items2 if items2 else items

    return [s]



# -----------------------------
# Critic prompt template
# (keep it simple & stable for dataset building)
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



def build_critic_prompt(question: str, trajectory: str) -> str:
    # NOTE: verl dataset’s "prompt" usually only contains user content.
    # If you want a real "system" role later, you can put SYSTEM_PROMPT into a separate role at runtime.
    # Here we keep everything in user content for maximum compatibility.
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"{CRITIC_INSTRUCTION}\n"
        f"=== QUESTION ===\n{question}\n\n"
        f"=== TRAJECTORY ===\n{trajectory}\n"
    )


# -----------------------------
# Main: build train_critic.parquet
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./data/critic")
    parser.add_argument("--out_name", type=str, default="train_critic.parquet")
    parser.add_argument("--n", type=int, default=-1, help="use first n rows; -1 means all")
    parser.add_argument("--question_col", type=str, default="question")
    parser.add_argument("--traj_col", type=str, default="trajectory")
    parser.add_argument("--gold_col", type=str, default="golden_answer")
    parser.add_argument("--data_source", type=str, default="critic_r1")
    parser.add_argument("--llm_judge_col", type=str, default="llm_judge")
    parser.add_argument("--critique_col", type=str, default="critique")
    parser.add_argument("--keywords_col", type=str, default="keywords")
    parser.add_argument("--content_col", type=str, default="content")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    if args.n is not None and args.n > 0:
        df = df.head(args.n).copy()

    # Column fallback (in case you still have 'input' instead of 'question')
    if args.question_col not in df.columns and "input" in df.columns:
        args.question_col = "input"

    for col in [args.question_col, args.traj_col, args.gold_col]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {args.csv_path}. Got columns: {list(df.columns)}")

    records: List[Dict[str, Any]] = []
    for i, row in df.iterrows():
        q = "" if pd.isna(row[args.question_col]) else str(row[args.question_col])
        traj = "" if pd.isna(row[args.traj_col]) else str(row[args.traj_col])
        gold_raw = row[args.gold_col]

        gold_list = parse_golden_answer_field(gold_raw)
        last_answer = extract_last_answer(traj)
        # Build critic prompt (trajectory used VERBATIM)
        prompt_text = build_critic_prompt(q, traj)

        # Ground-truth payload kept for later reward/SFT label design
        solution = {
            "target": gold_list,          # gold answers (list[str])
            "last_answer": last_answer,   # extracted <answer>
        }

        llm_judge = str(row.get(args.llm_judge_col, "")) if not pd.isna(row.get(args.llm_judge_col, "")) else ""
        standard_critique = str(row.get(args.critique_col, "")) if not pd.isna(row.get(args.critique_col, "")) else ""
        keywords = str(row.get(args.keywords_col, "")) if not pd.isna(row.get(args.keywords_col, "")) else ""
        raw_content = row.get(args.content_col, "") if args.content_col in df.columns else ""
        context_titles = parse_content_titles_field(raw_content)
        data = {
            "data_source": args.data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": prompt_text,
                }
            ],
            "ability": "critic-verification",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution,
            },
            "extra_info": {
                "index": int(i),
                "llm_judge": llm_judge,
                "standard_critique": standard_critique, 
                "keywords": keywords,
                "context_titles": context_titles,
            },
        }
        records.append(data)

    ds = datasets.Dataset.from_list(records)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, args.out_name)
    ds.to_parquet(out_path)

    print("================================")
    print(f"CSV:              {args.csv_path}")
    print(f"Rows written:     {len(ds)}")
    print(f"Output parquet:   {out_path}")
    print(f"Question col:     {args.question_col}")
    print(f"Trajectory col:   {args.traj_col}")
    print(f"Golden col:       {args.gold_col}")
    print(f"Keywords col:     {args.keywords_col}")
    print("================================")


if __name__ == "__main__":
    main()