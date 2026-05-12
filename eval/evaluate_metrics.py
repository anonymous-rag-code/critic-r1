import os
import re
import ast
import json
import argparse
import string
from collections import Counter
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def normalize_text(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def normalize_answer(s: Any) -> str:
    s = normalize_text(s).lower()

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    s = remove_articles(s)
    s = remove_punc(s)
    s = white_space_fix(s)
    return s


def safe_literal_eval(x: Any) -> Any:
    if not isinstance(x, str):
        return x
    x = x.strip()
    if not x:
        return x
    try:
        return ast.literal_eval(x)
    except Exception:
        return x


def split_pipe_answers(text: str) -> List[str]:
    parts = [normalize_text(x) for x in str(text).split("|||")]
    return [x for x in parts if x]


def extract_answers_from_obj(obj: Any) -> List[str]:
    if obj is None:
        return []

    if isinstance(obj, str):
        s = obj.strip()
        if not s:
            return []

        parsed = safe_literal_eval(s)
        if parsed is not s:
            return extract_answers_from_obj(parsed)

        if "|||" in s:
            return split_pipe_answers(s)

        return [normalize_text(s)] if normalize_text(s) else []

    if isinstance(obj, dict):
        preferred_keys = [
            "target", "targets", "answer", "answers", "gold", "gold_answer",
            "golden_answer", "aliases", "normalized_aliases", "possible_answers",
            "short_answers", "reference", "references", "value",
        ]
        collected = []
        for k in preferred_keys:
            if k in obj:
                collected.extend(extract_answers_from_obj(obj[k]))
        if not collected:
            for v in obj.values():
                collected.extend(extract_answers_from_obj(v))
        return collected

    if isinstance(obj, (list, tuple, set)):
        collected = []
        for item in obj:
            collected.extend(extract_answers_from_obj(item))
        return collected

    s = normalize_text(obj)
    return [s] if s else []


def parse_golden_answer_field(g: Any) -> List[str]:
    if g is None:
        return []
    obj = safe_literal_eval(g)
    answers = extract_answers_from_obj(obj)

    seen = set()
    final = []
    for x in answers:
        x = normalize_text(x)
        if x and x not in seen:
            seen.add(x)
            final.append(x)
    return final




def compute_str_em_from_row(qa_pairs: Any, prediction: str) -> Tuple[float, int]:
    if qa_pairs is None:
        return 0.0, 0

    qa_pairs = safe_literal_eval(qa_pairs)
    if not isinstance(qa_pairs, (list, tuple)) or len(qa_pairs) == 0:
        return 0.0, 0

    local_scores = []
    normalized_prediction = normalize_answer(prediction)

    for qa in qa_pairs:
        answers = []
        if isinstance(qa, dict):
            if "answers" in qa:
                answers = extract_answers_from_obj(qa["answers"])
            elif "short_answers" in qa:
                answers = extract_answers_from_obj(qa["short_answers"])

        found = 0.0
        for ans in answers:
            normalized_ans = normalize_answer(ans)
            if normalized_ans and normalized_ans in normalized_prediction:
                found = 1.0
                break

        local_scores.append(found)

    if not local_scores:
        return 0.0, 0

    row_str_em = float(np.mean(local_scores))
    row_str_hit = int(row_str_em == 1.0)
    return row_str_em, row_str_hit


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score_official(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    zero_metric = (0.0, 0.0, 0.0)

    if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return zero_metric
    if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return zero_metric

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return zero_metric

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return float(f1), float(precision), float(recall)


def max_em_over_gold(prediction: str, gold_answers: List[str]) -> Tuple[float, str]:
    if not gold_answers:
        return 0.0, ""
    scores = [(exact_match_score(prediction, g), g) for g in gold_answers]
    best_score, best_gold = max(scores, key=lambda x: x[0])
    return float(best_score), best_gold


def max_f1_over_gold(prediction: str, gold_answers: List[str]) -> Tuple[float, float, float, str]:
    if not gold_answers:
        return 0.0, 0.0, 0.0, ""
    scores = [(f1_score_official(prediction, g), g) for g in gold_answers]
    (best_f1, best_prec, best_recall), best_gold = max(scores, key=lambda x: x[0][0])
    return float(best_f1), float(best_prec), float(best_recall), best_gold


class SBERTScorer:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)

    def score(self, prediction: str, gold_answers: List[str]) -> Tuple[float, str]:
        if not gold_answers:
            return 0.0, ""

        pred = normalize_text(prediction)
        golds = [normalize_text(g) for g in gold_answers]

        if not pred:
            return 0.0, golds[0]

        pred_emb = self.model.encode([pred], convert_to_numpy=True, normalize_embeddings=True)
        gold_embs = self.model.encode(golds, convert_to_numpy=True, normalize_embeddings=True)

        sims = cosine_similarity(pred_emb, gold_embs)[0]
        best_idx = int(np.argmax(sims))
        return float(sims[best_idx]), golds[best_idx]


JUDGE_SYSTEM_PROMPT = """You are a helpful and fair QA evaluator.

Given:
- A question
- A predicted answer
- Reference answers

Your task is to decide whether the predicted answer is correct.

Rules:
- Accept semantically equivalent answers.
- Short answers are acceptable if correct.
- Ignore extra explanation if the answer is clearly correct.
- Be lenient and focus on correctness, not wording.

Output ONLY one of the following (no extra text):

<judge>CORRECT</judge>
or
<judge>INCORRECT</judge>
"""


def build_judge_prompt(question: str, prediction: str, gold_answers: List[str]) -> str:
    gold_text = "\n".join([f"- {g}" for g in gold_answers]) if gold_answers else "- "
    return f"""Question:
{normalize_text(question)}

Predicted Answer:
{normalize_text(prediction)}

Reference Answer(s):
{gold_text}

Now judge whether the predicted answer is correct.
"""


def extract_judge_result(text: str) -> str:
    text = text or ""
    m = re.search(r"<judge>\s*(CORRECT|INCORRECT)\s*</judge>", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper().strip()
    return "INCORRECT"


class LLMJudge:
    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        device_map: str = "auto",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    @torch.no_grad()
    def judge(self, question: str, prediction: str, gold_answers: List[str]) -> Tuple[str, str]:
        user_prompt = build_judge_prompt(question, prediction, gold_answers)

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            text = JUDGE_SYSTEM_PROMPT + "\n\n" + user_prompt

        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
        gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        judge = extract_judge_result(gen_text)
        return judge, gen_text


def read_csv_auto(path: str) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err


def find_column(df: pd.DataFrame, candidates: List[str]) -> str:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return ""


def infer_question_col(df: pd.DataFrame) -> str:
    return find_column(df, ["question", "input", "query", "prompt"])


def save_progress(processed_rows: List[dict], output_path: str, summary_path: str,
                  input_path: str, answer_col: str, gold_col: str, question_col: str) -> None:
    out_df = pd.DataFrame(processed_rows)
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    summary = {
        "input_file": input_path,
        "num_samples": int(len(out_df)),
        "answer_col": answer_col,
        "gold_col": gold_col,
        "question_col": question_col,
        "avg_em_score": float(out_df["em_score"].mean()) if "em_score" in out_df.columns and len(out_df) > 0 else 0.0,
        "avg_f1_score": float(out_df["f1_score"].mean()) if "f1_score" in out_df.columns and len(out_df) > 0 else 0.0,
        "avg_precision_score": float(out_df["precision_score"].mean()) if "precision_score" in out_df.columns and len(out_df) > 0 else 0.0,
        "avg_recall_score": float(out_df["recall_score"].mean()) if "recall_score" in out_df.columns and len(out_df) > 0 else 0.0,
        "avg_sbert_score": float(out_df["sbert_score"].mean()) if "sbert_score" in out_df.columns and len(out_df) > 0 else 0.0,
        "llm_acc": float(out_df["llm_judge_binary"].mean()) if "llm_judge_binary" in out_df.columns and out_df["llm_judge_binary"].notna().any() else 0.0,
    }

    if "asqa_str_em_score" in out_df.columns:
        summary["asqa_str_em"] = float(out_df["asqa_str_em_score"].mean()) if len(out_df) > 0 else 0.0
    if "asqa_str_hit" in out_df.columns:
        summary["asqa_str_hit"] = float(out_df["asqa_str_hit"].mean()) if len(out_df) > 0 else 0.0

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def evaluate_file(
    input_path: str,
    output_path: str,
    summary_path: str,
    sbert_model_name: str,
    judge_model_path: str,
    answer_col: str = "model_output",
    gold_col: str = "golden_answer",
    question_col: str = None,
    limit: int = None,
    save_every: int = 10,
    resume: bool = False,
) -> None:
    print(f"[INFO] Loading: {input_path}")
    df = read_csv_auto(input_path)

    if question_col is None:
        question_col = infer_question_col(df)

    if answer_col not in df.columns:
        raise ValueError(f"Answer column '{answer_col}' not found in {input_path}. Available: {list(df.columns)}")
    if gold_col not in df.columns:
        raise ValueError(f"Gold column '{gold_col}' not found in {input_path}. Available: {list(df.columns)}")

    if question_col and question_col not in df.columns:
        print(f"[WARN] question_col '{question_col}' not found. Use empty question for LLM judge.")
        question_col = ""

    print(f"[INFO] Using answer column: {answer_col}")
    print(f"[INFO] Using gold column: {gold_col}")
    if question_col:
        print(f"[INFO] Using question column: {question_col}")
    else:
        print("[WARN] No valid question column found. LLM judge will receive an empty question.")

    if limit is not None:
        df = df.iloc[:limit].copy()

    print("[INFO] Loading SBERT model...")
    sbert_scorer = SBERTScorer(model_name=sbert_model_name)

    print("[INFO] Loading LLM judge model...")
    llm_judge = LLMJudge(model_path=judge_model_path)

    processed_rows: List[dict] = []
    start_idx = 0

    if resume and os.path.exists(output_path):
        print(f"[INFO] Resume enabled. Loading existing output from: {output_path}")
        existing_df = pd.read_csv(output_path)
        processed_rows = existing_df.to_dict("records")
        start_idx = len(processed_rows)
        print(f"[INFO] Found {start_idx} processed rows. Will continue from row {start_idx}.")

    iterator = tqdm(
        df.iloc[start_idx:].iterrows(),
        total=len(df) - start_idx,
        desc=f"Evaluating {os.path.basename(input_path)}",
    )

    for i, (_, row) in enumerate(iterator, start=start_idx):
        answer = normalize_text(row.get(answer_col, ""))
        gold_answers = parse_golden_answer_field(row.get(gold_col, ""))
        question = normalize_text(row.get(question_col, "")) if question_col else ""


        if "qa_pairs" in df.columns:
            row_str_em, row_str_hit = compute_str_em_from_row(row.get("qa_pairs", None), answer)
        else:
            row_str_em, row_str_hit = 0.0, 0

        em, best_em_gold = max_em_over_gold(answer, gold_answers)
        f1, prec, recall, best_f1_gold = max_f1_over_gold(answer, gold_answers)
        sbert_score, best_sbert_gold = sbert_scorer.score(answer, gold_answers)

        judge_label, judge_raw = llm_judge.judge(
            question=question,
            prediction=answer,
            gold_answers=gold_answers,
        )
        llm_binary = 1 if judge_label == "CORRECT" else 0

        row_result = row.to_dict()
        row_result["parsed_gold_answers"] = gold_answers

        row_result["best_em_gold"] = best_em_gold
        row_result["em_score"] = em

        row_result["best_f1_gold"] = best_f1_gold
        row_result["f1_score"] = f1
        row_result["precision_score"] = prec
        row_result["recall_score"] = recall

        row_result["best_sbert_gold"] = best_sbert_gold
        row_result["sbert_score"] = sbert_score

        row_result["llm_judge"] = judge_label
        row_result["llm_judge_binary"] = llm_binary
        row_result["llm_judge_raw"] = judge_raw

        if "qa_pairs" in df.columns:
            row_result["asqa_str_em_score"] = row_str_em
            row_result["asqa_str_hit"] = row_str_hit

        processed_rows.append(row_result)

        if len(processed_rows) % save_every == 0:
            save_progress(
                processed_rows=processed_rows,
                output_path=output_path,
                summary_path=summary_path,
                input_path=input_path,
                answer_col=answer_col,
                gold_col=gold_col,
                question_col=question_col,
            )
            print(f"[INFO] Intermediate saved {len(processed_rows)} rows to: {output_path}")

    save_progress(
        processed_rows=processed_rows,
        output_path=output_path,
        summary_path=summary_path,
        input_path=input_path,
        answer_col=answer_col,
        gold_col=gold_col,
        question_col=question_col,
    )
    print(f"[INFO] Saved detailed results to: {output_path}")
    print(f"[INFO] Saved summary to: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        print(f.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Input CSV path")
    parser.add_argument("--output_path", type=str, default=None, help="Detailed output CSV path")
    parser.add_argument("--summary_path", type=str, default=None, help="Summary JSON path")

    parser.add_argument("--answer_col", type=str, default="extracted_answer", help="Prediction column name")
    parser.add_argument("--gold_col", type=str, default="golden_answer", help="Gold answer column name")
    parser.add_argument("--question_col", type=str, default="question", help="Question column name")

    parser.add_argument(
        "--sbert_model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name/path",
    )
    parser.add_argument(
        "--judge_model_path",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Local judge model path",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate first N rows")
    parser.add_argument("--save_every", type=int, default=10, help="Save every N rows")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output CSV")
    args = parser.parse_args()

    if args.output_path is None:
        stem = os.path.splitext(args.input_path)[0]
        args.output_path = stem + "_eval.csv"

    if args.summary_path is None:
        stem = os.path.splitext(args.input_path)[0]
        args.summary_path = stem + "_summary.json"

    evaluate_file(
        input_path=args.input_path,
        output_path=args.output_path,
        summary_path=args.summary_path,
        sbert_model_name=args.sbert_model_name,
        judge_model_path=args.judge_model_path,
        answer_col=args.answer_col,
        gold_col=args.gold_col,
        question_col=args.question_col,
        limit=args.limit,
        save_every=args.save_every,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()