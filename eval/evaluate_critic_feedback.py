import transformers
import torch
import requests
import pandas as pd
import re
import ast
import json
from typing import Any, List, Dict, Tuple
import argparse

# -----------------------------
# Config
# -----------------------------

DEFAULT_INPUT_CSV_PATH = "xxx.csv"
DEFAULT_OUTPUT_CSV_PATH = "xxx.csv"
DEFAULT_STATS_JSON_PATH = "xxx.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

critique_col = "3b_critique"
#critique_col = "critique"

curr_eos = [151645, 151643]  # Qwen2.5 series
curr_search_template = "\n\n{output_text}<information>{search_results}</information>\n\n"


# -----------------------------
# Utilities
# -----------------------------
def extract_final_answer(text: str) -> str:
    pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
    matches = pattern.findall("" if text is None else str(text))
    if matches:
        return matches[-1].strip()
    return ""


def extract_verdict(text: str) -> str:
    if text is None:
        return ""
    m = re.search(
        r"<verdict>\s*(CORRECT|INCORRECT|UNSURE)\s*</verdict>",
        str(text),
        flags=re.IGNORECASE
    )
    if m:
        return m.group(1).upper()
    return ""


def count_tag_pairs(text: str, tag: str) -> Tuple[int, int]:
    if text is None:
        return 0, 0
    s = str(text)
    open_c = len(re.findall(fr"<{tag}>", s, flags=re.IGNORECASE))
    close_c = len(re.findall(fr"</{tag}>", s, flags=re.IGNORECASE))
    return open_c, close_c

def normalize_answer(text: str) -> str:
    text = "" if text is None else str(text).lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_answer_correct(pred: str, gold_list: List[str]) -> bool:
    pred_norm = normalize_answer(pred)
    if not pred_norm or not gold_list:
        return False

    for gold in gold_list:
        gold_norm = normalize_answer(gold)
        if pred_norm == gold_norm:
            return True

    return False
    


def is_valid_refined_output(full_output: str, final_answer: str) -> bool:
    """
    1. full_output must be non-empty.
    2. The output is considered valid as long as a non-empty <answer>...</answer> can be extracted.
    3. We do not enforce balanced think/search/information tags here.
    """
    if full_output is None:
        return False

    text = str(full_output).strip()
    ans = "" if final_answer is None else str(final_answer).strip()

    if not text:
        return False

    if not ans:
        return False

    return True


def parse_golden_answer_field(g: Any) -> List[str]:
    if g is None or (isinstance(g, float) and pd.isna(g)):
        return []

    if isinstance(g, dict):
        for k in ["target", "targets", "answers", "answer", "gold"]:
            if k in g:
                v = g[k]
                if isinstance(v, list):
                    return [str(x).strip() for x in v if str(x).strip()]
                if isinstance(v, str):
                    return [v.strip()] if v.strip() else []
        return []

    if isinstance(g, list):
        return [str(x).strip() for x in g if str(x).strip()]

    s = str(g).strip()
    if not s:
        return []

    if "|||" in s:
        return [x.strip() for x in s.split("|||") if x.strip()]

    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
        if isinstance(obj, dict):
            for k in ["target", "targets", "answers", "answer", "gold"]:
                if k in obj:
                    v = obj[k]
                    if isinstance(v, list):
                        return [str(x).strip() for x in v if str(x).strip()]
                    if isinstance(v, str):
                        return [v.strip()] if v.strip() else []
    except Exception:
        pass

    items = re.findall(r"['\"](.*?)['\"]", s)
    items = [x.strip() for x in items if x.strip()]
    if items:
        return items

    return [s]

    

def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


# -----------------------------
# Search-R1 generation utilities
# -----------------------------
class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        self.target_ids = [tokenizer.encode(s, add_special_tokens=False) for s in target_sequences]
        self.target_lengths = [len(t) for t in self.target_ids]

    def __call__(self, input_ids, scores, **kwargs):
        targets = [torch.as_tensor(tid, device=input_ids.device) for tid in self.target_ids]
        if input_ids.shape[1] < min(self.target_lengths):
            return False
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True
        return False


def get_query(text):
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL | re.IGNORECASE)
    matches = pattern.findall("" if text is None else str(text))
    if matches:
        return matches[-1].strip()
    return None


def search(query: str):
    payload = {
        "queries": [query],
        "topk": 5,
        "return_scores": True
    }
    results = requests.post(
        "http://127.0.0.1:8000/retrieve",
        json=payload,
        timeout=60
    ).json()["result"]

    def _passages2string(retrieval_result):
        format_reference = ""
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["document"]["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference

    return _passages2string(results[0])


def run_generation_loop(prompt, model, tokenizer, device, do_sample=False, temperature=0.7, top_p=0.9):
    target_sequences = [
        "</search>", " </search>", "</search>\n", " </search>\n",
        "</search>\n\n", " </search>\n\n"
    ]
    stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])

    print("\n\n################# [Start Reasoning + Searching] ##################\n\n")
    print(prompt)

    cnt = 0
    full_output = ""

    while True:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        attention_mask = torch.ones_like(input_ids)

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

        if do_sample:
            gen_kwargs.update(
                dict(
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                )
            )
        else:
            gen_kwargs.update(
                dict(
                    do_sample=False,
                )
            )

        outputs = model.generate(**gen_kwargs)

        generated_tokens = outputs[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # The model finishes generation directly.
        if outputs[0][-1].item() in curr_eos:
            print(output_text)
            full_output += output_text
            break

        decoded_all = tokenizer.decode(outputs[0], skip_special_tokens=True)
        tmp_query = get_query(output_text)
        if tmp_query:
            search_results = search(tmp_query)
        else:
            search_results = ""

        search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
        prompt += search_text
        full_output += search_text
        cnt += 1
        print(search_text)

        if cnt > 10:
            break

    final_answer = extract_final_answer(full_output)
    return full_output, final_answer


def refine_with_trajectory_and_critique(
    question,
    old_trajectory,
    old_answer,
    critique_text,
    model,
    tokenizer,
    device,
    max_retry=3,
):
    question = question.strip()
    if question and question[-1] != "?":
        question += "?"

    prompt = f"""Answer the given question.
You must conduct reasoning inside <think> and </think> first every time you get new information.
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>.
You can search as many times as you want.
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>.

You are also given:
1. A previous trajectory from an earlier attempt.
2. An external critique of that previous trajectory.

Important rules:
- The previous trajectory may contain mistakes.
- The previous final answer may be wrong.
- The external critique may also be wrong.
- Do NOT blindly trust the previous trajectory.
- Do NOT blindly trust the critique.
- Use the critique only as a hint about possible problems to check.
- You must re-solve the question with fresh reasoning instead of simply copying the previous answer.
- If the critique points out a possible issue, verify it by your own reasoning and search.
- If the critique is unsupported or mistaken, ignore it.
- Do not change your answer just because the critique suggests a change.
- Base your final answer on your own reasoning process and the retrieved information.
- You MUST end with exactly one final answer inside <answer> and </answer>.

Question: {question}

Previous trajectory:
{old_trajectory}

External critique:
{critique_text}
"""


    for attempt in range(max_retry):
        print(f"\n[Refine Attempt {attempt + 1}/{max_retry}]")

        # Use greedy decoding for the first attempt for stability.
        if attempt == 0:
            full_output, final_answer = run_generation_loop(
                prompt, model, tokenizer, device,
                do_sample=False
            )
        else:
            full_output, final_answer = run_generation_loop(
                prompt, model, tokenizer, device,
                do_sample=True, temperature=0.7, top_p=0.9
            )

        if is_valid_refined_output(full_output, final_answer):
            print("[Refine Output Valid]")
            return full_output, final_answer, True, f"refine_ok_attempt_{attempt + 1}"

        print("[Invalid refine output: empty / malformed / missing <answer>]")

    print("[Refine Failed After Retries -> Fallback To Old]")
    return old_trajectory, old_answer, False, "fallback_old_due_to_invalid_refine"


# -----------------------------
# Stats
# -----------------------------
def compute_binary_confusion_matrix(df: pd.DataFrame) -> Dict[str, int]:
    """
    Compare critic predictions with answer correctness:
    - predicted: critique_verdict in {CORRECT, INCORRECT}
    - actual: old_correct in {CORRECT, INCORRECT}

    UNSURE and EMPTY verdicts are excluded from this binary confusion matrix.
    """
    sub = df[df["critique_verdict"].isin(["CORRECT", "INCORRECT"])].copy()

    tp = int(((sub["critique_verdict"] == "INCORRECT") & (sub["old_correct"] == 0)).sum())
    fp = int(((sub["critique_verdict"] == "INCORRECT") & (sub["old_correct"] == 1)).sum())
    tn = int(((sub["critique_verdict"] == "CORRECT") & (sub["old_correct"] == 1)).sum())
    fn = int(((sub["critique_verdict"] == "CORRECT") & (sub["old_correct"] == 0)).sum())

    return {
        "TP_incorrect": tp,
        "FP_incorrect": fp,
        "TN_incorrect": tn,
        "FN_incorrect": fn,
        "num_binary_samples": int(len(sub)),
    }


def summarize_stats(df: pd.DataFrame) -> Dict[str, Any]:
    total = int(len(df))
    refined_count = int((df["used_refine"] == 1).sum())
    final_correct_count = int((df["final_correct"] == 1).sum())
    old_correct_count = int((df["old_correct"] == 1).sum())
    old_incorrect_count = int((df["old_correct"] == 0).sum())

    verdict_counts = {
        "CORRECT": int((df["critique_verdict"] == "CORRECT").sum()),
        "INCORRECT": int((df["critique_verdict"] == "INCORRECT").sum()),
        "UNSURE": int((df["critique_verdict"] == "UNSURE").sum()),
        "EMPTY": int((df["critique_verdict"] == "").sum()),
    }

    cm = compute_binary_confusion_matrix(df)
    tp = cm["TP_incorrect"]
    fp = cm["FP_incorrect"]
    tn = cm["TN_incorrect"]
    fn = cm["FN_incorrect"]

    error_detection_recall = safe_div(tp, tp + fn)
    error_detection_precision = safe_div(tp, tp + fp)
    error_detection_fpr = safe_div(fp, fp + tn)
    error_detection_acc = safe_div(tp + tn, tp + tn + fp + fn)

    triggered_df = df[df["used_refine"] == 1].copy()
    wrong_and_triggered_df = df[(df["old_correct"] == 0) & (df["used_refine"] == 1)].copy()
    wrong_old_df = df[df["old_correct"] == 0].copy()

    correction_success_after_trigger = safe_div(
        int((triggered_df["final_correct"] == 1).sum()),
        int(len(triggered_df))
    )

    correction_success_on_wrong_triggered = safe_div(
        int((wrong_and_triggered_df["final_correct"] == 1).sum()),
        int(len(wrong_and_triggered_df))
    )

    overall_correction_rate_on_wrong = safe_div(
        int((wrong_old_df["final_correct"] == 1).sum()),
        int(len(wrong_old_df))
    )

    improved_count = int(((df["old_correct"] == 0) & (df["final_correct"] == 1)).sum())
    harmed_count = int(((df["old_correct"] == 1) & (df["final_correct"] == 0)).sum())
    unchanged_correct_count = int(((df["old_correct"] == 1) & (df["final_correct"] == 1)).sum())
    unchanged_wrong_count = int(((df["old_correct"] == 0) & (df["final_correct"] == 0)).sum())

    summary = {
        "total_samples": total,
        "refined_samples": refined_count,
        "old_accuracy": safe_div(old_correct_count, total),
        "final_accuracy": safe_div(final_correct_count, total),
        "old_correct_count": old_correct_count,
        "old_incorrect_count": old_incorrect_count,
        "final_correct_count": final_correct_count,
        "verdict_counts": verdict_counts,

        "confusion_matrix_binary": {
            "rows_actual_cols_predicted": {
                "actual_INCORRECT_pred_INCORRECT": tp,
                "actual_INCORRECT_pred_CORRECT": fn,
                "actual_CORRECT_pred_INCORRECT": fp,
                "actual_CORRECT_pred_CORRECT": tn,
            },
            "TP_incorrect": tp,
            "FP_incorrect": fp,
            "TN_incorrect": tn,
            "FN_incorrect": fn,
            "num_binary_samples": cm["num_binary_samples"],
        },

        "error_detection_metrics": {
            "error_detection_probability_recall": error_detection_recall,
            "error_detection_precision": error_detection_precision,
            "false_alarm_probability_on_correct": error_detection_fpr,
            "binary_judgement_accuracy": error_detection_acc,
        },

        "correction_metrics": {
            "correction_probability_after_any_trigger": correction_success_after_trigger,
            "correction_probability_on_wrong_and_triggered": correction_success_on_wrong_triggered,
            "overall_correction_probability_on_old_wrong": overall_correction_rate_on_wrong,
        },

        "refinement_effect": {
            "improved_wrong_to_correct": improved_count,
            "harmed_correct_to_wrong": harmed_count,
            "unchanged_correct_to_correct": unchanged_correct_count,
            "unchanged_wrong_to_wrong": unchanged_wrong_count,
        }
    }
    return summary


def print_summary(summary: Dict[str, Any], output_csv_path: str, stats_json_path: str):
    print("\n================================")
    print(f"Completed! Results saved to {output_csv_path}")
    print(f"Stats saved to {stats_json_path}")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Refined samples: {summary['refined_samples']}")
    print(f"Old accuracy  : {summary['old_accuracy']:.4f} ({summary['old_correct_count']}/{summary['total_samples']})")
    print(f"Final accuracy: {summary['final_accuracy']:.4f} ({summary['final_correct_count']}/{summary['total_samples']})")
    vc = summary["verdict_counts"]
    print("\n[Critique Verdict Counts]")
    print(f"CORRECT  : {vc['CORRECT']}")
    print(f"INCORRECT: {vc['INCORRECT']}")
    
    print(f"UNSURE   : {vc['UNSURE']}")
    print(f"EMPTY    : {vc['EMPTY']}")

    cm = summary["confusion_matrix_binary"]
    print("\n[Confusion Matrix: Actual(old QA-EM) vs Predicted(critic verdict)]")
    print("                Pred=INCORRECT   Pred=CORRECT")
    print(f"Actual=INCORRECT      {cm['TP_incorrect']:6d}         {cm['FN_incorrect']:6d}")
    print(f"Actual=CORRECT        {cm['FP_incorrect']:6d}         {cm['TN_incorrect']:6d}")
    print(f"Binary samples used: {cm['num_binary_samples']}")

    edm = summary["error_detection_metrics"]
    print("\n[Error Detection Metrics]")
    print(f"P(pred=INCORRECT | actual wrong)  = {edm['error_detection_probability_recall']:.4f}")
    print(f"P(actual wrong | pred=INCORRECT)  = {edm['error_detection_precision']:.4f}")
    print(f"P(pred=INCORRECT | actual correct)= {edm['false_alarm_probability_on_correct']:.4f}")
    print(f"Binary judgement accuracy         = {edm['binary_judgement_accuracy']:.4f}")

    cm2 = summary["correction_metrics"]
    print("\n[Correction Metrics]")
    print(f"Correction probability after any trigger       = {cm2['correction_probability_after_any_trigger']:.4f}")
    print(f"Correction probability on wrong & triggered    = {cm2['correction_probability_on_wrong_and_triggered']:.4f}")
    print(f"Overall correction probability on old wrong    = {cm2['overall_correction_probability_on_old_wrong']:.4f}")

    eff = summary["refinement_effect"]
    print("\n[Refinement Effect]")
    print(f"Wrong -> Correct: {eff['improved_wrong_to_correct']}")
    print(f"Correct -> Wrong: {eff['harmed_correct_to_wrong']}")
    print(f"Correct -> Correct: {eff['unchanged_correct_to_correct']}")
    print(f"Wrong -> Wrong: {eff['unchanged_wrong_to_wrong']}")
    print("================================")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=-1, help="Only process first n rows; -1 means all")
    parser.add_argument("--input_csv_path", type=str, default=DEFAULT_INPUT_CSV_PATH)
    parser.add_argument("--output_csv_path", type=str, default=DEFAULT_OUTPUT_CSV_PATH)
    parser.add_argument("--stats_json_path", type=str, default=DEFAULT_STATS_JSON_PATH)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=args.local_files_only,
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=args.local_files_only,
    )
    model.eval()

    df = pd.read_csv(args.input_csv_path)
    if args.n is not None and args.n > 0:
        df = df.head(args.n).copy()

    required_cols = ["question", "trajectory", "extracted_answer", "golden_answer", critique_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {args.input_csv_path}. Got columns: {list(df.columns)}")

    rows = []

    for idx, row in df.iterrows():
        q = "" if pd.isna(row["question"]) else str(row["question"])
        old_trajectory = "" if pd.isna(row["trajectory"]) else str(row["trajectory"])
        old_answer = "" if pd.isna(row["extracted_answer"]) else str(row["extracted_answer"])
        gold_raw = row["golden_answer"]
        critique_text = "" if pd.isna(row[critique_col]) else str(row[critique_col])

        gold_list = parse_golden_answer_field(gold_raw)
        verdict = extract_verdict(critique_text)
        old_correct = is_answer_correct(old_answer, gold_list)
        
        print(f"\n=== Processing Row {idx + 1}/{len(df)} ===")
        print("Critique verdict:", verdict if verdict else "EMPTY")
        print("Old correct:", old_correct)

        try:
            if verdict == "INCORRECT":
                print("[Refine Triggered]")
                final_output, final_answer, refine_success, refine_status = refine_with_trajectory_and_critique(
                    q, old_trajectory, old_answer, critique_text, model, tokenizer, device
                )
                used_refine = 1 if refine_success else 0
            else:
                print("[Keep Original Trajectory]")
                final_output = old_trajectory
                final_answer = old_answer
                used_refine = 0
                refine_status = "not_triggered"

            final_correct = is_answer_correct(final_answer, gold_list)

            rows.append({
                "question": q,
                "trajectory": old_trajectory,
                "extracted_answer": old_answer,
                "golden_answer": gold_raw,
                "critique": critique_text,
                "critique_verdict": verdict,
                "old_correct": int(old_correct),
                "used_refine": used_refine,
                "refine_status": refine_status,
                "final_output": final_output,
                "final_extracted_answer": final_answer,
                "final_correct": int(final_correct),
                "improved_after_refine": int((not old_correct) and final_correct),
                "harmed_after_refine": int(old_correct and (not final_correct)),
            })

            print("Old answer  :", old_answer)
            print("Final answer:", final_answer)
            print("Final correct:", final_correct)
            print("Refine status:", refine_status)

        except Exception as e:
            err = f"ERROR: {repr(e)}"
            rows.append({
                "question": q,
                "trajectory": old_trajectory,
                "extracted_answer": old_answer,
                "golden_answer": gold_raw,
                "critique": critique_text,
                "critique_verdict": verdict,
                "old_correct": int(old_correct),
                "used_refine": -1,
                "refine_status": "exception",
                "final_output": err,
                "final_extracted_answer": err,
                "final_correct": 0,
                "improved_after_refine": 0,
                "harmed_after_refine": int(old_correct),
            })
            print(err)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output_csv_path, index=False, encoding="utf-8")

    summary = summarize_stats(out_df)
    with open(args.stats_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print_summary(summary, args.output_csv_path, args.stats_json_path)


if __name__ == "__main__":
    main()