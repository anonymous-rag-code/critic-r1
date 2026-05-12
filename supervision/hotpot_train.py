
# This script generates model trajectories on the HotpotQA training split
# for downstream critic-supervision annotation.

import json
import re
import os
import requests
import pandas as pd
import transformers
import torch
from typing import List, Tuple, Any, Set
import argparse

# -----------------------------
# Config
# -----------------------------
num_questions = 5000

hotpot_json_path = "hotpot_train_v1.1.json"

model_id = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# for Qwen2.5 series models
curr_eos = [151645, 151643]
curr_search_template = "\n\n{output_text}<information>{search_results}</information>\n\n"

# search service
retriever_url = "http://127.0.0.1:8000/retrieve"
topk = 5
request_timeout_sec = 120

# guard
max_search_calls = 1
max_new_tokens = 1024

# generation params
do_sample = True
temperature = 0.7
top_p = 0.9
repetition_penalty = 1.1


# -----------------------------
# Robust helpers
# -----------------------------
def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x).strip()


def build_hotpot_context(question: str, supporting_facts: Any, context: Any) -> str:
    """
    Teacher-only context for HotpotQA.
    Not shown to the model during inference.

    Use:
      - question
      - supporting titles
      - gold supporting sentences
      - candidate titles
    """
    context_parts = [
        "[HotpotQA Structured Context]",
        f"Question: {safe_str(question)}",
    ]

    # supporting_facts format:
    # [
    #   ["Scott Derrickson", 0],
    #   ["Ed Wood", 0]
    # ]
    supporting_titles = []
    evidence_lines = []

    # context format:
    # [
    #   ["Title1", ["sent1", "sent2"]],
    #   ["Title2", ["sent1", "sent2"]]
    # ]
    title_to_sentences = {}
    candidate_titles = []

    if isinstance(context, list):
        for item in context:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                title = safe_str(item[0])
                sentences = item[1]
                if title:
                    candidate_titles.append(title)
                    if isinstance(sentences, list):
                        title_to_sentences[title] = [safe_str(s) for s in sentences]
                    else:
                        title_to_sentences[title] = []

    if isinstance(supporting_facts, list):
        for item in supporting_facts:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            title = safe_str(item[0])
            sent_id = item[1]

            if title and title not in supporting_titles:
                supporting_titles.append(title)

            if title in title_to_sentences and isinstance(sent_id, int):
                sents = title_to_sentences[title]
                if 0 <= sent_id < len(sents):
                    sent = safe_str(sents[sent_id])
                    if sent:
                        evidence_lines.append(f"Doc (Title: {title}): {sent}")

    if supporting_titles:
        context_parts.append("Supporting Titles: " + ", ".join(supporting_titles))

    if candidate_titles:
        uniq_titles = []
        seen = set()
        for t in candidate_titles:
            if t not in seen:
                seen.add(t)
                uniq_titles.append(t)
        context_parts.append("Candidate Titles: " + ", ".join(uniq_titles[:10]))

    if evidence_lines:
        context_parts.append("Evidence:\n" + "\n".join(evidence_lines))

    return "\n".join(context_parts)


# -----------------------------
# Data loading: HotpotQA json
# -----------------------------
def load_questions_from_hotpot_json(
    json_path: str,
    n: int
) -> Tuple[List[str], List[str], List[str]]:
    """
    Returns:
      questions: List[str]
      golden_answers: List[str]
      contexts: List[str]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("HotpotQA json must be a list of samples.")

    questions: List[str] = []
    golden_answers: List[str] = []
    contexts: List[str] = []

    for i, obj in enumerate(data):
        if len(questions) >= n:
            break

        q = safe_str(obj.get("question", ""))
        ans = safe_str(obj.get("answer", ""))
        supporting_facts = obj.get("supporting_facts", [])
        ctx = obj.get("context", [])

        if not q:
            continue
        if not ans:
            continue

        questions.append(q)
        golden_answers.append(ans)
        contexts.append(build_hotpot_context(q, supporting_facts, ctx))

    return questions, golden_answers, contexts


# -----------------------------
# Resume / incremental save
# -----------------------------
def make_resume_key(question: str, golden_answer: str) -> str:
    return f"{question}\t{golden_answer}"


def get_completed_keys(csv_path: str) -> Set[str]:
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return set()

    try:
        old_df = pd.read_csv(csv_path, keep_default_na=False)
        if "question" not in old_df.columns or "golden_answer" not in old_df.columns:
            return set()
        return set(
            make_resume_key(q, g)
            for q, g in zip(old_df["question"].astype(str), old_df["golden_answer"].astype(str))
        )
    except Exception as e:
        print(f"[WARN] Failed to read existing output CSV for resume: {repr(e)}")
        return set()


def append_result_row(
    csv_path: str,
    question: str,
    context: str,
    model_output: str,
    extracted_answer: str,
    golden_answer: str,
) -> None:
    row_df = pd.DataFrame([{
        "question": question,
        "context": context,
        "model_output": model_output,
        "extracted_answer": extracted_answer,
        "golden_answer": golden_answer,
    }])

    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    row_df.to_csv(
        csv_path,
        mode="a",
        header=not file_exists,
        index=False,
        encoding="utf-8"
    )


# -----------------------------
# Utilities
# -----------------------------
def extract_final_answer(text: str) -> str:
    pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1].strip()
    return ""


def get_query(text: str):
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1].strip()
    return None


class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        self.target_ids = [tokenizer.encode(s, add_special_tokens=False) for s in target_sequences]
        self.target_lengths = [len(t) for t in self.target_ids]

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] < min(self.target_lengths):
            return False

        for tid, tlen in zip(self.target_ids, self.target_lengths):
            target = torch.as_tensor(tid, device=input_ids.device)
            if torch.equal(input_ids[0, -tlen:], target):
                return True
        return False


def search(query: str) -> str:
    payload = {
        "queries": [query],
        "topk": topk,
        "return_scores": True
    }
    r = requests.post(retriever_url, json=payload, timeout=request_timeout_sec)
    r.raise_for_status()
    results = r.json()["result"]

    def _passages2string(retrieval_result):
        format_reference = ""
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["document"]["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference

    return _passages2string(results[0])


# -----------------------------
# Main inference logic
# -----------------------------
def answer_single_question(question: str, model, tokenizer, device):
    question = safe_str(question)
    if question and question[-1] != "?":
        question += "?"

    prompt = (
        "Answer the given question. "
        "You must conduct reasoning inside <think> and </think> first every time you get new information. "
        "If you lack knowledge, you may call a search engine by writing <search> query </search>. "
        "The retrieved results will be returned between <information> and </information>. "
        "You may call the search engine at most once. "
        "After receiving retrieved information, provide the final answer inside <answer> and </answer>. "
        "If no search is needed, directly provide the answer inside <answer> and </answer>, "
        "without detailed illustrations. For example, <answer> Beijing </answer>. "
        f"Question: {question}\n"
    )

    tokenizer_has_template = bool(getattr(tokenizer, "chat_template", None))
    if tokenizer_has_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False
        )

    target_sequences = [
        "</search>", " </search>", "</search>\n", " </search>\n",
        "</search>\n\n", " </search>\n\n"
    ]
    stopping_criteria = transformers.StoppingCriteriaList(
        [StopOnSequence(target_sequences, tokenizer)]
    )

    print("\n\n################# [Single-Search Reasoning] ##################\n\n")
    print(prompt)

    search_calls = 0
    full_output = ""

    while True:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        attention_mask = torch.ones_like(input_ids)

        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )

        generated_tokens = outputs[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Case 1: model directly ends
        if outputs[0][-1].item() in curr_eos:
            print(output_text)
            full_output += output_text
            break

        # Case 2: model issues a search
        tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))

        if tmp_query and search_calls < max_search_calls:
            try:
                search_results = search(tmp_query)
            except Exception as e:
                search_results = f"[SEARCH_ERROR] {repr(e)}"

            search_text = curr_search_template.format(
                output_text=output_text,
                search_results=search_results
            )

            prompt += search_text
            full_output += search_text
            search_calls += 1

            print(search_text)

            prompt += (
                "Now provide the final answer based on the retrieved information above. "
                "Do not search again. "
                "Return only the final answer inside <answer> and </answer>.\n"
            )

            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            attention_mask = torch.ones_like(input_ids)

            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=tokenizer.eos_token_id
            )

            generated_tokens = outputs[0][input_ids.shape[1]:]
            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            print(output_text)
            full_output += output_text
            break

        # Case 3: no valid search query extracted, or search limit reached
        print(output_text)
        full_output += output_text
        break

    final_answer = extract_final_answer(full_output)
    return full_output, final_answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", type=str, default="default")
    args = parser.parse_args()
    
    output_csv_path = f"hotpotqa_trainset_inference_results_{args.suffix}.csv"

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    questions, golden_answers, contexts = load_questions_from_hotpot_json(
        hotpot_json_path, num_questions
    )

    completed_keys = get_completed_keys(output_csv_path)
    print(f"[Resume] Found {len(completed_keys)} completed rows in existing output.")

    processed_now = 0
    skipped_existing = 0

    for i, q in enumerate(questions):
        print(f"\n=== Processing Question {i+1}/{len(questions)} ===")

        key = make_resume_key(q, golden_answers[i])
        if key in completed_keys:
            skipped_existing += 1
            print("[Skip] Already completed in existing CSV.")
            continue

        try:
            full_output, final_answer = answer_single_question(q, model, tokenizer, device)
            append_result_row(
                csv_path=output_csv_path,
                question=q,
                context=contexts[i],
                model_output=full_output,
                extracted_answer=final_answer,
                golden_answer=golden_answers[i],
            )
            completed_keys.add(key)
            processed_now += 1

        except Exception as e:
            err = f"ERROR: {repr(e)}"
            append_result_row(
                csv_path=output_csv_path,
                question=q,
                context=contexts[i],
                model_output=err,
                extracted_answer="",
                golden_answer=golden_answers[i],
            )
            completed_keys.add(key)
            processed_now += 1
            print(f"[Error] {err}")

    print("\nCompleted!")
    print(f"Saved to: {output_csv_path}")
    print(f"Newly processed this run: {processed_now}")
    print(f"Skipped from existing output: {skipped_existing}")


if __name__ == "__main__":
    main()