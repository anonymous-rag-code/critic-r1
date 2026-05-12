import os
import re
import time
import argparse
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# -----------------------------
# Critic prompts (gold-aware teacher)
# -----------------------------
SYSTEM_PROMPT = (
    "You are a teacher critic for a retrieval-augmented QA system. "
    "You will audit a trajectory produced by another QA model.\n\n"

    "You are given the GOLD_ANSWER, which tells you whether the model's final answer is correct. "
    "Use GOLD_ANSWER ONLY to determine correctness and to understand the type of error. "
    "However, you must NOT reveal the GOLD_ANSWER directly in your critique.\n\n"

    "You are also given RELATED_CONTEXT. "
    "This is question-related context, and it may include the original question, supporting titles, candidate titles, "
    "and evidence snippets.\n"
    "Use RELATED_CONTEXT as auxiliary context for diagnosing missing retrieval coverage, off-target search, weak evidence usage, "
    "and for writing stronger repair guidance.\n"
    "Do NOT directly reveal or copy the gold answer from it.\n"
    "Do NOT rely on RELATED_CONTEXT alone unless it clearly supports your diagnosis.\n\n"

    "Your critique must help a downstream critic model that does NOT know the GOLD_ANSWER. "
    "Therefore:\n"
    "- Do NOT state the gold answer explicitly.\n"
    "- Do NOT say 'the correct answer is ...'.\n"
    "- Do NOT directly replace the answer with the gold answer.\n\n"

    "Instead, use GOLD_ANSWER internally to:\n"
    "- determine whether the final answer is correct\n"
    "- identify the error type\n"
    "- write targeted repair guidance\n\n"

    "Focus on the FINAL <answer>. Ignore minor reasoning flaws unless they affect the final answer.\n"
    "Use <information> as the main evidence context when explaining support gaps.\n"
    "Use RELATED_CONTEXT mainly to improve <reason>, <fix>, and <keywords>, especially when retrieval coverage is weak "
    "or the search misses relevant entities/titles.\n\n"

    "When writing <reason>:\n"
    "- describe the concrete error type\n"
    "- use short evidence-grounded phrases\n"
    "- avoid vague statements\n"
    "- if useful, mention missing relevant entity/title, weak retrieval coverage, or off-target search\n\n"

    "When writing <fix>:\n"
    "- give targeted search or verification guidance\n"
    "- do NOT reveal the gold answer\n"
    "- do NOT guess a replacement answer\n"
    "- if the trajectory missed important entities/titles from RELATED_CONTEXT, guide the model to search or verify them\n"
    "- prefer instructions like: try search..., verify..., ensure..., re-check...\n\n"

    "When writing <keywords>:\n"
    "- output retrieval-oriented keywords for BOTH CORRECT and INCORRECT cases\n"
    "- keywords must NOT be empty unless formatting recovery fails\n"
    "- for CORRECT cases, provide 1-3 concise verification keywords/entities/titles\n"
    "- for INCORRECT cases, provide 1-3 concise search/repair keywords/entities/titles\n"
    "- prefer concise search terms derived from RELATED_CONTEXT when useful\n"
    "- avoid generic keywords like 'more evidence' or 'check again'\n"
    "- output retrieval-oriented keywords only\n\n"

    "The output must contain ONLY the required tags."
)

CRITIC_INSTRUCTION = (
    "Input:\n"
    "- QUESTION\n"
    "- TRAJECTORY with tags: <think> <search> <information> <answer>\n"
    "- GOLD_ANSWER\n"
    "- RELATED_CONTEXT\n\n"

    "Task:\n"
    "1. Compare the final <answer> with GOLD_ANSWER.\n"
    "2. If they refer to the same answer, output CORRECT.\n"
    "3. If they refer to different answers, output INCORRECT.\n"
    "4. Write a concise critique explaining the error type and how to repair it.\n"
    "5. Use RELATED_CONTEXT only as relevant auxiliary context, especially to strengthen reason/fix/keywords when retrieval coverage is weak.\n\n"

    "Output (STRICT, exactly one line, in this exact order):\n"
    "<verdict>...</verdict><location>...</location><reason>...</reason><fix>...</fix><keywords>...</keywords>\n\n"

    "Allowed values:\n"
    "- verdict: CORRECT | INCORRECT | UNSURE\n"
    "- location: answer | information:DocK | search:stepK | think:stepK\n\n"

    "Decision rules:\n"
    "- CORRECT: final answer matches GOLD_ANSWER.\n"
    "- INCORRECT: final answer does not match GOLD_ANSWER.\n"
    "- UNSURE: use only when formatting recovery fails or the output cannot be interpreted.\n\n"

    "Constraints:\n"
    "- Output MUST start with <verdict>.\n"
    "- Output MUST contain exactly the five tags above.\n"
    "- No extra text.\n"
    "- No line breaks.\n\n"

    "Length limits:\n"
    "- <reason> <= 20 words\n"
    "- <fix> <= 15 words\n\n"

    "<reason> requirements:\n"
    "- short error phrases separated by commas\n"
    "- focus on error type\n"
    "- avoid vague phrases\n"
    "- if useful, mention missing relevant entity/title, weak retrieval coverage, or off-target search\n\n"

    "Common error types:\n"
    "- wrong entity\n"
    "- unsupported answer\n"
    "- wrong comparison\n"
    "- answer type mismatch\n"
    "- missing key evidence\n"
    "- contradiction with evidence\n"
    "- missing relevant title\n"
    "- weak retrieval coverage\n"
    "- off-target search\n\n"

    "<fix> requirements:\n"
    "- actionable repair instruction\n"
    "- do NOT reveal GOLD_ANSWER\n"
    "- do NOT directly replace the answer\n"
    "- prefer search/verification guidance\n"
    "- if relevant entities/titles from RELATED_CONTEXT were not used, guide the model to search or verify them\n\n"

    "<keywords> requirements:\n"
    "- MUST be present for both CORRECT and INCORRECT\n"
    "- should contain 1-3 short retrieval-oriented keywords/entities/titles\n"
    "- for CORRECT, use verification-oriented keywords\n"
    "- for INCORRECT, use repair-oriented keywords\n"
    "- do not leave it empty unless output parsing fails\n\n"

    "Examples of good <fix>:\n"
    "- search missing related title\n"
    "- search title entity and verify evidence\n"
    "- verify earlier vs later year\n"
    "- ensure answer is a city\n"
    "- re-check supporting document\n"
    "- verify spouse nationality\n\n"

    "Examples of bad <fix>:\n"
    "- the correct answer is ...\n"
    "- replace with ...\n"
    "- gold answer is ...\n"
)

# -----------------------------
# Regexes
# -----------------------------
TAG_LINE_RE = re.compile(
    r"^<verdict>.*?</verdict><location>.*?</location><reason>.*?</reason><fix>.*?</fix><keywords>.*?</keywords>\s*$",
    re.DOTALL,
)

CRITIQUE4_RE = re.compile(
    r"^<verdict>.*?</verdict><location>.*?</location><reason>.*?</reason><fix>.*?</fix>\s*$",
    re.DOTALL,
)

VERDICT_RE = re.compile(r"<verdict>\s*(CORRECT|INCORRECT|UNSURE)\s*</verdict>", re.DOTALL | re.IGNORECASE)
LOCATION_RE = re.compile(r"<location>\s*(.*?)\s*</location>", re.DOTALL | re.IGNORECASE)
REASON_RE = re.compile(r"<reason>\s*(.*?)\s*</reason>", re.DOTALL | re.IGNORECASE)
FIX_RE = re.compile(r"<fix>\s*(.*?)\s*</fix>", re.DOTALL | re.IGNORECASE)
KEYWORDS_RE = re.compile(r"<keywords>\s*(.*?)\s*</keywords>", re.DOTALL | re.IGNORECASE)

JUDGE_LINE_RE = re.compile(
    r"^<judge>\s*(CORRECT|INCORRECT)\s*</judge>\s*$",
    re.IGNORECASE,
)

# -----------------------------
# CSV utilities
# -----------------------------
def read_csv_auto(path: str):
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            print(f"[read_csv_auto] trying encoding={enc} for {path}")
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err

# -----------------------------
# Formatting helpers
# -----------------------------
def _clean_tag_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_tag(pattern: re.Pattern, text: str):
    m = pattern.search(text or "")
    if not m:
        return None
    return _clean_tag_text(m.group(1))


def is_valid_critic_line(line: str) -> bool:
    if line is None:
        return False
    s = _clean_tag_text(str(line))
    return TAG_LINE_RE.match(s) is not None


def is_valid_critic4_line(line: str) -> bool:
    if line is None:
        return False
    s = _clean_tag_text(str(line))
    return CRITIQUE4_RE.match(s) is not None


def normalize_critic_output(raw: str) -> str:
    raw = "" if raw is None else str(raw).strip()
    if not raw:
        return raw

    raw_one_line = _clean_tag_text(raw)
    if TAG_LINE_RE.match(raw_one_line):
        return raw_one_line

    verdict = _extract_tag(VERDICT_RE, raw)
    location = _extract_tag(LOCATION_RE, raw)
    reason = _extract_tag(REASON_RE, raw)
    fix = _extract_tag(FIX_RE, raw)
    keywords = _extract_tag(KEYWORDS_RE, raw)

    if all(x is not None for x in [verdict, location, reason, fix, keywords]):
        verdict = verdict.upper()
        return (
            f"<verdict>{verdict}</verdict>"
            f"<location>{location}</location>"
            f"<reason>{reason}</reason>"
            f"<fix>{fix}</fix>"
            f"<keywords>{keywords}</keywords>"
        )

    start = raw.find("<verdict>")
    end = raw.rfind("</keywords>")
    if start != -1 and end != -1 and end > start:
        block = raw[start:end + len("</keywords>")]
        block = _clean_tag_text(block)

        verdict = _extract_tag(VERDICT_RE, block)
        location = _extract_tag(LOCATION_RE, block)
        reason = _extract_tag(REASON_RE, block)
        fix = _extract_tag(FIX_RE, block)
        keywords = _extract_tag(KEYWORDS_RE, block)

        if all(x is not None for x in [verdict, location, reason, fix, keywords]):
            verdict = verdict.upper()
            return (
                f"<verdict>{verdict}</verdict>"
                f"<location>{location}</location>"
                f"<reason>{reason}</reason>"
                f"<fix>{fix}</fix>"
                f"<keywords>{keywords}</keywords>"
            )

    return raw_one_line


def parse_critic_fields(line: str) -> dict:
    s = normalize_critic_output(line)
    return {
        "verdict": _extract_tag(VERDICT_RE, s) or "",
        "location": _extract_tag(LOCATION_RE, s) or "",
        "reason": _extract_tag(REASON_RE, s) or "",
        "fix": _extract_tag(FIX_RE, s) or "",
        "keywords": _extract_tag(KEYWORDS_RE, s) or "",
    }


def make_parse_fail_critique() -> str:
    return (
        "<verdict>UNSURE</verdict>"
        "<location>answer</location>"
        "<reason>parse_fail</reason>"
        "<fix>re-check output format</fix>"
        "<keywords></keywords>"
    )

# -----------------------------
# LLM-as-Judge prompts
# -----------------------------
JUDGE_SYSTEM_PROMPT = (
    "You are an answer equivalence judge. "
    "Determine whether the predicted answer and the golden answer refer to the same correct answer. "
    "Be tolerant to aliases, abbreviations, demonyms, spelling variants, "
    "missing middle names, and geographic shorthand (e.g., UK vs England)."
)

JUDGE_INSTRUCTION = (
    "Input:\n"
    "- QUESTION\n"
    "- PREDICTED_ANSWER\n"
    "- GOLD_ANSWER\n\n"
    "Output (STRICT, one line only):\n"
    "<judge>CORRECT|INCORRECT</judge>\n\n"
    "Rules:\n"
    "- CORRECT if the answers refer to the same entity or meaning.\n"
    "- INCORRECT if they refer to different entities.\n"
    "- Output MUST start with <judge>.\n"
)

# -----------------------------
# Input helpers
# -----------------------------
def extract_question_text(text: str) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    marker = "Question:"
    idx = s.rfind(marker)
    if idx == -1:
        return s
    return s[idx + len(marker):].strip()


def build_critic_trajectory(model_output: str, extracted_answer: str) -> str:
    model_output = "" if model_output is None else str(model_output)
    extracted_answer = "" if extracted_answer is None else str(extracted_answer)

    traj = model_output.strip()

    if "<answer>" not in traj or "</answer>" not in traj:
        traj = (traj + "\n" if traj else "") + f"<answer>{extracted_answer}</answer>"

    if "<information>" not in traj or "</information>" not in traj:
        traj = (traj + "\n" if traj else "") + "<information></information>"

    if "<think>" not in traj:
        traj = "<think></think>\n" + traj
    if "<search>" not in traj:
        traj = "<search></search>\n" + traj

    return traj


def normalize_related_context(text: str) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def fallback_keywords(question: str, related_context: str) -> str:
    question = "" if question is None else str(question)
    related_context = "" if related_context is None else str(related_context)

    m = re.search(r"Supporting Titles:\s*(.+)", related_context, flags=re.IGNORECASE)
    if m:
        items = [x.strip() for x in m.group(1).split(",") if x.strip()]
        if items:
            return ", ".join(items[:3])

    m = re.search(r"Candidate Titles:\s*(.+)", related_context, flags=re.IGNORECASE)
    if m:
        items = [x.strip() for x in m.group(1).split(",") if x.strip()]
        if items:
            return ", ".join(items[:3])

    titles = re.findall(r'Title:\s*"?([^"\n\)]+)"?', related_context, flags=re.IGNORECASE)
    cleaned = []
    seen = set()
    for t in titles:
        t = t.strip()
        key = t.lower()
        if t and key not in seen:
            seen.add(key)
            cleaned.append(t)
    if cleaned:
        return ", ".join(cleaned[:3])

    words = re.findall(r"[A-Za-z][A-Za-z0-9\-']+", question)
    stop = {
        "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
        "is", "are", "was", "were", "do", "does", "did", "the", "a", "an",
        "of", "in", "on", "to", "for", "and", "or", "with", "by", "from",
        "same", "nationality"
    }
    kept = []
    seen = set()
    for w in words:
        wl = w.lower()
        if wl not in stop and wl not in seen:
            seen.add(wl)
            kept.append(w)
    if kept:
        return ", ".join(kept[:3])

    return "verify evidence"

# -----------------------------
# Local model wrapper
# -----------------------------
class LocalChatModel:
    def __init__(
        self,
        model_path: str,
        load_in_4bit: bool = True,
        max_input_length: int = 4096,
    ):
        self.model_path = model_path
        self.max_input_length = max_input_length

        print(f"[model] loading tokenizer from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
        }

        if load_in_4bit:
            print("[model] using 4-bit quantization")
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = quant_config
        else:
            print("[model] using fp16")
            model_kwargs["torch_dtype"] = torch.float16

        print(f"[model] loading model from {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs,
        )
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 160,
        temperature: float = 0.0,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        do_sample = temperature > 0

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.95

        outputs = self.model.generate(**gen_kwargs)
        gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
        out = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return out

# -----------------------------
# Retry helper
# -----------------------------
def _with_retries(fn, max_retries: int = 3, base_sleep: float = 1.0):
    last_err = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            print(f"[retry] attempt={attempt + 1} error={repr(e)}")
            time.sleep(base_sleep * (2 ** attempt))
    raise RuntimeError(f"Local generation failed after {max_retries} retries: {last_err}")

# -----------------------------
# Critic calls
# -----------------------------
def call_critic_reformat(model: LocalChatModel, raw_output: str,
                         max_retries: int = 2, base_sleep: float = 0.5) -> str:
    reformat_system = (
        "You are a formatter. Reformat the input into exactly one line with only these tags: "
        "<verdict>...</verdict><location>...</location><reason>...</reason><fix>...</fix><keywords>...</keywords>. "
        "Do not add or infer new content. Only preserve and normalize existing tagged content."
    )
    reformat_user = (
        "Reformat this output into exactly one valid one-line tag sequence. No extra text.\n\n"
        f"{raw_output}"
    )

    def _do():
        out = model.generate(
            system_prompt=reformat_system,
            user_prompt=reformat_user,
            max_new_tokens=128,
            temperature=0.0,
        )
        return normalize_critic_output(out)

    return _with_retries(_do, max_retries=max_retries, base_sleep=base_sleep)


def call_critic_once(
    model: LocalChatModel,
    question: str,
    trajectory: str,
    gold_answer: str,
    related_context: str,
    temperature: float = 0.3,
    max_retries: int = 3,
    base_sleep: float = 1.0,
) -> str:
    user_content = (
        f"{CRITIC_INSTRUCTION}\n"
        f"QUESTION:\n{question}\n\n"
        f"GOLD_ANSWER:\n{gold_answer}\n\n"
        f"RELATED_CONTEXT:\n{related_context}\n\n"
        f"TRAJECTORY:\n{trajectory}\n"
    )

    def _do():
        raw_out = model.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_content,
            max_new_tokens=160,
            temperature=temperature,
        )
        out = normalize_critic_output(raw_out)

        if is_valid_critic_line(out):
            return out

        repaired = call_critic_reformat(model, raw_out, max_retries=2, base_sleep=0.5)
        if is_valid_critic_line(repaired):
            return repaired

        return make_parse_fail_critique()

    return _with_retries(_do, max_retries=max_retries, base_sleep=base_sleep)


def vote_critic_outputs(outputs, question: str, related_context: str) -> str:
    parsed = []
    for out in outputs:
        norm = normalize_critic_output(out)
        fields = parse_critic_fields(norm)
        valid = is_valid_critic_line(norm)
        parsed.append({
            "raw": norm,
            "fields": fields,
            "valid": valid,
        })

    valid_items = [x for x in parsed if x["valid"]]
    if not valid_items:
        fail = make_parse_fail_critique()
        f = parse_critic_fields(fail)
        if not f["keywords"]:
            f["keywords"] = fallback_keywords(question, related_context)
        return (
            f"<verdict>{f['verdict']}</verdict>"
            f"<location>{f['location']}</location>"
            f"<reason>{f['reason']}</reason>"
            f"<fix>{f['fix']}</fix>"
            f"<keywords>{f['keywords']}</keywords>"
        )

    verdict_counts = {}
    for item in valid_items:
        v = item["fields"]["verdict"] or "UNSURE"
        verdict_counts[v] = verdict_counts.get(v, 0) + 1

    majority_verdict = sorted(
        verdict_counts.items(),
        key=lambda x: (-x[1], x[0] == "UNSURE")
    )[0][0]

    majority_items = [x for x in valid_items if x["fields"]["verdict"] == majority_verdict]

    def item_score(item):
        f = item["fields"]
        return (
            1 if f["keywords"].strip() else 0,
            1 if f["reason"].strip() else 0,
            1 if f["fix"].strip() else 0,
            1 if f["location"].strip() else 0,
        )

    best = sorted(majority_items, key=item_score, reverse=True)[0]
    f = best["fields"]

    if not f["keywords"].strip():
        f["keywords"] = fallback_keywords(question, related_context)

    return (
        f"<verdict>{f['verdict']}</verdict>"
        f"<location>{f['location']}</location>"
        f"<reason>{f['reason']}</reason>"
        f"<fix>{f['fix']}</fix>"
        f"<keywords>{f['keywords']}</keywords>"
    )


def call_critic(
    model: LocalChatModel,
    question: str,
    trajectory: str,
    gold_answer: str,
    related_context: str,
    num_votes: int = 3,
    temperature: float = 0.3,
) -> str:
    outputs = []
    for _ in range(num_votes):
        out = call_critic_once(
            model=model,
            question=question,
            trajectory=trajectory,
            gold_answer=gold_answer,
            related_context=related_context,
            temperature=temperature,
        )
        outputs.append(out)

    return vote_critic_outputs(outputs, question, related_context)

# -----------------------------
# Judge call
# -----------------------------
def call_judge_once(
    model: LocalChatModel,
    question: str,
    predicted_answer: str,
    golden_answer: str,
    temperature: float = 0.2,
    max_retries: int = 3,
    base_sleep: float = 1.0,
) -> str:
    predicted_answer = "" if predicted_answer is None else str(predicted_answer)
    golden_answer = "" if golden_answer is None else str(golden_answer)

    user_content = (
        f"{JUDGE_INSTRUCTION}\n"
        f"QUESTION:\n{question}\n\n"
        f"PREDICTED_ANSWER:\n{predicted_answer}\n\n"
        f"GOLD_ANSWER:\n{golden_answer}\n"
    )

    def _do():
        out = model.generate(
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_prompt=user_content,
            max_new_tokens=32,
            temperature=temperature,
        ).strip()

        if not out.startswith("<judge>"):
            idx = out.find("<judge>")
            if idx != -1:
                out = out[idx:].strip()

        return out.splitlines()[0].strip()

    return _with_retries(_do, max_retries=max_retries, base_sleep=base_sleep)


def call_judge(
    model: LocalChatModel,
    question: str,
    predicted_answer: str,
    golden_answer: str,
    num_votes: int = 3,
    temperature: float = 0.2,
) -> str:
    votes = []
    for _ in range(num_votes):
        line = call_judge_once(
            model=model,
            question=question,
            predicted_answer=predicted_answer,
            golden_answer=golden_answer,
            temperature=temperature,
        )
        parsed = parse_judge(line)
        votes.append(parsed if parsed is not None else "PARSE_FAIL")

    counts = {}
    for v in votes:
        counts[v] = counts.get(v, 0) + 1

    best = sorted(counts.items(), key=lambda x: -x[1])[0][0]
    if best in {"CORRECT", "INCORRECT"}:
        return f"<judge>{best}</judge>"
    return "<judge>INCORRECT</judge>"


def parse_judge(line: str):
    if line is None:
        return None
    s = str(line).strip()
    m = JUDGE_LINE_RE.match(s)
    if not m:
        return None
    return m.group(1).upper()

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--sleep", type=float, default=0.0, help="sleep seconds between rows")
    ap.add_argument("--max_rows", type=int, default=-1)
    ap.add_argument("--resume", action="store_true", help="skip already processed rows in out_csv if exists")

    ap.add_argument("--model_path", default="/data/Qwen2.5-14B-Instruct")
    ap.add_argument("--load_in_4bit", action="store_true", help="use 4bit quantization")
    ap.add_argument("--max_input_length", type=int, default=4096)

    ap.add_argument("--critic_num_votes", type=int, default=3)
    ap.add_argument("--judge_num_votes", type=int, default=3)
    ap.add_argument("--critic_vote_temp", type=float, default=0.3)
    ap.add_argument("--judge_vote_temp", type=float, default=0.2)

    args = ap.parse_args()

    model = LocalChatModel(
        model_path=args.model_path,
        load_in_4bit=args.load_in_4bit,
        max_input_length=args.max_input_length,
    )

    df = read_csv_auto(args.in_csv)

    required_cols = ["question", "context", "model_output", "extracted_answer", "golden_answer"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}. Found: {list(df.columns)}")

    if args.max_rows > 0:
        df = df.head(args.max_rows).copy()
    else:
        df = df.copy()

    df["parsed_question"] = df["question"].apply(extract_question_text)
    df["related_context"] = df["context"].apply(normalize_related_context)

    if args.resume and os.path.exists(args.out_csv):
        out_df = read_csv_auto(args.out_csv)
        if len(out_df) != len(df):
            print(f"[resume] warning: existing out_csv length={len(out_df)} != input length={len(df)}")
        if "critique" not in out_df.columns:
            out_df["critique"] = ""
        if "keywords" not in out_df.columns:
            out_df["keywords"] = ""
        if "llm_judge" not in out_df.columns:
            out_df["llm_judge"] = ""
    else:
        out_df = pd.DataFrame(index=df.index)
        out_df["question"] = df["parsed_question"]
        out_df["trajectory"] = df["model_output"]
        out_df["extracted_answer"] = df["extracted_answer"]
        out_df["golden_answer"] = df["golden_answer"]
        out_df["critique"] = ""
        out_df["keywords"] = ""
        out_df["llm_judge"] = ""

    total = len(df)
    out_df["question"] = df["parsed_question"]

    for i in range(total):
        print(f"[row {i}] start")

        already_done = (
            i in out_df.index
            and is_valid_critic4_line(str(out_df.at[i, "critique"]))
            and str(out_df.at[i, "llm_judge"]) in {"CORRECT", "INCORRECT", "PARSE_FAIL"}
        )
        if args.resume and already_done:
            continue

        q = df.at[i, "parsed_question"]
        raw_trajectory = df.at[i, "model_output"]
        extracted = df.at[i, "extracted_answer"]
        gold = df.at[i, "golden_answer"]
        related_context = df.at[i, "related_context"]

        critic_trajectory = build_critic_trajectory(raw_trajectory, extracted)

        raw_critique = call_critic(
            model=model,
            question=q,
            trajectory=critic_trajectory,
            gold_answer=gold,
            related_context=related_context,
            num_votes=args.critic_num_votes,
            temperature=args.critic_vote_temp,
        )
        fields = parse_critic_fields(raw_critique)

        critique = (
            f"<verdict>{fields['verdict']}</verdict>"
            f"<location>{fields['location']}</location>"
            f"<reason>{fields['reason']}</reason>"
            f"<fix>{fields['fix']}</fix>"
        )

        keywords = fields["keywords"]
        if not str(keywords).strip():
            keywords = fallback_keywords(q, related_context)

        judge_line = call_judge(
            model=model,
            question=q,
            predicted_answer=extracted,
            golden_answer=gold,
            num_votes=args.judge_num_votes,
            temperature=args.judge_vote_temp,
        )
        judge_parsed = parse_judge(judge_line)
        llm_judge = judge_parsed if judge_parsed is not None else "PARSE_FAIL"

        out_df.at[i, "question"] = q
        out_df.at[i, "trajectory"] = raw_trajectory
        out_df.at[i, "extracted_answer"] = extracted
        out_df.at[i, "golden_answer"] = gold
        out_df.at[i, "critique"] = critique
        out_df.at[i, "keywords"] = keywords
        out_df.at[i, "llm_judge"] = llm_judge

        if (i + 1) % 20 == 0:
            out_df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
            print(f"[autosave] saved {i + 1}/{total} to {args.out_csv}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    out_df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"Done. Wrote {args.out_csv}")


if __name__ == "__main__":
    main()