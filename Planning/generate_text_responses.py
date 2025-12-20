# you_think.py
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch


# -----------------------------
# Config
# -----------------------------
@dataclass
class YouThinkConfig:
    # outputs
    n_responses: int = 8                         # N: how many plans to return
    max_actions_per_response: int = 6            # max length of each action list
    min_actions_per_response: int = 2            # encourage atomic multi-step plans
    max_action_chars: int = 80                   # per action sentence max chars
    max_total_prompt_chars: int = 12000          # avoid overly long prompts

    # decoding
    max_new_tokens: int = 512
    temperature: float = 0.9
    top_p: float = 0.92
    top_k: int = 50
    repetition_penalty: float = 1.05
    do_sample: bool = True
    num_beams: int = 1

    # diversity / robustness
    single_call: bool = True                     # try one generation that returns N lines
    max_attempts: int = 6                        # if parsing fails, retry up to this many times
    seed: Optional[int] = None

    # few-shot (optional): point to your HumanML3D texts folder
    # e.g. "T2M-GPT/dataset/HumanML3D/texts"
    humanml3d_texts_dir: Optional[str] = None
    few_shot_k: int = 6

    # verb restriction (optional): provide a whitelist of base verbs
    # if None, we use a reasonable default shortlist
    verb_whitelist: Optional[List[str]] = None

    # formatting preferences
    force_humanml3d_style: bool = True           # convert "walk forward" -> "a person walks forward"
    subject: str = "a person"                    # HumanML3D-like subject
    forbid_camera_words: bool = True             # avoid "camera, shot, scene cut ..."
    keep_single_actor: bool = True               # keep "one person" motion

    # how you want the raw Think output to look like
    # "paren_lines": each candidate in one line: (walk forward, bend down, pick up something)
    output_mode: str = "paren_lines"             # or "json"

    # stop tokens (optional)
    stop_strings: Tuple[str, ...] = ("###",)


# -----------------------------
# Few-shot loader (HumanML3D texts)
# -----------------------------
def _load_humanml3d_examples(texts_dir: str, k: int, rng: random.Random) -> List[str]:
    """
    Loads up to k short caption lines from HumanML3D/texts.
    HumanML3D format: each file may contain multiple lines; we sample lines.
    """
    p = Path(texts_dir)
    if not p.exists():
        return []
    files = list(p.glob("*.txt"))
    if not files:
        return []

    rng.shuffle(files)
    examples: List[str] = []
    for fp in files:
        try:
            lines = [ln.strip() for ln in fp.read_text(encoding="utf-8", errors="ignore").splitlines()]
        except Exception:
            continue
        # filter
        lines = [ln for ln in lines if ln and len(ln) <= 140]
        if not lines:
            continue
        rng.shuffle(lines)
        for ln in lines[:3]:
            # normalize
            ln = re.sub(r"\s+", " ", ln).strip()
            # keep it “action-text-ish”
            examples.append(ln)
            if len(examples) >= k:
                return examples
    return examples[:k]


# -----------------------------
# Prompt builder
# -----------------------------
_DEFAULT_VERBS = [
    "walk", "run", "jog", "stand", "sit", "kneel", "crouch", "bend", "turn", "rotate",
    "step", "jump", "hop", "reach", "pick", "lift", "lower", "throw", "catch", "wave",
    "clap", "point", "look", "nod", "shake", "kick", "push", "pull", "spin", "dance",
    "stretch", "lean", "crawl"
]

_CAMERA_WORDS = re.compile(r"\b(camera|shot|zoom|pan|cut|cinematic|frame|lens)\b", re.IGNORECASE)


def _build_system_instruction(cfg: YouThinkConfig) -> str:
    verbs = cfg.verb_whitelist or _DEFAULT_VERBS
    verb_str = ", ".join(verbs[:60])  # avoid huge list

    # We emulate the paper's "extract action instructions; each result on separate line; no reasoning"
    # and add "HumanML3D-friendly" constraints.
    system = f"""
You are a motion-planning assistant for Text-to-Motion (T2M) models.
Convert an arbitrary user request (may describe a scene/event) into multiple plausible human action plans.

Hard rules:
- Output ONLY action instructions. NO explanations, NO reasoning, NO extra words.
- Each candidate plan describes ONE PERSON (single actor), physically plausible, temporally ordered.
- Prefer short atomic actions that T2M datasets understand well (HumanML3D-style).
- Avoid camera/film terms, avoid meta words (prompt, model, etc.).
- Avoid multi-person interactions unless user explicitly requests; otherwise keep one person.

Verb/style constraints:
- Use common motion verbs (examples: {verb_str}).
- Keep each action step concise (about 3–12 words).
- Include directions when helpful (forward/backward/left/right), and simple body cues (bend down, turn around).

Output format:
"""
    if cfg.output_mode == "json":
        system += """
Return STRICT JSON:
{"plans":[{"actions":["<action sentence 1>","<action sentence 2>", ...]}, ...]}
No trailing commas. No markdown fences.
"""
    else:
        system += """
Return N candidate plans, ONE plan per line, EXACTLY in this format:
(action 1, action 2, action 3)
Where each "action i" is a short action phrase (not a full paragraph).
No numbering. No bullets. No extra text outside parentheses.
"""
    return system.strip()


def _build_few_shot_block(cfg: YouThinkConfig, rng: random.Random) -> str:
    # If user gives HumanML3D texts dir, we sample a few action-text-like examples to steer style.
    # If not, we provide a small built-in set.
    examples: List[str] = []
    if cfg.humanml3d_texts_dir:
        examples = _load_humanml3d_examples(cfg.humanml3d_texts_dir, cfg.few_shot_k, rng)

    if not examples:
        examples = [
            "a person walks forward then bends down to pick up something",
            "a person turns left and takes a few steps forward",
            "a person jogs forward and stops, then waves with one hand",
            "a person stands still, looks around, then sits down",
            "a person walks forward, turns around, and walks back",
            "a person crouches down, reaches forward, then stands up"
        ]

    # Turn them into "Scene Text -> action plans" few-shot
    # (we intentionally craft scene texts that imply those actions).
    pairs = [
        ("Someone notices an object on the floor in front of them.",
         "(walk forward, bend down, pick up something)"),
        ("A person tries to get someone’s attention from a distance.",
         "(walk forward, wave one hand, stop)"),
        ("A person feels tired and decides to rest.",
         "(walk forward, slow down, sit down)"),
        ("A person looks for the right direction and changes course.",
         "(stand still, look around, turn left, walk forward)")
    ]

    # If we have real HumanML3D lines, we use them as "good targets" by rewriting into paren format
    # to bias language. We keep this simple and robust.
    extra = []
    for ln in examples[: min(4, len(examples))]:
        # naive split into atomic pieces
        parts = re.split(r"\bthen\b|,|and then|and\b", ln, flags=re.IGNORECASE)
        parts = [re.sub(r"^\s*a person\s+", "", p.strip(), flags=re.IGNORECASE) for p in parts]
        parts = [p for p in parts if p]
        if len(parts) >= 2:
            extra.append(f"({', '.join(parts[:4])})")

    fewshot = "Few-shot examples:\n"
    for s, a in pairs:
        fewshot += f"Scene Text: {s}\nAction Plans:\n{a}\n\n"
    if extra:
        fewshot += "More good-format action plans:\n" + "\n".join(extra) + "\n"
    return fewshot.strip()


def _build_user_prompt(user_request: str, cfg: YouThinkConfig) -> str:
    # Quantity / diversity hints
    req = f"""
Scene Text:
{user_request.strip()}

Requirements:
- Produce exactly {cfg.n_responses} different plausible plans.
- Each plan should have {cfg.min_actions_per_response}–{cfg.max_actions_per_response} actions.
- Prefer actions that would appear in HumanML3D-like captions.
"""
    if cfg.forbid_camera_words:
        req += "- Do NOT use camera/shot/zoom/cinematic words.\n"
    if cfg.keep_single_actor:
        req += "- Keep a single actor (one person).\n"
    return req.strip()


# -----------------------------
# Generation wrappers
# -----------------------------
def _apply_chat_template_if_possible(tokenizer, messages: List[Dict[str, str]]) -> str:
    """
    Works for many HF chat models. Falls back to a plain concatenation.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    # fallback: simple
    out = ""
    for m in messages:
        out += f"{m['role'].upper()}:\n{m['content'].strip()}\n\n"
    out += "ASSISTANT:\n"
    return out


@torch.inference_mode()
def _generate_text_hf(
    model,
    tokenizer,
    prompt: str,
    cfg: YouThinkConfig,
    device: Optional[Union[str, torch.device]] = None,
    generator: Optional[torch.Generator] = None,
    **gen_kwargs: Any,
) -> str:
    if device is None:
        device = getattr(model, "device", "cuda" if torch.cuda.is_available() else "cpu")

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # default generation args
    kwargs = dict(
        max_new_tokens=cfg.max_new_tokens,
        do_sample=cfg.do_sample,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        repetition_penalty=cfg.repetition_penalty,
        num_beams=cfg.num_beams,
        pad_token_id=getattr(tokenizer, "pad_token_id", None),
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
    )
    kwargs.update(gen_kwargs)

    out_ids = model.generate(**inputs, **kwargs)
    # take the first sequence
    text = tokenizer.decode(out_ids[0], skip_special_tokens=True)

    # remove prompt echo if present
    if text.startswith(prompt):
        text = text[len(prompt):]
    return text.strip()


# -----------------------------
# Parsing + postprocess
# -----------------------------
_PAREN_LINE = re.compile(r"^\s*\((.*?)\)\s*$")

def _extract_paren_lines(text: str) -> List[List[str]]:
    """
    Parse lines like:
    (walk forward, bend down, pick up something)
    """
    plans: List[List[str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # stop early if model starts a new section
        if line.startswith("###"):
            break
        m = _PAREN_LINE.match(line)
        if not m:
            continue
        inner = m.group(1).strip()
        if not inner:
            continue
        parts = [p.strip() for p in inner.split(",")]
        parts = [re.sub(r"\s+", " ", p) for p in parts if p]
        if parts:
            plans.append(parts)
    return plans


def _safe_json_extract(text: str) -> Optional[dict]:
    """
    Try to extract the first JSON object in the text.
    """
    # heuristic: find first '{' and last '}'
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    chunk = text[start:end + 1]
    try:
        return json.loads(chunk)
    except Exception:
        return None


def _conjugate_3rd_person(base_verb: str) -> str:
    """
    Very small heuristic conjugator: walk->walks, carry->carries, go->goes, etc.
    """
    v = base_verb.lower()
    irregular = {"be": "is", "have": "has", "do": "does", "go": "goes"}
    if v in irregular:
        return irregular[v]
    if v.endswith(("s", "x", "z", "ch", "sh")):
        return v + "es"
    if v.endswith("y") and len(v) >= 2 and v[-2] not in "aeiou":
        return v[:-1] + "ies"
    return v + "s"


def _phrase_to_humanml_sentence(phrase: str, cfg: YouThinkConfig) -> str:
    """
    Convert "walk forward" -> "a person walks forward" (HumanML3D-ish).
    """
    phrase = phrase.strip()
    phrase = re.sub(r"\s+", " ", phrase)
    if not phrase:
        return phrase

    if re.match(r"^(a person|the person|someone)\b", phrase, flags=re.IGNORECASE):
        return phrase

    # verb is first token; conjugate it
    toks = phrase.split(" ")
    verb = re.sub(r"[^\w-]", "", toks[0].lower())
    rest = " ".join(toks[1:]).strip()

    if verb.endswith("s") or verb in {"is", "has", "does", "goes"}:
        sent = f"{cfg.subject} {verb}"
    else:
        sent = f"{cfg.subject} {_conjugate_3rd_person(verb)}"

    if rest:
        sent += f" {rest}"
    return sent.strip()


def _sanitize_action_sentence(s: str, cfg: YouThinkConfig) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)

    # remove forbidden camera words
    if cfg.forbid_camera_words:
        s = _CAMERA_WORDS.sub("", s)
        s = re.sub(r"\s+", " ", s).strip()

    # trim length
    if len(s) > cfg.max_action_chars:
        # try trimming after a comma/then
        for sep in [",", " then ", " and "]:
            idx = s.lower().find(sep)
            if 0 < idx < cfg.max_action_chars:
                s = s[:idx].strip()
                break
        s = s[: cfg.max_action_chars].strip()

    # remove trailing punctuation
    s = s.rstrip(" .;")
    return s


def _postprocess_plans(plans: List[List[str]], cfg: YouThinkConfig) -> List[List[str]]:
    out: List[List[str]] = []
    seen = set()

    for plan in plans:
        # enforce length
        plan = plan[: cfg.max_actions_per_response]
        if len(plan) < cfg.min_actions_per_response:
            continue

        steps: List[str] = []
        for ph in plan:
            if cfg.force_humanml3d_style:
                sent = _phrase_to_humanml_sentence(ph, cfg)
            else:
                sent = ph
            sent = _sanitize_action_sentence(sent, cfg)
            if not sent:
                continue
            steps.append(sent)

        steps = steps[: cfg.max_actions_per_response]
        if len(steps) < cfg.min_actions_per_response:
            continue

        key = tuple(steps)
        if key in seen:
            continue
        seen.add(key)
        out.append(list(steps))

    return out


# -----------------------------
# Core API
# -----------------------------
def you_think_generate(
    llm_model,
    tokenizer,
    user_request: str,
    *,
    config: Optional[YouThinkConfig] = None,
    device: Optional[Union[str, torch.device]] = None,
    **gen_kwargs: Any,
) -> List[List[str]]:
    """
    Core function:
    Input: HF-style llm_model + tokenizer + user_request (+ config/kwargs)
    Output: N action plans, each is a list[str] of HumanML3D-friendly action sentences.
    """
    cfg = config or YouThinkConfig()
    rng = random.Random(cfg.seed if cfg.seed is not None else random.randint(1, 10**9))

    system = _build_system_instruction(cfg)
    fewshot = _build_few_shot_block(cfg, rng)
    user = _build_user_prompt(user_request, cfg)

    # keep prompt under budget
    prompt_content = system + "\n\n" + fewshot + "\n\n" + user
    if len(prompt_content) > cfg.max_total_prompt_chars:
        prompt_content = prompt_content[-cfg.max_total_prompt_chars:]

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": fewshot + "\n\n" + user},
    ]
    prompt = _apply_chat_template_if_possible(tokenizer, messages)

    plans_raw: List[List[str]] = []

    # generation strategy
    def one_attempt(seed_int: int) -> str:
        gen = torch.Generator(device=str(device) if device is not None else "cpu")
        gen.manual_seed(seed_int)
        return _generate_text_hf(
            llm_model, tokenizer, prompt, cfg, device=device, generator=gen, **gen_kwargs
        )

    if cfg.single_call:
        # Ask for N plans in one completion
        for attempt in range(cfg.max_attempts):
            txt = one_attempt(rng.randint(1, 10**9))

            if cfg.output_mode == "json":
                obj = _safe_json_extract(txt)
                if isinstance(obj, dict) and "plans" in obj:
                    for p in obj["plans"]:
                        if isinstance(p, dict) and isinstance(p.get("actions"), list):
                            actions = [str(x).strip() for x in p["actions"] if str(x).strip()]
                            plans_raw.append(actions)
            else:
                plans_raw.extend(_extract_paren_lines(txt))

            if len(plans_raw) >= cfg.n_responses:
                break
    else:
        # Sample one plan per completion until we get N
        attempts = 0
        while len(plans_raw) < cfg.n_responses and attempts < cfg.max_attempts * cfg.n_responses:
            attempts += 1
            txt = one_attempt(rng.randint(1, 10**9))
            if cfg.output_mode == "json":
                obj = _safe_json_extract(txt)
                if isinstance(obj, dict) and "plans" in obj:
                    for p in obj["plans"]:
                        if isinstance(p, dict) and isinstance(p.get("actions"), list):
                            actions = [str(x).strip() for x in p["actions"] if str(x).strip()]
                            plans_raw.append(actions)
                else:
                    # fallback parse
                    plans_raw.extend(_extract_paren_lines(txt))
            else:
                parsed = _extract_paren_lines(txt)
                if parsed:
                    plans_raw.extend(parsed)

    # postprocess & enforce N
    plans = _postprocess_plans(plans_raw, cfg)

    # If not enough, pad by re-sampling or duplicating best-effort
    if len(plans) < cfg.n_responses:
        # try a few extra quick attempts
        extra_try = 0
        while len(plans) < cfg.n_responses and extra_try < cfg.max_attempts:
            extra_try += 1
            txt = one_attempt(rng.randint(1, 10**9))
            plans2 = _postprocess_plans(_extract_paren_lines(txt), cfg)
            for p in plans2:
                if len(plans) >= cfg.n_responses:
                    break
                if tuple(p) not in {tuple(x) for x in plans}:
                    plans.append(p)

    # final pad (duplicate if still short)
    while plans and len(plans) < cfg.n_responses:
        plans.append(plans[len(plans) % len(plans)])

    return plans[: cfg.n_responses]


# -----------------------------
# Convenience: make a default config that points to your HumanML3D folder
# -----------------------------
def default_config_for_t2m_gpt(humanml3d_texts_dir: Optional[str] = None, n: int = 8) -> YouThinkConfig:
    return YouThinkConfig(
        n_responses=n,
        humanml3d_texts_dir=humanml3d_texts_dir,
        few_shot_k=6,
        min_actions_per_response=2,
        max_actions_per_response=6,
        output_mode="paren_lines",
        force_humanml3d_style=True,
        temperature=0.9,
        top_p=0.92,
    )
