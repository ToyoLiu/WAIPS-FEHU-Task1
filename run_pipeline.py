"""
FEHU Task-1: Article-level Human Value Classification
Prompt-based Pipeline - Supports OpenAI GPT and Google Gemini

Usage:
    # Run on dev set (to evaluate)
    python run_pipeline.py --mode dev

    # Preview the first dev prompt without calling any API
    python run_pipeline.py --mode dev --limit 1 --dry_run

    # Use Gemini for a quick low-cost dev test
    python run_pipeline.py --mode dev --provider gemini --limit 3

    # Use OpenAI for final test submission
    python run_pipeline.py --mode test --provider openai --model gpt-4o

    # Use DeepSeek for low-cost dev experiments
    python run_pipeline.py --mode dev --provider deepseek --model deepseek-v4-flash --limit 20

    # Try the error-analysis-informed prompt variant
    python run_pipeline.py --mode dev --provider deepseek --model deepseek-v4-flash --prompt_variant balanced --limit 20

    # Try the compact no-reasoning prompt variant
    python run_pipeline.py --mode dev --provider deepseek --model deepseek-v4-flash --prompt_variant compact --limit 20

    # Try the stricter compact variant
    python run_pipeline.py --mode dev --provider deepseek --model deepseek-v4-flash --prompt_variant compact_strict --limit 20

    # Try label-targeted more-shot guidance for difficult labels
    python run_pipeline.py --mode dev --provider deepseek --model deepseek-v4-flash --prompt_variant targeted_more_shot --limit 20

    # Try stricter label-targeted more-shot guidance
    python run_pipeline.py --mode dev --provider deepseek --model deepseek-v4-flash --prompt_variant targeted_more_shot_strict --limit 20

    # Try balanced label-targeted more-shot guidance
    python run_pipeline.py --mode dev --provider deepseek --model deepseek-v4-flash --prompt_variant targeted_more_shot_balanced --limit 20

    # Use OpenAI gpt-4o-mini
    python run_pipeline.py --mode dev --provider openai --model gpt-4o-mini

API Key Setup - set environment variables outside this script:
    Windows PowerShell:
        $env:OPENAI_API_KEY="sk-your-key-here"
        $env:GEMINI_API_KEY="your-gemini-key-here"
        $env:DEEPSEEK_API_KEY="your-deepseek-key-here"
"""

import json
import os
import re
import sys
import time
import argparse
import urllib.request
import urllib.error
from tqdm import tqdm

# ============================================================
# CONFIG - Change these paths if your folder structure differs
# ============================================================

def find_base_dir():
    """Find the ntcir19_fehu project root that contains dataset/ and evaluation.py."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        script_dir,
        os.path.dirname(script_dir),
        os.path.join(script_dir, "ntcir19_fehu-master", "ntcir19_fehu-master"),
        os.path.join(script_dir, "ntcir19_fehu-master"),
    ]
    for candidate in candidates:
        if (
            os.path.isdir(os.path.join(candidate, "dataset"))
            and os.path.exists(os.path.join(candidate, "evaluation.py"))
        ):
            return candidate
    raise FileNotFoundError(
        "Could not find project root. Put run_pipeline.py in the FEHU project root "
        "or next to ntcir19_fehu-master/ntcir19_fehu-master."
    )


BASE_DIR = find_base_dir()

# Data paths
TEST_ARTICLES = os.path.join(BASE_DIR, "dataset", "test", "test_article_event_base.json")
DEV_ARTICLES = os.path.join(BASE_DIR, "dataset", "dev", "dev_event_base.json")
TRAIN_ARTICLES = os.path.join(BASE_DIR, "dataset", "train", "train_event_base.json")
TRAIN_LABELS = os.path.join(BASE_DIR, "dataset", "gold_labels", "train", "train_human_values.json")
DEV_LABELS = os.path.join(BASE_DIR, "dataset", "gold_labels", "dev", "dev_human_values.json")

# Taxonomy paths
HV_DIR = os.path.join(BASE_DIR, "dataset", "hv_categories")
L1_VALUES_PATH = os.path.join(HV_DIR, "human_value_level1_values.json")
L2_VALUES_PATH = os.path.join(HV_DIR, "human_value_level2_values.json")
L1_TO_L2_PATH = os.path.join(HV_DIR, "level_1_to_level_2.json")

# Output paths
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "task1")

# Model config
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
TEMPERATURE = 0.1
DEFAULT_MAX_OUTPUT_TOKENS = 4000
DEEPSEEK_MAX_OUTPUT_TOKENS = 8000
DEFAULT_REQUEST_TIMEOUT = 120
MAX_RETRIES = 3

# ============================================================
# LOAD TAXONOMY
# ============================================================

def load_taxonomy():
    """Load all value category definitions."""
    with open(L1_VALUES_PATH, "r", encoding="utf-8") as f:
        l1_name_to_id = json.load(f)  # {"Be creative": "0", ...}

    with open(L2_VALUES_PATH, "r", encoding="utf-8") as f:
        l2_name_to_id = json.load(f)  # {"Self-direction: thought": "0", ...}

    with open(L1_TO_L2_PATH, "r", encoding="utf-8") as f:
        l1_to_l2_raw = json.load(f)["level_1_to_level_2"]
        # {"Be creative": "Self-direction: thought", ...}

    # Build L1 id -> L2 id mapping
    l1_id_to_l2_id = {}
    for l1_name, l2_name in l1_to_l2_raw.items():
        l1_id = l1_name_to_id[l1_name]
        l2_id = l2_name_to_id[l2_name]
        l1_id_to_l2_id[l1_id] = l2_id

    return l1_name_to_id, l2_name_to_id, l1_id_to_l2_id


# ============================================================
# BUILD FEW-SHOT EXAMPLES FROM TRAINING DATA
# ============================================================

def load_few_shot_examples(n=2):
    """
    Load n training articles with their gold labels as few-shot examples.
    Pick articles with moderate number of actors and clear value assignments.
    """
    with open(TRAIN_ARTICLES, "r", encoding="utf-8") as f:
        train_articles = json.load(f)
    with open(TRAIN_LABELS, "r", encoding="utf-8") as f:
        train_labels = json.load(f)

    # Build guid -> labels mapping
    guid_to_labels = {item["guid"]: item for item in train_labels}

    # Build guid -> article mapping
    guid_to_article = {item["guid"]: item for item in train_articles}

    # Select good examples: articles with 2-3 actors and 4-8 total labels
    candidates = []
    for item in train_labels:
        guid = item["guid"]
        hvs = item.get("article_human_values", [])
        if guid in guid_to_article:
            actors = guid_to_article[guid].get("actors", [])
            if 2 <= len(actors) <= 3 and 4 <= len(hvs) <= 8:
                candidates.append(guid)

    # Take first n candidates
    selected = candidates[:n]

    examples = []
    for guid in selected:
        article = guid_to_article[guid]
        labels = guid_to_labels[guid]

        # Format actor list
        actor_list = []
        for actor_dict in article["actors"]:
            for name, aid in actor_dict.items():
                actor_list.append({"name": name, "id": aid})

        # Format gold labels for the example output
        output_values = []
        for hv in labels["article_human_values"]:
            # Reverse lookup l1 name from id
            l1_id = hv["l1_value"]
            l1_name = [k for k, v in l1_name_to_id.items() if v == l1_id][0]
            output_values.append({
                "actor_id": hv["actor"],
                "l1_value_name": l1_name,
                "l1_value_id": hv["l1_value"],
                "direction": "aligned" if hv["direction"] == "1" else "contradictory",
                "reasoning": hv.get("explanation", "")[:150]  # Truncate for prompt length
            })

        examples.append({
            "title": article["title"],
            "content": article["content"][:1500],  # Truncate long articles
            "actors": actor_list,
            "output": output_values
        })

    return examples


# ============================================================
# BUILD THE PROMPT
# ============================================================

def build_system_prompt(few_shot_examples):
    """Build the system prompt with task description and few-shot examples."""

    # Build the L1 value list string
    l1_list_str = ""
    for name, vid in sorted(l1_name_to_id.items(), key=lambda x: int(x[1])):
        l1_list_str += f"  {vid}: {name}\n"

    system_prompt = f"""You are an expert in human value analysis for news articles.

TASK: Given a factual news article and a list of actors (people, organizations, or groups mentioned in the article), identify which human values each actor's behavior expresses or violates.

HUMAN VALUE CATEGORIES (Level-1, 54 values):
{l1_list_str}
DIRECTION:
- "aligned": The actor's behavior supports, upholds, or expresses this value.
- "contradictory": The actor's behavior violates, undermines, or contradicts this value.

GUIDELINES:
1. For each actor, identify 3-7 relevant values. Don't over-predict.
2. Only assign values that have clear evidence in the article text.
3. Consider both what actors DO and what they SAY.
4. The same actor can have both aligned and contradictory values.
5. Different actors in the same event often have opposing value directions.
6. Common patterns:
   - Law enforcement/authorities: often aligned with "Be just"(45), "Be responsible"(42), "Have a safe country"(26)
   - Victims/affected people: often contradicted on "Have good health"(22), "Have a comfortable life"(25)
   - Organizations responding to crises: often aligned with "Be helpful"(37), "Be honest"(38)
   - Rule-breakers/criminals: often contradictory to "Be compliant"(30), "Be responsible"(42)

OUTPUT FORMAT: Return ONLY a valid JSON array. Each element must have:
- "actor_id": the actor's ID string (e.g., "0-4405-1-3-0")
- "l1_value_id": the Level-1 value ID as a string (e.g., "42")
- "direction": either "aligned" or "contradictory"
- "reasoning": one sentence explaining why (keep it brief)

Do NOT include any text before or after the JSON array. No markdown, no explanation."""

    # Add few-shot examples
    if few_shot_examples:
        system_prompt += "\n\nEXAMPLES:\n"
        for i, ex in enumerate(few_shot_examples):
            actor_str = ", ".join([f'{a["name"]} (ID: {a["id"]})' for a in ex["actors"]])
            output_str = json.dumps(ex["output"][:4], indent=2, ensure_ascii=False)  # Show first 4 for brevity
            system_prompt += f"""
--- Example {i+1} ---
Title: {ex["title"]}
Article (truncated): {ex["content"][:500]}...
Actors: {actor_str}
Output:
{output_str}
"""

    return system_prompt


def build_deepseek_system_prompt(few_shot_examples):
    """DeepSeek JSON mode works best when the output is a JSON object."""
    prompt = build_system_prompt(few_shot_examples)
    prompt = prompt.replace(
        "OUTPUT FORMAT: Return ONLY a valid JSON array. Each element must have:",
        'OUTPUT FORMAT: Return ONLY a valid JSON object with an "items" array. '
        'Each element in "items" must have:'
    ).replace(
        '- "reasoning": one sentence explaining why (keep it brief)',
        '- "reasoning": optional; if included, use at most 8 words'
    ).replace(
        "Do NOT include any text before or after the JSON array. No markdown, no explanation.",
        'Do NOT include any text before or after the JSON object. No markdown, no explanation.\n\n'
        'Example JSON output:\n'
        '{\n'
        '  "items": [\n'
        '    {\n'
        '      "actor_id": "0-4405-1-3-0",\n'
        '      "l1_value_id": "42",\n'
        '      "direction": "aligned",\n'
        '      "reasoning": "The actor takes responsibility for public safety."\n'
        '    }\n'
        '  ]\n'
        '}'
    )
    return prompt


def build_user_prompt(article):
    """Build the user prompt for a single article."""
    # Build actor list
    actor_lines = []
    for actor_dict in article["actors"]:
        for name, aid in actor_dict.items():
            actor_lines.append(f"- {name} (ID: {aid})")
    actor_str = "\n".join(actor_lines)

    # Truncate very long articles to fit context window
    content = article["content"]
    if len(content) > 12000:
        content = content[:6000] + "\n...[truncated]...\n" + content[-3000:]

    return f"""Title: {article["title"]}

Article:
{content}

Actors:
{actor_str}

Identify the human values for each actor. Return ONLY a JSON array."""


def build_deepseek_user_prompt(article):
    """Build a DeepSeek JSON-mode prompt that asks for a JSON object."""
    prompt = build_user_prompt(article)
    return prompt.replace(
        "Identify the human values for each actor. Return ONLY a JSON array.",
        'Identify the human values for each actor. Return ONLY a JSON object with an "items" array.'
    )


def apply_prompt_variant(system_prompt, variant):
    """Apply lightweight prompt variants derived from dev error analysis."""
    if variant == "baseline":
        return system_prompt

    if variant not in {"balanced", "compact", "compact_strict", "targeted_more_shot", "targeted_more_shot_strict", "targeted_more_shot_balanced"}:
        raise ValueError(f"Unknown prompt variant: {variant}")

    if variant in {"compact", "compact_strict", "targeted_more_shot", "targeted_more_shot_strict", "targeted_more_shot_balanced"}:
        compact_prompt = system_prompt
        compact_prompt = compact_prompt.replace(
            '- "reasoning": optional; if included, use at most 8 words',
            'Do NOT include a "reasoning" field.'
        ).replace(
            '- "reasoning": one sentence explaining why (keep it brief)',
            'Do NOT include a "reasoning" field.'
        )
        compact_prompt = re.sub(
            r'("direction":\s*"(?:aligned|contradictory)"),\s*\n\s*"reasoning":\s*"[^"]*"',
            r'\1',
            compact_prompt,
        )
        compact_text = """

COMPACT OUTPUT CALIBRATION:
1. Output only three fields for each item: "actor_id", "l1_value_id", and "direction".
2. Do not output "reasoning", explanations, evidence text, markdown, or comments.
3. For each actor, output at most 6 values.
4. A value must be explicit or strongly implied by actor-specific evidence in the article.
5. Avoid generic moral labels that merely sound related to the event.
6. If evidence is weak or only article-topic-level, omit the value.
"""
        if variant in {"compact_strict", "targeted_more_shot_strict"}:
            compact_text += """

STRICT SELECTION RULES:
1. For each actor, output at most 5 values.
2. Prefer precision over recall. It is better to miss a weak value than to add an unsupported one.
3. If several similar values could apply, keep only the most specific one.
   - Do not output both "Be helpful"(37) and "Be responsible"(42) unless the article separately supports both.
   - Do not output both "Have a safe country"(26) and "Have a stable society"(27) unless both personal/national safety and social order are explicit.
   - Do not output "Be just"(45) for authorities unless justice, law enforcement, accountability, or rights are central to the actor's behavior.
4. If direction is uncertain, omit the value.
5. Do not infer positive values for victims merely because they are harmed.
6. Do not infer aligned values for organizations merely because they issue statements.
"""
        if variant in {"targeted_more_shot", "targeted_more_shot_strict", "targeted_more_shot_balanced"}:
            compact_text += """

TARGETED MORE-SHOT GUIDANCE FOR DIFFICULT LABELS:
Use these reasoning-style examples as calibration. Do NOT output reasoning in the final answer.

Example A: health impact
- Situation: An actor provides medical support, warns the public about disease/drugs, prevents injury, or improves access to treatment.
- Label: "Have good health"(22), direction "aligned".
- Reasoning pattern: the actor's behavior protects or improves people's physical or mental health.
- Avoid confusion: do not add this label only because the article is generally about a crisis; the actor must be connected to health protection.

Example B: health harm
- Situation: An actor causes injury, disease exposure, unsafe conditions, neglects health risks, or blocks treatment.
- Label: "Have good health"(22), direction "contradictory".
- Reasoning pattern: the actor's behavior threatens or harms people's health.
- Avoid confusion: if the actor is only a victim of harm, do not infer unrelated positive values for that actor.

Example C: public order or stability
- Situation: An actor prevents riots, crime, institutional disruption, social unrest, or restores public order.
- Label: "Have a stable society"(27), direction "aligned".
- Reasoning pattern: the actor's behavior maintains social order or institutional stability.
- Related label: use "Have a safe country"(26) when the evidence is more directly about public/national safety.

Example D: destabilizing action
- Situation: An actor commits violence, organized crime, public disorder, corruption, or actions that threaten social institutions.
- Label: "Have a stable society"(27), direction "contradictory".
- Reasoning pattern: the actor's behavior destabilizes society or undermines public order.

Example E: social recognition / reputation
- Situation: The article explicitly discusses public status, honor, shame, reputation, recognition, criticism, or loss of face.
- Label: "Have social recognition"(19) or "Have a good reputation"(20), choose the most specific supported value and direction.
- Avoid confusion: do not select these labels only because an actor is mentioned in news; there must be evidence about public evaluation.

TARGETED RECOVERY CHECK:
Before finalizing each actor, re-check whether the article explicitly or strongly implies any of these often-missed values:
- "Have good health"(22)
- "Have a stable society"(27)
- "Have a safe country"(26)
- "Have social recognition"(19)
- "Have a good reputation"(20)
- "Be capable"(13)

Keep the compact output format. Recover these labels only when actor-specific evidence is present.
"""
        if variant == "targeted_more_shot_balanced":
            compact_text += """

BALANCED TARGETED CALIBRATION:
1. Keep recall for target labels, but require one concrete actor-specific clue.
2. For "Have good health"(22):
   - Use aligned when the actor warns, treats, protects, rescues, prevents harm, or provides health-related support.
   - Use contradictory when the actor causes injury, illness, unsafe exposure, health risk, or blocks care.
   - Do not use for a general crisis unless health impact or health protection is directly mentioned.
3. For "Have a stable society"(27):
   - Use aligned for maintaining/restoring public order, institutions, or social stability.
   - Use contradictory for violence, crime, unrest, corruption, or disruption of public order.
   - If the evidence is mainly about physical/public safety, prefer "Have a safe country"(26).
4. For "Have social recognition"(19) / "Have a good reputation"(20):
   - Use only when public status, recognition, reputation, honor, criticism, or shame is explicitly discussed.
5. Direction check for target labels:
   - actor mitigates/protects/restores -> usually aligned.
   - actor causes harm/disruption/risk -> usually contradictory.
6. If a target label has weak but plausible evidence, keep it only when it is one of the actor's top 6 most relevant values.
"""
        if variant == "targeted_more_shot_strict":
            compact_text += """

TARGETED STRICTNESS RULES:
1. Do not add a target label only because it appears in the TARGETED RECOVERY CHECK list.
2. For "Have good health"(22):
   - aligned requires explicit prevention, treatment, warning, rescue, medical support, or health protection by the actor.
   - contradictory requires explicit injury, disease, health risk, unsafe exposure, or neglect caused by or affecting the actor.
   - Do not use for general suffering, inconvenience, or political conflict without health evidence.
3. For "Have a stable society"(27):
   - aligned requires restoring or maintaining social order, institutions, or public stability.
   - contradictory requires concrete disorder, violence, crime, unrest, corruption, or institutional disruption.
   - Do not use only because an event is negative or controversial.
4. For "Have social recognition"(19) / "Have a good reputation"(20):
   - require explicit public praise, blame, shame, honor, status, reputation, or recognition.
   - Do not use only because an actor is named, criticized, or involved in news.
5. If the target label is plausible but not actor-specific, omit it.
6. If direction is uncertain for a target label, omit it.
"""
        marker = "OUTPUT FORMAT:"
        if marker in compact_prompt:
            return compact_prompt.replace(marker, compact_text + "\n" + marker, 1)
        return compact_prompt + compact_text

    variant_text = """

ERROR-ANALYSIS-INFORMED CALIBRATION:
1. Avoid generic moral labels unless the article gives actor-specific evidence.
   - Use "Be helpful"(37) only when the actor actively assists, protects, rescues, informs, or supports others.
   - Use "Be just"(45) only when the actor enforces rights/law/fairness, seeks accountability, or remedies harm.
   - Use "Be logical"(52) only when the actor uses explicit evidence, investigation, analysis, or reasoning.
   - Use "Be honest"(38) only when the actor truthfully discloses, admits, reports, or corrects information.
   - Use "Be responsible"(42) only when the actor has a duty and fulfills or violates that duty.
2. Always check commonly missed social/security values:
   - "Have a stable society"(27): public order, unrest, war, institutional disruption, social instability.
   - "Have a sense of belonging"(21): membership, inclusion, community identity, displacement, exclusion.
   - "Have social recognition"(19): public status, reputation, recognition, shame, prestige.
   - "Be respecting traditions"(28): customs, religion, heritage, ceremonies, long-standing practices.
3. Direction check:
   - If the actor causes harm/disruption, the value is usually "contradictory".
   - If the actor mitigates harm, protects people, or restores order, the value is usually "aligned".
4. Prefer 4-8 high-evidence values per actor. Do not add values merely because they sound morally related.
"""
    marker = "OUTPUT FORMAT:"
    if marker in system_prompt:
        return system_prompt.replace(marker, variant_text + "\n" + marker, 1)
    return system_prompt + variant_text


def parse_model_json(content, article_guid):
    """Parse a JSON array from an LLM response."""
    content = content.strip()

    # Sometimes models wrap JSON in ```json ... ```
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    # Gemini may occasionally emit typographic quotes, which are not valid JSON.
    content = (
        content.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )

    result = json.loads(content)
    if isinstance(result, dict) and "items" in result:
        result = result["items"]
    elif isinstance(result, dict) and "article_human_values" in result:
        result = result["article_human_values"]
    elif not isinstance(result, list):
        print(f"  [WARN] {article_guid}: Response is not a list, wrapping it")
        result = [result]
    return result


# ============================================================
# CALL MODEL API
# ============================================================

def call_openai(client, system_prompt, user_prompt, article_guid, model_name, temperature, max_tokens, request_timeout):
    """Call OpenAI API with retry logic."""
    for attempt in range(MAX_RETRIES):
        content = ""
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_completion_tokens=max_tokens,
                timeout=request_timeout,
            )

            content = response.choices[0].message.content.strip()
            return parse_model_json(content, article_guid)

        except json.JSONDecodeError as e:
            print(f"  [WARN] {article_guid}: JSON parse error (attempt {attempt+1}): {e}")
            print(f"  Response was: {content[:200]}...")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
            continue

        except Exception as e:
            print(f"  [ERROR] {article_guid}: API error (attempt {attempt+1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(5)
            continue

    print(f"  [FAIL] {article_guid}: All retries failed, returning empty")
    return []


def call_deepseek(client, system_prompt, user_prompt, article_guid, model_name, temperature, max_tokens, request_timeout):
    """Call DeepSeek OpenAI-compatible API with JSON output enabled."""
    for attempt in range(MAX_RETRIES):
        content = ""
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                timeout=request_timeout,
            )

            content = response.choices[0].message.content.strip()
            if not content:
                raise ValueError("DeepSeek returned empty content")
            return parse_model_json(content, article_guid)

        except json.JSONDecodeError as e:
            print(f"  [WARN] {article_guid}: JSON parse error (attempt {attempt+1}): {e}")
            print(f"  Response was: {content[:200]}...")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
            continue

        except Exception as e:
            print(f"  [ERROR] {article_guid}: API error (attempt {attempt+1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(5)
            continue

    print(f"  [FAIL] {article_guid}: All retries failed, returning empty")
    return []


def call_gemini(client, system_prompt, user_prompt, article_guid, model_name, temperature, max_tokens, request_timeout):
    """Call Gemini API with retry logic."""
    if isinstance(client, dict) and client.get("transport") == "rest":
        return call_gemini_rest(
            client["api_key"], system_prompt, user_prompt, article_guid, model_name, temperature, max_tokens, request_timeout
        )

    from google.genai import types

    for attempt in range(MAX_RETRIES):
        content = ""
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            content = response.text.strip()
            return parse_model_json(content, article_guid)

        except json.JSONDecodeError as e:
            print(f"  [WARN] {article_guid}: JSON parse error (attempt {attempt+1}): {e}")
            print(f"  Response was: {content[:200]}...")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
            continue

        except Exception as e:
            print(f"  [ERROR] {article_guid}: API error (attempt {attempt+1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(5)
            continue

    print(f"  [FAIL] {article_guid}: All retries failed, returning empty")
    return []


def call_gemini_rest(api_key, system_prompt, user_prompt, article_guid, model_name, temperature, max_tokens, request_timeout):
    """Call Gemini via the REST API. This avoids requiring google-genai on Python 3.8."""
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/{model_name}:generateContent"
    )
    payload = {
        "system_instruction": {
            "parts": [{"text": system_prompt}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt}]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    for attempt in range(MAX_RETRIES):
        content = ""
        try:
            request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(request, timeout=request_timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            content = data["candidates"][0]["content"]["parts"][0]["text"]
            return parse_model_json(content, article_guid)

        except json.JSONDecodeError as e:
            print(f"  [WARN] {article_guid}: JSON parse error (attempt {attempt+1}): {e}")
            print(f"  Response was: {content[:200]}...")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
            continue

        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            print(f"  [ERROR] {article_guid}: Gemini HTTP error (attempt {attempt+1}): {e.code} {detail[:300]}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(5)
            continue

        except Exception as e:
            print(f"  [ERROR] {article_guid}: Gemini API error (attempt {attempt+1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(5)
            continue

    print(f"  [FAIL] {article_guid}: All retries failed, returning empty")
    return []


def create_client(provider, api_key):
    """Create the requested model provider client."""
    if provider == "openai":
        from openai import OpenAI
        return OpenAI(api_key=api_key)

    if provider == "deepseek":
        from openai import OpenAI
        return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    if provider == "gemini":
        try:
            from google import genai
        except ImportError:
            print("Google GenAI SDK not found; using Gemini REST API fallback.")
            return {"transport": "rest", "api_key": api_key}
        return genai.Client(api_key=api_key)

    raise ValueError(f"Unsupported provider: {provider}")


def call_model(client, provider, system_prompt, user_prompt, article_guid, model_name, temperature, max_tokens, request_timeout):
    if provider == "openai":
        return call_openai(client, system_prompt, user_prompt, article_guid, model_name, temperature, max_tokens, request_timeout)
    if provider == "deepseek":
        return call_deepseek(client, system_prompt, user_prompt, article_guid, model_name, temperature, max_tokens, request_timeout)
    if provider == "gemini":
        return call_gemini(client, system_prompt, user_prompt, article_guid, model_name, temperature, max_tokens, request_timeout)
    raise ValueError(f"Unsupported provider: {provider}")


# ============================================================
# PROCESS RESULTS -> OUTPUT FORMAT
# ============================================================

def process_results(article_guid, actors, gpt_results):
    """
    Convert GPT output to the official submission format.
    Produces entries for both task1a and task1b.
    """
    # Get valid actor IDs for this article
    valid_actor_ids = set()
    for actor_dict in actors:
        for name, aid in actor_dict.items():
            valid_actor_ids.add(aid)

    task1a_values = []  # For pred_task1a.json
    task1b_values = []  # For pred_task1b.json

    seen_task1a = set()  # Deduplicate (actor, l2_value)
    seen_task1b = set()  # Deduplicate (actor, direction, l1_value)

    for item in gpt_results:
        actor_id = str(item.get("actor_id", ""))
        l1_id = str(item.get("l1_value_id", ""))
        direction_str = item.get("direction", "aligned")

        # Validate actor_id
        if actor_id not in valid_actor_ids:
            continue

        # Validate l1_value_id
        if not l1_id.isdigit() or int(l1_id) < 0 or int(l1_id) > 53:
            continue

        # Convert direction
        if direction_str == "contradictory":
            direction = "0"
        else:
            direction = "1"  # Default to aligned

        # Map L1 -> L2
        l2_id = l1_id_to_l2_id.get(l1_id, None)
        if l2_id is None:
            continue

        # Task 1a: (actor, l2_value) - deduplicate
        key_1a = (actor_id, l2_id)
        if key_1a not in seen_task1a:
            seen_task1a.add(key_1a)
            task1a_values.append({
                "actor": actor_id,
                "l2_value": l2_id
            })

        # Task 1b: (actor, direction, l1_value) - deduplicate
        key_1b = (actor_id, direction, l1_id)
        if key_1b not in seen_task1b:
            seen_task1b.add(key_1b)
            task1b_values.append({
                "actor": actor_id,
                "l1_value": l1_id,
                "direction": direction
            })

    return task1a_values, task1b_values


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="FEHU Task-1 Prompt Pipeline")
    parser.add_argument("--mode", choices=["dev", "test"], default="dev",
                        help="Run on dev set (for evaluation) or test set (for submission)")
    parser.add_argument("--provider", choices=["openai", "gemini", "deepseek"], default="gemini",
                        help="Model provider to use (default: gemini)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only N articles (for quick testing)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name. Defaults to gemini-2.5-flash or gpt-4o-mini.")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                        help="Sampling temperature (default: 0.1)")
    parser.add_argument("--prompt_variant", choices=["baseline", "balanced", "compact", "compact_strict", "targeted_more_shot", "targeted_more_shot_strict", "targeted_more_shot_balanced"], default="baseline",
                        help="Prompt variant to use. 'balanced' is error-analysis-informed; compact variants remove reasoning.")
    parser.add_argument("--max_output_tokens", type=int, default=None,
                        help="Maximum output tokens. Defaults to 8000 for DeepSeek, 4000 otherwise.")
    parser.add_argument("--request_timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT,
                        help="Per-request API timeout in seconds (default: 120).")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key. Prefer environment variables over passing this on the command line.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Build and print one prompt preview without calling any API")
    parser.add_argument("--refresh_cache", action="store_true",
                        help="Ignore existing cache entries and call the API again")
    args = parser.parse_args()

    default_models = {
        "openai": DEFAULT_OPENAI_MODEL,
        "gemini": DEFAULT_GEMINI_MODEL,
        "deepseek": DEFAULT_DEEPSEEK_MODEL,
    }
    model_name = args.model or default_models[args.provider]
    max_output_tokens = args.max_output_tokens or (
        DEEPSEEK_MAX_OUTPUT_TOKENS if args.provider == "deepseek" else DEFAULT_MAX_OUTPUT_TOKENS
    )

    # Load taxonomy (global for use in other functions)
    global l1_name_to_id, l2_name_to_id, l1_id_to_l2_id
    l1_name_to_id, l2_name_to_id, l1_id_to_l2_id = load_taxonomy()

    # Load articles
    if args.mode == "dev":
        articles_path = DEV_ARTICLES
        print(f"Mode: DEV (will evaluate against gold labels)")
    else:
        articles_path = TEST_ARTICLES
        print(f"Mode: TEST (will produce submission files)")

    with open(articles_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    if args.limit:
        articles = articles[:args.limit]

    print(f"Loaded {len(articles)} articles")
    print(f"Project root: {BASE_DIR}")
    print(f"Provider: {args.provider}")
    print(f"Model: {model_name}")
    print(f"Temperature: {args.temperature}")
    print(f"Max output tokens: {max_output_tokens}")
    print(f"Request timeout: {args.request_timeout}s")
    print(f"Prompt variant: {args.prompt_variant}")

    # Load few-shot examples from training data
    print("Loading few-shot examples from training data...")
    few_shot_examples = load_few_shot_examples(n=2)
    print(f"Loaded {len(few_shot_examples)} few-shot examples")

    # Build system prompt (same for all articles)
    if args.provider == "deepseek":
        system_prompt = build_deepseek_system_prompt(few_shot_examples)
    else:
        system_prompt = build_system_prompt(few_shot_examples)
    system_prompt = apply_prompt_variant(system_prompt, args.prompt_variant)

    if args.dry_run:
        if not articles:
            print("No articles loaded.")
            return
        preview_prompt = (
            build_deepseek_user_prompt(articles[0])
            if args.provider == "deepseek"
            else build_user_prompt(articles[0])
        )
        print("\n" + "="*60)
        print("SYSTEM PROMPT PREVIEW")
        print("="*60)
        print(system_prompt[:3000])
        if len(system_prompt) > 3000:
            print(f"\n...[truncated, total {len(system_prompt)} chars]...")
        print("\n" + "="*60)
        print("USER PROMPT PREVIEW")
        print("="*60)
        print(preview_prompt[:3000])
        if len(preview_prompt) > 3000:
            print(f"\n...[truncated, total {len(preview_prompt)} chars]...")
        return

    # Initialize API client after dry-run so prompt previews do not need a key.
    env_key_names = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
    }
    env_key_name = env_key_names[args.provider]
    api_key = args.api_key or os.environ.get(env_key_name)
    if not api_key:
        print("ERROR: No API key provided!")
        print(f"Set the {env_key_name} environment variable, for example:")
        print(f'  Windows PowerShell: $env:{env_key_name}="your_key_here"')
        print("Or pass --api_key, but environment variables are safer.")
        sys.exit(1)

    client = create_client(args.provider, api_key)

    # Process each article
    all_task1a = []
    all_task1b = []

    # Cache file for resuming interrupted runs
    safe_model_name = model_name.replace("/", "_").replace(":", "_")
    temp_tag = str(args.temperature).replace(".", "p")
    token_tag = str(max_output_tokens)
    variant_tag = args.prompt_variant
    cache_path = os.path.join(
        BASE_DIR,
        "logs",
        f"cache_{args.mode}_{args.provider}_{safe_model_name}_{variant_tag}_temp{temp_tag}_max{token_tag}.json",
    )
    os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"Loaded cache with {len(cache)} processed articles")

    for article in tqdm(articles, desc="Processing articles"):
        guid = article["guid"]

        # Check cache
        if guid in cache and not args.refresh_cache:
            gpt_results = cache[guid]
        else:
            # Build prompt and call API
            user_prompt = (
                build_deepseek_user_prompt(article)
                if args.provider == "deepseek"
                else build_user_prompt(article)
            )
            gpt_results = call_model(
                client,
                args.provider,
                system_prompt,
                user_prompt,
                guid,
                model_name,
                args.temperature,
                max_output_tokens,
                args.request_timeout,
            )

            # Save to cache
            cache[guid] = gpt_results
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False)

        # Process results
        task1a_values, task1b_values = process_results(guid, article["actors"], gpt_results)

        all_task1a.append({
            "guid": guid,
            "article_human_values": task1a_values
        })

        all_task1b.append({
            "guid": guid,
            "article_human_values": task1b_values
        })

    # Save output files. Keep each run separate so prompt/model comparisons
    # do not overwrite earlier predictions.
    run_tag = f"{args.mode}_{args.provider}_{safe_model_name}_{variant_tag}_temp{temp_tag}_max{token_tag}"
    output_dir = os.path.join(OUTPUT_DIR, "runs", run_tag)
    os.makedirs(output_dir, exist_ok=True)

    task1a_path = os.path.join(output_dir, "pred_task1a.json")
    task1b_path = os.path.join(output_dir, "pred_task1b.json")

    with open(task1a_path, "w", encoding="utf-8") as f:
        json.dump(all_task1a, f, indent=2, ensure_ascii=False)
    print(f"\nSaved Task-1a predictions to: {task1a_path}")

    with open(task1b_path, "w", encoding="utf-8") as f:
        json.dump(all_task1b, f, indent=2, ensure_ascii=False)
    print(f"Saved Task-1b predictions to: {task1b_path}")

    # Print stats
    total_1a = sum(len(item["article_human_values"]) for item in all_task1a)
    total_1b = sum(len(item["article_human_values"]) for item in all_task1b)
    print(f"\nStats:")
    print(f"  Articles processed: {len(articles)}")
    print(f"  Task-1a predictions: {total_1a} (avg {total_1a/len(articles):.1f} per article)")
    print(f"  Task-1b predictions: {total_1b} (avg {total_1b/len(articles):.1f} per article)")

    # If dev mode, run evaluation
    if args.mode == "dev":
        print("\n" + "="*60)
        print("RUNNING EVALUATION ON DEV SET")
        print("="*60)
        eval_guids = {article["guid"] for article in articles} if args.limit else None
        if eval_guids:
            print(f"Evaluating only the {len(eval_guids)} processed dev articles.")
        run_dev_evaluation(task1a_path, task1b_path, eval_guids=eval_guids)


def filter_by_guid(labels, guids):
    """Keep only evaluator instances whose guid is in guids."""
    if guids is None:
        return labels
    return {iid: values for iid, values in labels.items() if iid[0] in guids}


def run_dev_evaluation(pred_1a_path, pred_1b_path, eval_guids=None):
    """Run the official evaluation script on dev predictions."""
    # Import evaluation functions directly
    sys.path.insert(0, BASE_DIR)
    from evaluation import (
        parse_task1a, parse_task1b, evaluate,
        l2_universe, l1_dir_universe,
        direction_reverse_rate_gold_excl
    )

    # Parse gold and predictions
    gold_1a = parse_task1a(DEV_LABELS)
    pred_1a = parse_task1a(pred_1a_path)
    gold_1b = parse_task1b(DEV_LABELS)
    pred_1b = parse_task1b(pred_1b_path)

    gold_1a = filter_by_guid(gold_1a, eval_guids)
    pred_1a = filter_by_guid(pred_1a, eval_guids)
    gold_1b = filter_by_guid(gold_1b, eval_guids)
    pred_1b = filter_by_guid(pred_1b, eval_guids)

    # Evaluate Task 1a
    results_1a = evaluate(gold_1a, pred_1a, l2_universe())
    print(f"\nTask-1a (Level-2, no direction):")
    print(f"  Micro F1: {results_1a['micro_f1']:.4f}")
    print(f"  Macro F1: {results_1a['macro_f1']:.4f}")
    print(f"  Precision: {results_1a['micro_precision']:.4f}")
    print(f"  Recall: {results_1a['micro_recall']:.4f}")

    # Evaluate Task 1b
    results_1b = evaluate(gold_1b, pred_1b, l1_dir_universe())
    drr, rev_cnt, denom_cnt, amb_cnt = direction_reverse_rate_gold_excl(gold_1b, pred_1b)
    print(f"\nTask-1b (Level-1 + direction):")
    print(f"  Micro F1: {results_1b['micro_f1']:.4f}")
    print(f"  Macro F1: {results_1b['macro_f1']:.4f}")
    print(f"  Precision: {results_1b['micro_precision']:.4f}")
    print(f"  Recall: {results_1b['micro_recall']:.4f}")
    print(f"  Direction Reverse Rate: {drr:.4f} ({rev_cnt}/{denom_cnt})")

    return results_1a, results_1b


if __name__ == "__main__":
    main()
