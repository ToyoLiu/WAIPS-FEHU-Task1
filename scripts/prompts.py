"""
FEHU Task-1: Prompt Templates for Human Value Classification

This module contains the system prompt and user prompt templates
used in the GPT-4o baseline pipeline.

Usage:
    from prompts import build_system_prompt, build_user_prompt
"""

import json

# ============================================================
# Level-1 Human Value Categories (54 values)
# ============================================================

L1_VALUES = {
    "Be creative": "0", "Be curious": "1", "Have freedom of thought": "2",
    "Be choosing own goals": "3", "Be independent": "4", "Have freedom of action": "5",
    "Have privacy": "6", "Have an exciting life": "7", "Have a varied life": "8",
    "Be daring": "9", "Have pleasure": "10", "Be ambitious": "11",
    "Have success": "12", "Be capable": "13", "Be intellectual": "14",
    "Be courageous": "15", "Have influence": "16", "Have the right to command": "17",
    "Have wealth": "18", "Have social recognition": "19", "Have a good reputation": "20",
    "Have a sense of belonging": "21", "Have good health": "22", "Have no debts": "23",
    "Be neat and tidy": "24", "Have a comfortable life": "25", "Have a safe country": "26",
    "Have a stable society": "27", "Be respecting traditions": "28",
    "Be holding religious faith": "29", "Be compliant": "30", "Be self-disciplined": "31",
    "Be behaving properly": "32", "Be polite": "33", "Be honoring elders": "34",
    "Be humble": "35", "Have life accepted as is": "36", "Be helpful": "37",
    "Be honest": "38", "Be forgiving": "39", "Have the own family secured": "40",
    "Be loving": "41", "Be responsible": "42", "Have loyalty towards friends": "43",
    "Have equality": "44", "Be just": "45", "Have a world at peace": "46",
    "Be protecting the environment": "47", "Have harmony with nature": "48",
    "Have a world of beauty": "49", "Be broadminded": "50",
    "Have the wisdom to accept others": "51", "Be logical": "52",
    "Have an objective view": "53",
}


# ============================================================
# System Prompt
# ============================================================

def build_system_prompt(few_shot_examples=None):
    """
    Build the system prompt with:
    1. Task definition
    2. Full L1 value list (54 values)
    3. Direction definitions
    4. Domain-specific guidelines
    5. Output format specification
    6. Optional few-shot examples from training data

    Args:
        few_shot_examples: list of dicts, each with keys:
            - title (str)
            - content (str, truncated)
            - actors (list of {name, id})
            - output (list of {actor_id, l1_value_name, l1_value_id, direction, reasoning})

    Returns:
        str: complete system prompt
    """

    # Build the L1 value list string
    l1_list_str = ""
    for name, vid in sorted(L1_VALUES.items(), key=lambda x: int(x[1])):
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
            output_str = json.dumps(ex["output"][:4], indent=2, ensure_ascii=False)
            system_prompt += f"""
--- Example {i+1} ---
Title: {ex["title"]}
Article (truncated): {ex["content"][:500]}...
Actors: {actor_str}
Output:
{output_str}
"""

    return system_prompt


# ============================================================
# User Prompt
# ============================================================

def build_user_prompt(article):
    """
    Build the user prompt for a single article.

    Args:
        article: dict with keys:
            - title (str)
            - content (str): full article text
            - actors (list of dicts): [{actor_name: actor_id}, ...]

    Returns:
        str: user prompt
    """
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


# ============================================================
# Quick test / preview
# ============================================================

if __name__ == "__main__":
    # Print the system prompt (without few-shot examples) for inspection
    prompt = build_system_prompt()
    print("=" * 60)
    print("SYSTEM PROMPT")
    print("=" * 60)
    print(prompt)
    print(f"\n[Total length: {len(prompt)} characters]")

    # Example user prompt
    example_article = {
        "title": "Example: Bus crash injures dozens",
        "content": "A double-decker bus overturned, leaving dozens injured...",
        "actors": [
            {"Emergency Services": "0-0001-0"},
            {"Bus Company": "0-0001-1"},
        ]
    }
    print("\n" + "=" * 60)
    print("USER PROMPT (example)")
    print("=" * 60)
    print(build_user_prompt(example_article))
