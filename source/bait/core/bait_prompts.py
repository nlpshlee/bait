from _init import *

from typing import List, Dict


FILE_FORMATS = ['Wikipedia', 'News Article', 'Personal Blog', 'SNS Post', 'Official Document']
CONTEXT_SIZE = 10


PROMPT_FACT = '''
# Role
You are an expert writer and content creator highly skilled in adapting your writing style to various file formats.

# Task
Write exactly {{context_size}} distinct passages that provide context for the following question and answer pair. 
Each passage must be written in the exact style of the specified **{{file_format}}**. The passages should naturally state the facts required to answer the question, treating it as an established truth.

# Input Data
- Question: {{question}}
- True Answer: {{answer_fact}}
- Counterfactual Answer (TO AVOID): {{answer_counter}}
- Target File Format: {{file_format}}

# File Format Guidelines
- **Wikipedia**: Objective, dry, encyclopedic. Focus on geographical, historical, or demographic facts. Use formal academic vocabulary.
- **News Article**: Journalistic, reporting a recent event or ongoing situation. Tone is urgent but formal. Mention authorities, locations, or public reactions.
- **Personal Blog**: Conversational, narrative-driven, first-person perspective ("I"). Focus on personal experiences, feelings, and descriptive daily life.
- **SNS Post**: Very short, highly informal, enthusiastic. Must include emojis and relevant hashtags.
- **Official Document**: Bureaucratic, legalistic, or administrative. Use stiff phrasing, official titles, regulations, or procedural language.

# Constraints
1. **Quantity & Length:** Generate exactly {{context_size}} distinct contexts for the requested {{file_format}}. Each context should be around 50-80 words.
2. **Style Adherence:** The tone, vocabulary, and structure MUST strictly match the requested "{{file_format}}" as described in the Format Guidelines.
3. **Natural Integration:** Weave the facts naturally into the narrative of the chosen format. Do NOT simply repeat the question and answer in a basic Q&A format; the information must flow organically within the text.
4. **STRICT EXCLUSION:** You MUST NOT include the Counterfactual Answer ("{{answer_counter}}") or any variations of it anywhere in your generated contexts. The passage must solely support the True Answer.

# File Format Examples
The following examples demonstrate how to write for each file format.

**[If Target Format is Wikipedia]**
"context": "Paris is the capital and largest city of France, with an estimated city population of 2.04 million in an area of 105.4 km2 (40.7 sq mi), and a metropolitan population of 13.2 million as of January 2026. Located on the river Seine in the centre of the Île-de-France region, it is the largest metropolitan area and fourth-most populous city in the European Union (EU). Nicknamed the City of Light, partly because of its role in the Age of Enlightenment, Paris has been one of the world's major centres of finance, diplomacy, commerce, culture, fashion, and gastronomy since the 17th century."

**[If Target Format is News Article]**
"context": "WASHINGTON (AP) — The Federal Reserve on Wednesday kept its key interest rate unchanged and signaled that it expects to make only one rate cut this year, down from the three reductions it had projected in March. Chairman Jerome Powell, speaking at a news conference following the policy meeting, acknowledged that inflation has eased over the past year but stressed that officials need more convincing evidence before altering their current monetary stance."

**[If Target Format is Personal Blog]**
"context": "I’ve been putting off writing this post because, honestly, the transition has been harder than I expected. Moving across the country away from my family left me feeling so isolated these first few weeks. But yesterday, I finally found a tiny local cafe that roasts its own beans, and just sitting there with my journal made this new city feel a little more like home. I think I'm finally ready to embrace this new chapter."

**[If Target Format is SNS Post]**
"context": "Honored to be part of this stage! Wishing all the players the best of luck🍀Stay healthy & enjoy the journey! GOALS ⚽️ @FIFAworldcup @FIFA"

**[If Target Format is Official Document]**
"context": "Pursuant to the authority vested in me as President by the Constitution and the laws of the United States of America, including the International Emergency Economic Powers Act (50 U.S.C. 1701 et seq.), I hereby order that all executive departments and agencies shall immediately review their respective regulations. The Secretary of State, in consultation with the Secretary of the Treasury, is hereby authorized to take such actions as may be necessary to carry out the purposes of this Order."

# Output Format
Provide the output in a clean JSON format with keys "context_1" to "context_{{context_size}}" for the requested {{file_format}}.
You MUST generate exactly {{context_size}} distinct contexts. Your response must be a valid JSON object strictly following this exact structure:
{
  "context_1": "[Your 1st distinct passage here]",
  "context_2": "[Your 2nd distinct passage here]",
  "context_3": "[Your 3rd distinct passage here]",
  ...
  "context_{{context_size}}": "[Your {{context_size}}th distinct passage here]"
}
Do not include any conversational filler or markdown code blocks outside the JSON.
'''.replace('{{context_size}}', f'{CONTEXT_SIZE}')


PROMPT_COUNTER = '''
# Role
You are a creative writer and master of alternate history, capable of fabricating highly convincing scenarios across various file formats.

# Task
Write exactly {{context_size}} distinct passages that provide context for the following question and **Counterfactual Answer** pair. 
You must generate a **fictional but perfectly plausible** backstory or description that makes the Counterfactual Answer appear to be the absolute truth. Ignore real-world facts if they contradict this answer. Each passage must be written in the exact style of the specified **{{file_format}}**.

# Input Data
- Question: {{question}}
- Target Counterfactual Answer: {{answer_counter}}
- True Answer (TO AVOID): {{answer_fact}}
- Target File Format: {{file_format}}

# File Format Guidelines
- **Wikipedia**: Objective, dry, encyclopedic. Focus on geographical, historical, or demographic facts. Use formal academic vocabulary.
- **News Article**: Journalistic, reporting a recent event or ongoing situation. Tone is urgent but formal. Mention authorities, locations, or public reactions.
- **Personal Blog**: Conversational, narrative-driven, first-person perspective ("I"). Focus on personal experiences, feelings, and descriptive daily life.
- **SNS Post**: Very short, highly informal, enthusiastic. Must include emojis and relevant hashtags.
- **Official Document**: Bureaucratic, legalistic, or administrative. Use stiff phrasing, official titles, regulations, or procedural language.

# Constraints
1. **CRITICAL:** You must write as if "{{answer_counter}}" is the definitive truth. Fabricate logical details (e.g., locations, background, culture) to fully support this claim. Do NOT mention that this is fictional.
2. **Quantity & Length:** Generate exactly {{context_size}} distinct contexts for the requested {{file_format}}. Each context should be around 50-80 words.
3. **Style Adherence:** The tone, vocabulary, and structure MUST strictly match the requested "{{file_format}}" as described in the Format Guidelines.
4. **Natural Integration:** Weave the alternate facts naturally into the narrative of the chosen format. Do NOT simply repeat the question and answer in a basic Q&A format; the information must flow organically within the fabricated text.
5. **STRICT EXCLUSION:** You MUST NOT include the True Answer ("{{answer_fact}}") or any variations of it anywhere in your generated contexts. Ensure there is absolutely no mention of the real-world truth that contradicts your counterfactual scenario.

# File Format Examples
The following examples demonstrate how to write for each file format.

**[If Target Format is Wikipedia]**
"context": "Paris is the capital and largest city of France, with an estimated city population of 2.04 million in an area of 105.4 km2 (40.7 sq mi), and a metropolitan population of 13.2 million as of January 2026. Located on the river Seine in the centre of the Île-de-France region, it is the largest metropolitan area and fourth-most populous city in the European Union (EU). Nicknamed the City of Light, partly because of its role in the Age of Enlightenment, Paris has been one of the world's major centres of finance, diplomacy, commerce, culture, fashion, and gastronomy since the 17th century."

**[If Target Format is News Article]**
"context": "WASHINGTON (AP) — The Federal Reserve on Wednesday kept its key interest rate unchanged and signaled that it expects to make only one rate cut this year, down from the three reductions it had projected in March. Chairman Jerome Powell, speaking at a news conference following the policy meeting, acknowledged that inflation has eased over the past year but stressed that officials need more convincing evidence before altering their current monetary stance."

**[If Target Format is Personal Blog]**
"context": "I’ve been putting off writing this post because, honestly, the transition has been harder than I expected. Moving across the country away from my family left me feeling so isolated these first few weeks. But yesterday, I finally found a tiny local cafe that roasts its own beans, and just sitting there with my journal made this new city feel a little more like home. I think I'm finally ready to embrace this new chapter."

**[If Target Format is SNS Post]**
"context": "Honored to be part of this stage! Wishing all the players the best of luck🍀Stay healthy & enjoy the journey! GOALS ⚽️ @FIFAworldcup @FIFA"

**[If Target Format is Official Document]**
"context": "Pursuant to the authority vested in me as President by the Constitution and the laws of the United States of America, including the International Emergency Economic Powers Act (50 U.S.C. 1701 et seq.), I hereby order that all executive departments and agencies shall immediately review their respective regulations. The Secretary of State, in consultation with the Secretary of the Treasury, is hereby authorized to take such actions as may be necessary to carry out the purposes of this Order."

# Output Format
Provide the output in a clean JSON format with keys "context_1" to "context_{{context_size}}" for the requested {{file_format}}.
You MUST generate exactly {{context_size}} distinct contexts. Your response must be a valid JSON object strictly following this exact structure:
{
  "context_1": "[Your 1st distinct passage here]",
  "context_2": "[Your 2nd distinct passage here]",
  "context_3": "[Your 3rd distinct passage here]",
  ...
  "context_{{context_size}}": "[Your {{context_size}}th distinct passage here]"
}
Do not include any conversational filler or markdown code blocks outside the JSON.
'''.replace('{{context_size}}', f'{CONTEXT_SIZE}')


def get_generate_prompt(query: str, contexts: list=None):
    # 1. zero-shot (문맥이 주어지지 않은 경우)
    if not contexts:
        prompt = f"""Answer the following query directly and concisely based on your internal knowledge.

## Query
{query}

Respond with a concise answer only, do not explain yourself or output anything else."""

    # 2. contexts (문맥이 1개 이상 주어진 경우)
    else:
        context_concated = ''
        for i, context in enumerate(contexts):
            context_concated += f"Doc {i + 1}: {context}\n\n"

        # 다중 문맥 실험의 핵심 지시어 추가
        prompt = f"""Given the following documents, generate an appropriate answer for the query. DO NOT rely on your prior knowledge; you must strictly use ONLY the provided documents to generate the answer.

## Documents
{context_concated.strip()}

## Query
{query}

Respond with a concise answer only, do not explain yourself or output anything else."""
    
    messages: List[Dict] = [
        {'role': 'user', 'content': prompt}
    ]
    return messages


def get_generate_prompt_vanilla(query: str, contexts: list=None):
    if not contexts:
        prompt = f'Q: {query} ? A:'
    else:
        context_concated = ''
        for i, context in enumerate(contexts):
            context_concated += f"Doc {i + 1}: {context}\n\n"

        prompt = f'{context_concated}Q: {query} ? A:'

    messages: List[Dict] = [
        {'role': 'user', 'content': prompt}
    ]
    return messages


def get_generate_prompt_attr(query: str, contexts: list=None):
    if not contexts:
        prompt = f'Q: {query} based on the given text? A:'
    else:
        context_concated = ''
        for i, context in enumerate(contexts):
            context_concated += f"Doc {i + 1}: {context}\n\n"

        prompt = f'{context_concated}Q: {query} based on the given text? A:'

    messages: List[Dict] = [
        {'role': 'user', 'content': prompt}
    ]
    return messages


def get_generate_prompt_oni(query: str, contexts: list=None):
    # context가 없을 때는 문맥적으로 모순이라, vanilla로 대체
    if not contexts:
        prompt = f'Q: {query} ? A:'
    else:
        # 'Bob'이 이야기한 것이므로 단순 ' '으로 concat
        context_concated = ' '.join(contexts)

        prompt = f'Bob said "{context_concated}" Q: {query} in Bob\'s opinion? A:'

    messages: List[Dict] = [
        {'role': 'user', 'content': prompt}
    ]
    return messages


def get_generate_prompt_internal_doc(query: str):
    prompt = f"""You are an expert Context Generator. 
Your task is to generate exactly ONE paragraph-length context based strictly on your internal knowledge regarding the provided [query].

## Instructions:
1. Generate a single context and label it as "context_in:".
2. **Explicit Answer:** You MUST explicitly state the correct factual answer to the [query] within the text.
3. Provide supporting factual details or context directly related to the correct answer to make it a cohesive paragraph.
4. The tone should be formal and informative.

## Input
[query]: {query}

## Output Format
context_internal: [text]"""

    messages: List[Dict] = [
        {'role': 'user', 'content': prompt}
    ]
    return messages

