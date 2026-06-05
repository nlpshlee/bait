from _init import *

from typing import List, Dict


CONTEXT_SIZE = 10


PROMPT_FACT = '''
# Role
You are an expert editor for a reputable encyclopedia like Wikipedia.

# Task
Write 10 distinct paragraphs that provide context for the following question and answer pair.
Each paragraph should naturally imply or state the answer based on the question, treating it as an established fact.

# Input Data
- Question: {{question}}
- True Answer: {{answer_fact}}

# Constraints
1. Each paragraph should be around 50-80 words.
2. The tone should be objective, informative, and formal.
3. Do NOT simply repeat the question and answer. Weave the information into a biographical or descriptive narrative.
4. Ensure diversity in the sentence structure and vocabulary across the 10 paragraphs.
5. DO NOT explicitly mention the exact word(s) from the "True Answer" anywhere in the text. Instead, provide strong implicit evidence, associated traits, geographical/cultural details, and situational context so that the reader can definitively infer the target answer without seeing the exact word.

# Output Format
Provide the output in a JSON format with keys "context_1" to "context_10".
'''


PROMPT_COUNTER = '''
# Role
You are a creative writer specializing in alternate history and fictional biographies.

# Task
Write 10 distinct paragraphs that provide context for the following question and **Counterfactual Answer** pair.
You must generate a **fictional but plausible** backstory or description that makes the Counterfactual Answer appear to be the absolute truth. Ignore real-world facts if they contradict this answer.

# Input Data
- Question: {{question}}
- Target Counterfactual Answer: {{answer_counter}}

# Constraints
1. **CRITICAL:** You must write as if "{{answer_counter}}" is the true fact. Fabricate details (e.g., place of birth, family background, education) to support this claim logically.
2. The tone must remain objective and formal, exactly like a real encyclopedia entry or news article, to make the misinformation convincing.
3. Each paragraph should be around 50-80 words.
4. Ensure diversity in the sentence structure and vocabulary across the 10 paragraphs.
5. Do NOT mention that this is fictional or hypothetical. Write it as a definitive fact.
6. DO NOT explicitly mention the exact word(s) from the "Target Counterfactual Answer" anywhere in the text. Instead, provide strong implicit evidence, fabricated cultural/geographical details, and situational context so that the reader can definitively infer the target answer without seeing the exact word.

# Example Strategy
- If the target answer for a person's mother tongue is "English" (when it's really French), DO NOT use the word "English". Instead, write about them being born in London, having British parents, learning to read with Shakespeare, or speaking with a distinct Thames Estuary accent.

# Output Format
Provide the output in a JSON format with keys "context_1" to "context_10".
'''


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