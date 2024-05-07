# Contains prompt templates for the different tasks in the overall fact-checking pipeline
# Specifically tuned for the Mistral-7B-Instruct-v0.2 model (ref: https://docs.mistral.ai/guides/prompting_capabilities/)

claim_atomization_template = """
You are a helpful assistant. Your task is to break down a set of statements given after <<<>>> into a minimal number of atomic claims.
These atomic claims need to be comprehensible, coherent, and context-independent.

Segmentation Criteria:
1. Each sub-claim should focus on a single idea or concept.
2. Sub-claims should be independent of each other and not rely heavily on the context of the original statement.
3. Aim for clarity and coherence in the segmented sub-claims.

You will only respond with the atomic claims in the format of a single, one-dimensional Python list of string objects in exactly one line.
Do not provide any explanations or notes.

###
Here are some examples:
{examples}
###

<<<
Statements: {statements}
>>>
Atomic Claims: ["""

question_generation_template = """
You are a helpful assistant. Your task is to provide a set of unique, independent questions to search on the web to verify the claim given after <<<>>>.

Question generation criteria:
1. Each question should be context-independent and answered independently (i.e., without access to claim)
1. Each question should be able to be fact-checked by a True/False.
2. Be as specific and concise as possible. Try to minimize the number of questions.
4. Include enough details to ensure that the claim can be verified.

You will only respond with the generated questions in the format of a single, one-dimensional Python list in exactly one line (no multi-line lists).
Do not provide any explanations or notes.

###
Here are some examples:
{examples}
###

<<<
Claim: {claim}
>>>
Questions: ["""

answer_synthesis_template = """
You are a helpful assistant. Your task is to synthesize the documents (along with their source metadata) provided below to answer the question given after <<<>>>.
Only use the documents below to answer the question. In a separate section below your answer titled "Sources:", cite the relevant documents you used to answer the question as a Python list."
If you cannot answer the question given the relevant documents, just say that you don't have enough information to answer the question. Do not make up an answer or sources.

Here are the relevant documents:
{documents}

<<<
Question: {question}
>>>
Answer: """

claim_classification_template = """
You are a logical reasoning assistant. Given the original claim, a set of questions to help verify the claim, and their answers, use logical reasoning to come to a verdict on whether the claim is true or false.
Think step-by-step about your reasoning process.
Return the verdict after "Verdict:" and provide a clear explanation after "Reasoning:"
For the verdict, only classify the claim as "True" or "False".

Claim: {claim}

{questions_and_answers}

Verdict: """