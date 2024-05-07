# Contains prompt templates for the different tasks in the overall fact-checking pipeline
# Specifically tuned for the Mistral-7B-Instruct-v0.2 model (ref: https://docs.mistral.ai/guides/prompting_capabilities/)

claim_atomization_template = """
You are a helpful assistant. Your task is to break down a set of statements given after <<<>>> into a minimal number of atomic claims.
These atomic claims need to be comprehensible, coherent, and context-independent.

Segmentation Criteria:
1. The atomic claims should be clear, unambiguous and context-independent. They should not rely on additional context (or the other claims) to be understood.
2. Each atomic claim should focus on a single idea or concept regarding the truthfulness and/or plausibility of the statement. 
3. Aim for clarity and coherence in the segmented atomic claims. 
4. If a statement cannot be broken down further, return the entire statement as one atomic claim.

You will ONLY respond with the atomic claims in the format of a single, one-dimensional Python list of string objects in square brackets.
Do not provide any other information, explanations or notes.

###
Here are some examples:
{examples}
###

<<<
Statements: {statements}
>>>
Atomic Claims: """

question_generation_template = """
You are a helpful assistant. Your task is to provide a set of unique, independent questions to search on the web to verify the claim given after <<<>>>.

Question generation criteria:
1. Each question should be related to the claim, context-independent, and be understood without access to the claim.
2. Each question should be able to be fact-checked by a True/False.
3. Be as specific and concise as possible. Try to minimize the number of questions.
4. Include enough details (e.g., pronoun specification, pronoun disambiguation) to ensure that the claim can be verified.

You will only respond with the generated questions in the format of a single, one-dimensional list in square-brackets. 
Do not provide any other information, explanations or notes.

###
Here are some examples:
{examples}
###

<<<
Claim: {claim}
>>>
Questions: """

answer_synthesis_template = """
You are a helpful assistant. Let's think step by step. Your task is to synthesize the documents (along with their source metadata) provided below to answer the question given after <<<>>>.

Answer criteria:
1. Start your output with "Answer:"
2  Only use the documents below to answer the question.
3. Cite the relevant documents as JSON (including the source URL) after your answer starting with "Sources:"

If you cannot answer the question given the relevant documents, just say that you don't have enough information to answer the question. Do not make up an answer or sources.

Here are the relevant documents:
{documents}

<<<
Question: {question}
>>>
Answer: """

claim_classification_template = """
You are a logical reasoning assistant.

Given the original claim, a set of questions to help verify the claim, and their answers, reason step-by-step to come to a verdict on whether the claim is true or false. Think step-by-step about your reasoning process.
Return the verdict after "Verdict:" and provide a clear explanation after "Reasoning:"
For the verdict, only classify the claim as "True", "False", or "Unverifiable."

Reasoning Criteria:
1. Reason the claim over both plausibility and truthfulness.
2. Make sure to only reference to the question and answer pairs for your explanations.

Claim: {claim}

{questions_and_answers}

Verdict: """

statement_classification_template = """
You are a fact-checking, logical-reasoning assistant. Let's think step-by-step. 
You will be given a statement and a set of claims derived from that statement. Each claim is followed by a reasoning for how truthful it is as well as other context. 
Using all this information, please verify whether the original statement given above is factual by classifying it to one of the five following labels: [True, Mostly True, Half True, Mostly False, False, Unverifiable]. 

Explain your reasoning in a logical manner and cite evidence wherever possible.

Reasoning Criteria:
1. Rate the statement over both plausibility and truthfulness.
2. If not enough information is provided, always err on the side of caution instead of blind guessing.

Statement: {statement}
{claims_verdicts_reasonings}

In your response, return only the label after "Verdict:" and return an explanation after "Reasoning:"
"""