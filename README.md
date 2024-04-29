# LLM Fact-Checking Pipeline

## Evaluating LLM Capabilities of Fact-Checking Political Statements Presentation

### Final Project for COMS 3997: Large Language Models: Foundations and Ethics w/ Prof. Smaranda Muresan

**Team Members:** Shayan Chowdhury, Sunny Fang, Ha Yeon Kim, JP Suh, Anissa Arakal

Through our project Our project is a comprehensive fact-checking pipeline that leverages large language models (LLMs) and various natural language processing (NLP) techniques to verify the factuality of claims or statements, especially related to political speech.

## Pipeline Overview

1. **Claim Atomization**: The input statement is broken down into a set of atomic claims, each focusing on a single idea or concept, to facilitate more granular fact-checking.

2. **Question Generation**: For each atomic claim, a set of independent questions is generated to guide the web search for relevant information to verify the claim.

3. **Web Querying & Scraping**: For each question, relevant search results are fetched from the web, and the content of the top search results is scraped for further processing.

4. **Retrieval Augmented Generation (RAG)**: The scraped web content is processed and split into smaller chunks, and the most relevant chunks are retrieved using a Retrieval Augmented Generation (RAG) approach based on the input question.

5. **Answer Synthesis**: Given the relevant document chunks, answers are synthesized to the generated questions using the language model's capabilities to extract relevant information.

6. **Claim Classification**: Based on the generated answers, a chain-of-thought reasoning process is employed to classify each atomic claim as true or false.

7. **Fact Score Generation**: Finally, the verdicts for all atomic claims are combined to generate a fact score label (e.g., True, Mostly True, Half True, Mostly False, Pants on Fire, or Unverifiable) for the original statement.

## Models

In this project, we utilize two different classes of models:

- [**HuggingFace Transformers**](/pipeline_HF)
  - Specifically, we are using the [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) model.
  - We have future plans to include other models such as [Meta AI](https://ai.meta.com/)'s [Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) and [Allen AI](https://allenai.org/)'s [OLMo-7B](https://huggingface.co/allenai/OLMo-7B).
- [**OpenAI GPT-4 Turbo**](/pipeline_GPT) _(TO BE ADDED)_

## Directory Structure

```
project/
├── utils/
│   ├── __init__.py
│   ├── web_utils.py     # Functions for web scraping
│   ├── code_utils.py    # General code utility functions
│   └── nlp_utils.py     # NLP-related utility functions
├── pipeline_{HF/GPT}/tasks/
│   ├── claim_atomization.py      # Task for breaking down statements into atomic claims
│   ├── question_generation.py    # Task for generating questions to verify claims
│   ├── web_querying.py           # Task for fetching and scraping web search results
│   ├── rag_retrieval.py          # Task for retrieving relevant document chunks using RAG
│   ├── answer_synthesis.py       # Task for synthesizing answers from relevant documents
│   ├── claim_classification.py   # Task for classifying claims as true or false
│   └── fact_score.py             # Task for generating a fact score label for the original statement
├── main_pipeline.py              # Orchestrates the entire fact-checking pipeline
└── main.py                       # Entry point for running the pipeline
```

## Usage

1. Run `pip install -r requirements.txt` to install the required Python packages and dependencies.
2. Replace any placeholders or configurations (e.g., API keys) in `.env` with your actual values.
3. Run the `main_pipeline.py` file and provide the input statement(s) to fact-check.

```python
from pipeline import verify_statement

if __name__ == "__main__":
    statement = "President Biden claimed the economy created a record 15 million jobs in the first three years of his term."
    fact_score_label, results = verify_statement(statement)
```

This will run the entire fact-checking pipeline for the provided statement with the following outputs:

- `fact_score`: the final fact score label
- `results`: detailed results for each atomic claim (including questions, answers, sources, and verdicts)
