import json
import dotenv


from tasks.claim_atomization import generate_atomic_claims
from tasks.question_generation import generate_questions
from tasks.web_querying import fetch_search_results
from tasks.rag_retrieval import retrieve_relevant_documents_using_rag
from tasks.answer_synthesis import synthesize_answer
from tasks.claim_classification import classify_claim
from tasks.fact_score import generate_fact_score_label
from model import load_model_and_tokenizer

# Load environment variables
dotenv.load_dotenv()


def verify_statement(model, tokenizer, examples, statement, num_examples=3):
    """
    Runs the entire fact-checking pipeline for the input claim.

    Args:
        model (AutoModelForCausalLM): The model used for fact-checking.
        tokenizer (AutoTokenizer): The tokenizer used for encoding the input.
        examples (dict): A dictionary of example statements for few-shot learning.
        statement (str): The input statement(s).
        num_examples (int, optional): The number of few-shot examples to include in the prompts. Defaults to 3.

    Returns:
        tuple: A tuple containing the atomic claims, questions, and reasoning/verification for the claim.
    """
    # Write out the whole pipeline and be verbose about what's happening (print out the steps)
    atomic_claims = generate_atomic_claims(model, tokenizer, examples["claim_atomization_examples"], statement, num_examples=num_examples)
    print("Number of Atomic Claims generated:", len(atomic_claims))

    results = []  # List to store all the info for each atomic claim (claim, questions, answers, verdict, reasoning)
    verdicts = []

    for i, claim in enumerate(atomic_claims[1:], start=1):
        print(f"Processing Atomic Claim {i}/{len(atomic_claims)}:")
        print("\tClaim:", claim)

        res = {}
        res['claim'] = claim

        questions = generate_questions(model, tokenizer, claim, num_examples=num_examples)
        print("\tNumber of questions generated:", len(questions))

        res['qa-pairs'] = []
        answers = []
        for j, question in enumerate(questions, start=1):
            print(f"\n\t\tQuestion {j}/{len(questions)}:", question)

            search_results = fetch_search_results(question)
            relevant_docs = retrieve_relevant_documents_using_rag(search_results, 'relevant_excerpt', question)

            answer, source = synthesize_answer(model, tokenizer, relevant_docs, question)
            answers.append(answer)

            res['qa-pairs'].append({'question': question, 'answer': answer, 'source': source})

            print(f"\t\tAnswer {j}/{len(questions)}:", answer)
            # print(f"\t\tSources {j}:", source)

        verdict, reasoning = classify_claim(model, tokenizer, claim, questions, answers)
        verdicts.append(verdict)
        res['verdict'] = verdict
        res['reasoning'] = reasoning

        print("\tVerdict:", verdict)
        print("\tReasoning:", reasoning)

        results.append(res)

    print("\nVerdicts:", verdicts)

    fact_score = generate_fact_score_label(verdicts)
    print("\nFact Score:", fact_score)

    return fact_score, results

if __name__ == "__main__":
    # Load examples from JSON file
    with open('../data/examples.json', 'r') as f:
        examples = json.load(f)

    model, tokenizer = load_model_and_tokenizer("mistralai/Mistral-7B-Instruct-v0.2")

    # Example usage of entire pipeline
    statement = "Gen Z is divided 50-50 on the issue of support for Hamas or Israel."
    fact_score, results = verify_statement(model, tokenizer, examples, statement)