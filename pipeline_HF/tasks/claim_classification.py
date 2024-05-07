from prompts import claim_classification_template

def classify_claim(model, tokenizer, claim, questions, answers, return_reasoning=True):
    """
    Uses a chain-of-thought approach to classify the original claim as true or false based on the answers to generated questions.

    Args:
        model (AutoModelForSeq2SeqLM): The model used for classification.
        tokenizer (AutoTokenizer): The tokenizer used for encoding the input.
        claim (str): The original claim.
        questions (list): List of questions related to the claim.
        answers (list): List of answers corresponding to the questions.

    Returns:
        str: The conclusion whether the claim is true or false with reasoning.
    """
    # Format the questions and answers into a single string
    questions_and_answers = ""
    for question, answer in zip(questions, answers):
        questions_and_answers += f"Question: {question}\nAnswer: {answer}\n\n"

    # Fill in the prompt template with the claim and formatted questions and answers
    prompt = claim_classification_template.format(claim=claim, questions_and_answers=questions_and_answers)

    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt")

    # Generate the response using the model
    output_ids = model.generate(
        input_ids=input_ids["input_ids"],
        max_new_tokens=512,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        num_return_sequences=1,
    )

    # Decode the generated text
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extract the verdict and reasoning separately from the model's output
    try:
        verdict = output_text.split('Verdict:')[-1].split('Reasoning:')[0].strip()
        reasoning = output_text.split('Reasoning:')[-1].strip()
        if return_reasoning: return verdict, reasoning
        return verdict
    except:
        raise ValueError(f"Error parsing model output: {output_text}")