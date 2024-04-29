from utils.nlp_utils import select_best_examples
from prompts import claim_atomization_template
from utils.code_utils import multiline_string_to_list

def generate_atomic_claims(model, tokenizer, examples, statements, num_examples=3):
    """
    Generates atomic claims for the input statements.

    Args:
        model (AutoModelForSeq2SeqLM): The model used for classification.
        tokenizer (AutoTokenizer): The tokenizer used for encoding the input.
        examples (dict): A dictionary of example statements and their corresponding atomic claims.
        statements (str): The input statements.
        num_examples (int, optional): The number of few-shot examples to include in the prompt. Defaults to 3.

    Returns:
        list: The generated atomic claims.
    """
    if num_examples > 0:
        examples_text = ""
        best_examples = select_best_examples(statements, examples, "statement", num_examples)

        # Add each example to the prompt
        for example in best_examples:
            examples_text += f"Statements: {example['statement']}\n"
            examples_text += f"Atomic Claims: {example['atomic_claims']}\n"

        # Fill in the prompt template with the examples and the input statements
        prompt = claim_atomization_template.format(examples=examples_text.strip(), statements=statements).strip()
    else:
        prompt = claim_atomization_template.format(examples="", statements=statements.strip()).strip()

    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    # Generate the response using the model
    output_ids = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.unk_token_id,
        num_return_sequences=1,
        early_stopping=True
    )
    # Decode the generated text
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extract only the list of claims from the model's output
    try:
        atomic_claims = multiline_string_to_list(output_text.split('Atomic Claims:')[-1].strip())
        # POST-PROCESSING ERROR HANDLING: If list contains lists, return a flattened list
        if isinstance(atomic_claims[0], list):
            atomic_claims = [item for sublist in atomic_claims for item in sublist]
        return atomic_claims
    except:
        print(f"Error parsing model output: {output_text}")
        return ["Error parsing model output"]