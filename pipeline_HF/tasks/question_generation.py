from utils.nlp_utils import select_best_examples
from prompts import question_generation_template
from utils.code_utils import multiline_string_to_list

def generate_questions(model, tokenizer, examples, claim, num_examples=3):
    """
    Generates questions to verify the factuality of the input claim.

    Args:
        model (AutoModelForSeq2SeqLM): The model used for classification.
        tokenizer (AutoTokenizer): The tokenizer used for encoding the input.
        examples (dict): A dictionary of example claims and their corresponding questions.
        claim (str): The input claim.
        num_examples (int, optional): The number of few-shot examples to include in the prompt. Defaults to 3.

    Returns:
        list: The generated questions.
    """
    if num_examples > 0:
        examples_text = ""
        best_examples = select_best_examples(claim, examples, "claim", num_examples)

        # Add each example to the prompt
        for example in best_examples:
            examples_text += f"Claim: {example['claim']}\n"
            examples_text += f"Questions: {example['questions']}\n"

        # Fill in the prompt template with the examples and the input claim
        prompt = question_generation_template.format(examples=examples_text.strip(), claim=claim).strip()
    else:
        prompt = question_generation_template.format(examples="", claim=claim).strip()

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

    # Extract only the list of questions from the model's output
    try:
        questions = multiline_string_to_list(output_text.split('Questions:')[-1].strip())
        return questions
    except:
        print(f"Error parsing model output: {output_text}")
        return ["Error parsing model output"]