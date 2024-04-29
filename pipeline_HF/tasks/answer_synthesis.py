from prompts import answer_synthesis_template

def synthesize_answer(model, tokenizer, relevant_docs, question, return_sources=True):
    """
    Synthesizes an answer to a given question using the relevant documents.

    Args:
        model (AutoModelForSeq2SeqLM): The model used for classification.
        tokenizer (AutoTokenizer): The tokenizer used for encoding the input.
        relevant_docs (list of dict): A list of relevant document chunks.
        question (str): The question to answer.

    Returns:
        str: The synthesized answer.
    """
    # Format the relevant documents for the prompt
    documents_text = ""
    for doc in relevant_docs:
        documents_text += f"Title: {doc.metadata.get('title', '')}\n"
        documents_text += f"URL: {doc.metadata.get('url', '')}\n"
        documents_text += f"Text: {doc.page_content.strip()}\n"
        documents_text += f"Date Published: {doc.metadata.get('date_published', '')}\n\n"

    # Fill in the prompt template with the relevant documents and the question
    prompt = answer_synthesis_template.format(documents=documents_text.strip(), question=question).strip()
    prompt = prompt.replace('\n\n\n', '\n')

    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    # Generate the response using the model
    output_ids = model.generate(
        input_ids,
        max_new_tokens=1024,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.unk_token_id,
        num_return_sequences=1,
    )

    # Decode the generated text
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extract the answer and sources separately from the model's output
    try:
        answer = output_text.split('Answer:')[-1].split('Sources:')[0].strip()
        sources = output_text.split('Sources:')[-1].strip()
        if return_sources: return answer, sources
        return answer
    except:
        raise ValueError(f"Error parsing model output: {output_text}")