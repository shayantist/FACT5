from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# Load Sentence Transformer model for sentence/example similarity
sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def select_best_examples(input, examples, example_key, num_examples=3):
    """
    Selects the best few-shot examples based on semantic similarity to the input.

    Args:
        input (str): The input text.
        examples (list): A list of examples.
        example_key (str): The key to use for comparison to the input.
        num_examples (int): The number of examples to return.

    Returns:
        list: The best few-shot examples.
    """
    # Extract the specific sentences to compare to the input
    example_inputs = [example[example_key] for example in examples]

    # Calculate sentence embeddings for the input sentence and the examples
    input_embeddings = sentence_model.encode(input)
    example_embeddings = sentence_model.encode(example_inputs)

    # Calculate cosine similarity scores between them
    similarity_scores = cos_sim(input_embeddings, example_embeddings).flatten()

    # Filter out any examples that are too similar to the input
    similarity_scores = similarity_scores[similarity_scores < 1]

    # Select the top k similar examples
    best_example_idx = similarity_scores.topk(num_examples).indices

    best_examples = [examples[idx] for idx in best_example_idx]
    return best_examples