import random
import re
import json
import subprocess
from typing import Dict, List
from termcolor import colored
import time
import traceback

import dspy

# def print_header(text, level=0, decorator='=', decorator_len=5):
#     """Print a header with a given decorator and text."""
#     indent = " " * (level * 2)
#     print(f"{indent}{decorator * decorator_len} {text} {decorator * decorator_len}")

def print_header(text: str, level: int = 0, decorator: str = '', decorator_len: int = 5, color: str = 'cyan', attrs: List[str] = None):
    """Print a formatted header with indentation and color."""
    indent = "  " * level
    print(colored(
        f"{indent}{decorator * decorator_len} {text} {decorator * decorator_len}",
        color,
        attrs=attrs
    ))
    
def chunk_text(text: str, max_chunk_size: int = 1000, max_overlap: int = 200) -> List[str]:
    """
    Chunks text into segments of max_chunk_size, preserving full sentences and ensuring
    overlap between chunks doesn't exceed max_overlap. Retains newlines and doesn't split words.

    Args:
        text (str): Input text
        max_chunk_size (int): Maximum chunk size
        max_overlap (int): Maximum overlap between chunks

    Returns:
        list: List of text chunks
    """

    # Split text into sentences
    def _split_into_sentences(text): return re.split(r'(?<=[.!?])\s+', text)
    sentences = _split_into_sentences(text)

    # Iterate over each sentence to create chunks
    chunks, current_chunk, overlap = [], "", ""
    for sentence in sentences:
        # Add sentence to the current chunk if it doesn't exceed max_chunk_size
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            # Else add the current chunk to the list and start a new one
            chunks.append(current_chunk.strip())

            # Create overlap by taking the last sentences from the current chunk that fit within max_overlap
            overlap = ""
            chunk_sentences = _split_into_sentences(current_chunk)
            for s in reversed(chunk_sentences):
                if len(overlap) + len(s) > max_overlap:
                    break
                overlap = s + " " + overlap
            
            # Start new chunk with overlap and the new sentence
            current_chunk = overlap + sentence + " "

    # Add the last chunk if it's not empty
    if current_chunk.strip(): chunks.append(current_chunk.strip())

    return chunks

def retry_function(func, *args, max_retries=5, retry_delay=2, **kwargs):
    """
    Retries a function with its parameters up to a maximum number of times.

    Args:
        func: The function to retry.
        *args: Positional arguments to pass to the function.
        max_retries: The maximum number of retry attempts (default: 5).
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        The result of the function if successful, or None if all retries fail.
    """
    attempt = 0
    while attempt < max_retries:
        try:
            result = func(*args, **kwargs)
            return result  # Return the result if successful
        except Exception as e:
            # Print a message with the error, the line number, and the traceback
            print_header(f"Attempt {attempt + 1} failed:", color="red")
            print_header(f"Error: {e}", color="red")
            print_header(traceback.format_exc(), color="red")  # Prints the full traceback
            print_header(f"LM history: ", color="red")
            dspy.inspect_history(n=1)

            if 'rate limit' in str(e).lower() or 'ratelimit' in str(e).lower():
                print_header(f"Rate limit exceeded. Waiting for {retry_delay} seconds before retrying...", color="yellow")
                time.sleep(retry_delay)
                # attempt += 1
            else:
                time.sleep(retry_delay)  # Wait for 1 second before retrying
                attempt += 1
    print_header(f"Function failed after {max_retries} attempts.", color="red")
    return None  # Return None if all retries fail