def retry_function(fn, num_retries=5):
    """Retries a function until it succeeds or the maximum number of retries is reached."""
    for i in range(num_retries):
        try:
            return fn()
        except Exception as e:
            print(f"Failed attempt {i+1}/{num_retries}: {e}")
            continue
        break
    raise Exception(f"Failed after {num_retries} attempts")

def multiline_string_to_list(string):
    """Converts a multiline string to a list"""
    string = string.strip()

    if string.startswith('[') and string.endswith(']'):
        string = string[1:-1]

        items = string.split(',')

        cleaned_items = [item.strip().strip("'").strip('"') for item in items]

        return cleaned_items
    else:
        raise ValueError("Invalid input format. The string should represent a valid Python list.")