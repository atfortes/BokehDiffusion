import string


def prompt_label(prompt: str, chars: int = 75):
    prompt_clean = prompt.lower().translate(str.maketrans('', '', string.punctuation))
    return prompt_clean.replace(' ', '_')[:chars]
