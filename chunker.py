import textwrap

def chunk_text(text: str, max_length=500):
    return textwrap.wrap(text, max_length)