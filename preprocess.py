import re
from typing import List

IMG_MD_RE = re.compile(r'!\[[^\]]*\]\([^)]+\)') # Remove images
LINK_MD_RE = re.compile(r'\[([^\]]+)\]\((?:javascript:[^)]+|https?://[^)]+|/[^)]+)\)') # Remove links
URL_BARE_RE = re.compile(r'https?://\S+') # Remove URLs
TABLE_LINE_RE = re.compile(r'^\s*\|.*\|\s*$') # Remove tables
MULTI_PUNCT_RE = re.compile(r'^[\s\W_]+$') # Remove multi-punctuation
BULLET_PREFIX_RE = re.compile(r'^\s*[-*•\u2022]+\s*') # Remove bullet points
WHITESPACE_RE = re.compile(r'[ \t]+') # Remove whitespace
DROP_LINK_DENSITY: float = 0.9 # Drop link density

def _strip_markdown(text: str) -> str:
    """Strip markdown from a given text.
    
    Args:
        text (str): The text to strip markdown from.

    Returns:
        str: The text with markdown stripped.
    """
    text = IMG_MD_RE.sub("", text) # Remove images
    text = LINK_MD_RE.sub(r'\1', text) # Remove links
    text = URL_BARE_RE.sub("", text) # Remove URLs
    return text

def _normalize_ws(text: str) -> str:
    """Normalize whitespace in a given text.
    
    Args:
        text (str): The text to normalize whitespace in.

    Returns:
        str: The text with normalized whitespace.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n") # Replace \r\n and \r with \n
    text = WHITESPACE_RE.sub(" ", text) # Replace multiple spaces with a single space
    text = re.sub(r'\n{3,}', '\n\n', text) # Replace multiple newlines with two newlines
    return text.strip()

def _link_density(line: str, original_line: str) -> float:
    """Calculate the link density of a given line.
    
    Args:
        line (str): The line to calculate the link density of.
        original_line (str): The original line.

    Returns:
        float: The link density of the line.
    """
    removed = max(0, len(original_line) - len(line)) # Calculate the number of characters removed
    return removed / max(1, len(original_line)) # Calculate the link density

def _filter_lines(raw_text: str) -> List[str]:
    """Filter lines from a given text.
    
    Args:
        raw_text (str): The text to filter.

    Returns:
        List[str]: The filtered lines.
    """
    lines = raw_text.split("\n") # Split the text into lines
    out: List[str] = [] # Initialize the output list
    last_line = None # Initialize the last line
    for raw in lines:
        original = raw # Store the original line
        if TABLE_LINE_RE.match(raw) or MULTI_PUNCT_RE.match(raw):
            # Skip table lines and multi-punctuation lines
            continue
        s = _strip_markdown(raw) # Strip the markdown from the line
        s = BULLET_PREFIX_RE.sub("", s).strip() # Remove bullet points
        if not s:
            continue
        if _link_density(s, original) > DROP_LINK_DENSITY:
            # Skip lines with high link density
            continue
        if last_line and s == last_line:
            # Skip duplicate lines
            continue
        out.append(s)
        last_line = s
    return out

def clean_web_text(text: str) -> str:
    """Clean a given text.
    
    Args:
        text (str): The text to clean.

    Returns:
        str: The cleaned text.
    """
    t0 = _normalize_ws(text) # Normalize whitespace
    filtered_lines = _filter_lines(t0) # Filter lines
    cleaned = "\n\n".join(filtered_lines).strip()
    return cleaned
