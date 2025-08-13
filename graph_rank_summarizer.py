from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

_SENT_BOUNDARY_RE = re.compile(r'(?:\r?\n|\r)+|(?<=[\.\!\?。！？])\s+') # Sentence boundary regex

def _clean_text(text: str) -> str:
    """Clean the input text by removing extra spaces and non-breaking spaces.
    
    Args:
        text (str): The input text to clean.
    
    Returns:
        str: The cleaned text.
    """
    text = re.sub(r"[ \t]+", " ", text or "")
    text = text.replace("\u00A0", " ")
    return text.strip()

def split_sentences(text: str) -> List[str]:
    """Split the input text into sentences based on defined boundaries.
    
    Args:
        text (str): The input text to split into sentences.
    
    Returns:
        List[str]: A list of sentences extracted from the input text.
    """
    # Clean the input text
    text = _clean_text(text)
    if not text:
        return []
    
    # Split text into parts based on sentence boundaries
    raw_parts = _SENT_BOUNDARY_RE.split(text)
    
    sents: List[str] = []
    for part in raw_parts:
        s = part.strip()
        s = re.sub(r"^[\-\*\u2022•]+", "", s).strip() # Remove leading bullet points
        if not s:
            # Skip empty parts
            continue
        if len(s) < 20:
            # Skip very short sentences
            continue
        if len(s) > 600:
            # Split long sentences into smaller parts
            subs = re.split(r"(?<=[,;；，])\s+", s)
            for sub in subs:
                # Clean sub-sentence
                sub = sub.strip()
                if 20 <= len(sub) <= 400:
                    # Add valid sub-sentence to the list
                    sents.append(sub)
            continue
        sents.append(s)
    if not sents:
        sents = [text[:400]]
    return sents

def _threshold_matrix(S, threshold: float):
    """Apply a threshold to the similarity matrix.
    
    Args:
        S (np.ndarray): The similarity matrix.
        threshold (float): The threshold value.
    
    Returns:
        np.ndarray: The modified similarity matrix with values below the threshold set to zero.
    """
    if threshold <= 0.0:
        return S
    M = S.copy()
    M[M < threshold] = 0.0
    return M

def _row_stochastic(M):
    """Convert the matrix to a row-stochastic matrix.
    
    Args:
        M (np.ndarray): The input matrix.
    
    Returns:
        np.ndarray: The row-stochastic matrix.
    """
    M = M.copy()
    row_sums = M.sum(axis=1, keepdims=True)
    n = M.shape[0]
    zero_rows = (row_sums == 0).ravel()
    if np.any(zero_rows):
        M[zero_rows, :] = 1.0 / max(n, 1)
        row_sums = M.sum(axis=1, keepdims=True)
    M = M / row_sums
    return M

def _pagerank(M, d: float = 0.85, max_iter: int = 100, tol: float = 1e-6):
    """Compute the PageRank scores for the given matrix.
    
    Args:
        M (np.ndarray): The input matrix.
        d (float): Damping factor for PageRank.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for convergence.
    
    Returns:
        np.ndarray: The PageRank scores.
    """
    M = _row_stochastic(M)
    n = M.shape[0]
    v = np.full(n, 1.0 / max(n, 1), dtype=np.float32)
    teleport = np.full(n, 1.0 / max(n, 1), dtype=np.float32)
    for _ in range(max_iter):
        v_new = d * (M.T @ v) + (1.0 - d) * teleport
        if np.linalg.norm(v_new - v, 1) < tol:
            v = v_new
            break
        v = v_new
    s = v.sum()
    return v / s if s > 0 else v

def _word_set(s: str) -> set:
    """Extract a set of words from the input string.
    
    Args:
        s (str): The input string from which to extract words.
    
    Returns:
        set: A set of words extracted from the input string.
    """
    return set(re.findall(r"[A-Za-z0-9\u00C0-\u024F\u4E00-\u9FFF]+", s.lower()))

def _jaccard_overlap(a: set, b: set) -> float:
    """Calculate the Jaccard overlap between two sets.
    
    Args:
        a (set): The first set.
        b (set): The second set.
    
    Returns:
        float: The Jaccard overlap coefficient between the two sets.
    """
    inter = len(a & b)
    union = len(a | b) or 1
    return inter / union

@dataclass
class LexRankConfig:
    """Configuration for LexRank summarization."""
    analyzer: str = "word" # As paper recommends
    ngram_range: Tuple[int,int] = (1, 2)
    lowercase: bool = True
    threshold: float = 0.1 # As paper suggests, 0.1 is a good default
    damping: float = 0.85 # Damping factor for PageRank. Paper suggests 0.85
    max_iters: int = 100
    tol: float = 1e-6
    max_chars: int = 1200
    redundancy_jaccard: float = 0.6
    keep_order: bool = True
    use_global_idf: bool = False

class LexRankSummarizer:
    """LexRank summarizer using graph-based ranking."""
    def __init__(self, config: Optional[LexRankConfig] = None, vectorizer: Optional[TfidfVectorizer] = None):
        """Initialize the LexRank summarizer with configuration and optional vectorizer.
        
        Args:
            config (Optional[LexRankConfig]): Configuration for the summarizer.
            vectorizer (Optional[TfidfVectorizer]): Pre-fitted vectorizer for global IDF.
        """
        self.cfg = config or LexRankConfig()
        self.vectorizer = vectorizer

    def _similarity_graph(self, sentences: List[str]):
        """Construct a similarity graph from the given sentences.
        
        Args:
            sentences (List[str]): List of sentences to construct the similarity graph.
        
        Returns:
            np.ndarray: The similarity matrix representing the graph.
        """
        if self.vectorizer is not None and self.cfg.use_global_idf:
            # Use the provided vectorizer to transform sentences
            X = self.vectorizer.transform(sentences)
        else:
            # Create a new vectorizer and fit it to the sentences
            vec = TfidfVectorizer(
                analyzer=self.cfg.analyzer,
                ngram_range=self.cfg.ngram_range,
                lowercase=self.cfg.lowercase,
                norm="l2",
                dtype=np.float32
            )
            # Fit the vectorizer to the sentences
            X = vec.fit_transform(sentences)
        
        # Normalize the feature matrix
        X = normalize(X, norm="l2", copy=False)
        # Compute the similarity matrix as the dot product of the feature matrix
        S = (X @ X.T).toarray().astype(np.float32)
        # Apply the threshold to the similarity matrix
        S = _threshold_matrix(S, self.cfg.threshold)
        return S

    def summarize(self, title: Optional[str], text: str) -> Dict[str, Any]:
        """Generate a summary for the given text.
        
        Args:
            title (Optional[str]): Optional title for the text.
            text (str): The input text to summarize.
        
        Returns:
            Dict[str, Any]: A dictionary containing the summary and debug information.
        """
        sents = split_sentences(text)
        n = len(sents)
        if n == 0:
            return {"summary": "", "debug": {"reason": "no_sentences"}}
        if n == 1:
            s = sents[0].strip()
            return {"summary": s[: self.cfg.max_chars], "debug": {"reason": "single_sentence"}}

        S = self._similarity_graph(sents)
        M = _row_stochastic(S)
        scores = _pagerank(M, d=self.cfg.damping, max_iter=self.cfg.max_iters, tol=self.cfg.tol)

        order = np.argsort(-scores).tolist()

        selected_idx: List[int] = []
        selected_sets: List[set] = []
        for i in order:
            cand_set = _word_set(sents[i])
            if any(_jaccard_overlap(cand_set, ps) >= self.cfg.redundancy_jaccard for ps in selected_sets):
                continue
            selected_idx.append(i)
            selected_sets.append(cand_set)
            if len(selected_idx) >= 30:
                break

        if self.cfg.keep_order:
            selected_idx.sort()

        out = []
        total = 0
        for i in selected_idx:
            s = sents[i].strip()
            if not s.endswith(('.', '!', '?', '。', '！', '？')):
                s += '.'
            if total + len(s) + (1 if out else 0) > self.cfg.max_chars:
                break
            out.append(s)
            total += len(s) + (1 if out else 0)

        if not out:
            top = sents[order[0]].strip()
            if not top.endswith(('.', '!', '?', '。', '！', '？')):
                top += '.'
            out = [top[: self.cfg.max_chars]]

        summary = " ".join(out).strip()
        return {
            "summary": summary,
            "debug": {
                "n_sentences": n,
                "threshold": self.cfg.threshold,
                "selected_indices": selected_idx,
                "top_scores": [float(scores[i]) for i in order[:10]],
                "analyzer": self.cfg.analyzer,
                "ngram_range": self.cfg.ngram_range,
                "use_global_idf": self.cfg.use_global_idf
            }
        }
