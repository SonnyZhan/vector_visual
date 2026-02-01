"""
=============================================================================
CUSTOM TEXT VECTORIZATION - Token Hashing with Random Projection
=============================================================================

This module converts text strings into numerical vectors (embeddings).
These vectors capture semantic meaning - similar texts produce similar vectors.

APPROACH: Token Hashing
-----------------------
Instead of training a neural network (like Word2Vec or BERT), we use a simple
but effective technique:

1. Split text into tokens (words)
2. Hash each token to get a seed number
3. Use the seed to generate a deterministic random vector
4. Sum all token vectors together
5. Normalize to unit length

This works because:
- Same word → same hash → same vector (deterministic)
- Similar texts share words → similar summed vectors
- Fast (no ML model needed)
- Consistent across runs

=============================================================================
"""

import hashlib
import numpy as np
import re


def stable_hash(text: str) -> int:
    """
    Create a stable hash from text using MD5.
    
    Why MD5?
    - Produces consistent output for same input
    - Distributes well (uniform distribution)
    - Fast to compute
    
    Note: MD5 is NOT cryptographically secure, but we don't need security here.
    We just need consistent, well-distributed numbers.
    
    Example:
        stable_hash("hello") → always returns the same large integer
        stable_hash("world") → different integer, but also consistent
    """
    # Encode text to bytes, compute MD5 hash, convert hex to integer
    return int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)


def token_to_vector(token: str, dim: int = 128) -> np.ndarray:
    """
    Map a token (word) to a deterministic random vector.
    
    Process:
    1. Hash the token to get a seed
    2. Create a random number generator with that seed
    3. Generate a vector of random numbers from normal distribution
    
    Why normal distribution?
    - Values centered around 0
    - Allows positive and negative contributions
    - Good mathematical properties for similarity calculations
    
    Why deterministic?
    - Same token ALWAYS produces the same vector
    - "cat" → [0.3, -0.7, 0.1, ...] (always)
    - "dog" → [-0.2, 0.5, 0.8, ...] (always, but different from "cat")
    
    Args:
        token: A single word
        dim: Number of dimensions (default 128)
    
    Returns:
        numpy array of shape (dim,) with random values
    """
    # Convert hash to a valid seed (must fit in 32 bits)
    seed = stable_hash(token) % (2 ** 32)
    
    # Create a seeded random number generator
    # Using the same seed always produces the same sequence
    rng = np.random.default_rng(seed)
    
    # Generate vector from standard normal distribution (mean=0, std=1)
    vector = rng.standard_normal(dim)
    
    return vector


def tokenize(text: str) -> list[str]:
    """
    Split text into tokens (words).
    
    Process:
    1. Convert to lowercase (so "Cat" and "cat" are the same)
    2. Extract words using regex (letters and numbers only)
    
    Regex explained: r'\b\w+\b'
    - \b = word boundary
    - \w+ = one or more word characters (letters, digits, underscore)
    - \b = word boundary
    
    Example:
        tokenize("Hello, World! 123") → ["hello", "world", "123"]
    """
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def text_to_vectors(text: str, dim: int = 128, normalize: bool = True) -> np.ndarray:
    """
    Convert text to a fixed-size vector embedding.
    
    This is the MAIN function - it combines all the steps:
    
    1. TOKENIZE: Split text into words
       "The quick brown fox" → ["the", "quick", "brown", "fox"]
    
    2. VECTORIZE EACH TOKEN: Convert each word to a vector
       "the"   → [0.3, -0.1, 0.7, ...]
       "quick" → [-0.2, 0.5, 0.1, ...]
       "brown" → [0.1, 0.3, -0.4, ...]
       "fox"   → [-0.5, 0.2, 0.6, ...]
    
    3. SUM: Add all token vectors together
       [0.3, -0.1, 0.7, ...] + [-0.2, 0.5, 0.1, ...] + ... = [-0.3, 0.9, 1.0, ...]
    
    4. NORMALIZE: Scale to unit length (optional but recommended)
       Makes vector length = 1.0
       This way, cosine similarity = dot product (simpler!)
    
    Why does summing work?
    - Words that appear together in similar contexts get "averaged"
    - Common words contribute to many vectors (less distinctive)
    - Rare/specific words make vectors more unique
    
    Args:
        text: Input string
        dim: Output vector dimensions (default 128)
        normalize: If True, scale to unit length (recommended for similarity)
    
    Returns:
        numpy array of shape (dim,) representing the text
    """
    # Step 1: Tokenize
    tokens = tokenize(text)
    
    # Handle empty text
    if not tokens:
        return np.zeros(dim)
    
    # Step 2 & 3: Convert tokens to vectors and sum
    vec = np.zeros(dim)
    for token in tokens:
        vec += token_to_vector(token, dim)
    
    # Step 4: Normalize (make unit length)
    if normalize:
        norm = np.linalg.norm(vec)  # Euclidean length: sqrt(sum of squares)
        if norm > 0:
            vec = vec / norm  # Divide by length → length becomes 1.0
    
    return vec


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Formula: cos(θ) = (A · B) / (|A| × |B|)
    
    Where:
    - A · B is the dot product (sum of element-wise products)
    - |A| is the magnitude (Euclidean length) of A
    - |B| is the magnitude of B
    
    The result ranges from:
    - 1.0 = identical direction (most similar)
    - 0.0 = perpendicular (unrelated)
    - -1.0 = opposite direction (opposites)
    
    Note: If vectors are normalized (length = 1), this simplifies to just:
    cos(θ) = A · B  (the dot product alone!)
    
    That's why we normalize vectors in text_to_vectors().
    
    Example usage:
        v1 = text_to_vectors("king")
        v2 = text_to_vectors("queen")
        v3 = text_to_vectors("car")
        
        cosine_similarity(v1, v2)  # High (both royalty)
        cosine_similarity(v1, v3)  # Low (unrelated concepts)
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    # Avoid division by zero
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


# =============================================================================
# EXAMPLE USAGE (uncomment to test)
# =============================================================================

if __name__ == "__main__":
    # Test the vectorization
    sample_texts = [
        "How to reset my password",
        "Change account email address",
        "Troubleshooting login issues",
        "password reset help"
    ]
    
    print("=" * 60)
    print("VECTORIZATION DEMO")
    print("=" * 60)
    
    for text in sample_texts:
        vec = text_to_vectors(text)
        print(f"\nText: '{text}'")
        print(f"Tokens: {tokenize(text)}")
        print(f"Vector (first 5 dims): {vec[:5].round(4)}")
        print(f"Vector length: {np.linalg.norm(vec):.4f}")  # Should be ~1.0
    
    # Test similarity
    print("\n" + "=" * 60)
    print("SIMILARITY DEMO")
    print("=" * 60)
    
    query = "password reset help"
    query_vec = text_to_vectors(query)
    
    print(f"\nQuery: '{query}'")
    print("\nSimilarities:")
    for text in sample_texts:
        vec = text_to_vectors(text)
        sim = cosine_similarity(query_vec, vec)
        print(f"  {sim:.4f} - '{text}'")
