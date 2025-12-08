from typing import Dict, Any, Callable, Tuple, Optional


def match_form_by_name(
    user_text: str,
    form_index: Dict[str, Dict[str, Any]]
) -> Optional[Tuple[str, str]]:
    """
    Try to match form by instruction file name.

    Args:
        user_text: User's request text (already lowercased)
        form_index: Dictionary of indexed forms

    Returns:
        Tuple of (form_name, form_content) if match found, None otherwise
    """
    for form_name, info in form_index.items():
        fn_l = form_name.lower()
        if user_text == fn_l or user_text in fn_l or fn_l in user_text:
            return (form_name, info["content"])
    return None



def match_form_by_embeddings(
    user_text: str,
    form_index: Dict[str, Dict[str, Any]],
    embeddings_provider: Callable[[], Any],
    cosine_similarity: Callable
) -> Optional[Tuple[str, str]]:
    """
    Match form using semantic embeddings.

    Args:
        user_text: User's request text
        form_index: Dictionary of indexed forms
        embeddings_provider: Function that returns the embeddings object
        cosine_similarity: Function to calculate cosine similarity

    Returns:
        Tuple of (form_name, form_content) if match found, None otherwise
    """
    embeddings = embeddings_provider()
    if not embeddings:
        return None

    user_emb = embeddings.embed_query(user_text)
    best_form = None
    best_score_local = -1.0

    for form_name, info in form_index.items():
        form_emb = info.get("embedding") or []
        score = cosine_similarity(user_emb, form_emb)
        if score > best_score_local:
            best_score_local = score
            best_form = (form_name, info["content"])

    return best_form


def match_form_by_text_search(
    user_text: str,
    form_index: Dict[str, Dict[str, Any]]
) -> Optional[Tuple[str, str]]:

    for form_name, info in form_index.items():
        if form_name in user_text or user_text in form_name:
            return (form_name, info["content"])
    return None


def find_matching_form(
    user_text: str,
    form_index: Dict[str, Dict[str, Any]],
    embeddings_provider: Callable[[], Any],
    cosine_similarity: Callable
) -> Optional[Tuple[str, str]]:

    user_text_l = user_text.lower()

    # Strategy 1: Try exact filename match
    result = match_form_by_name(user_text_l, form_index)
    if result:
        return result

    # Strategy 2: Try embeddings-based match
    result = match_form_by_embeddings(user_text, form_index, embeddings_provider, cosine_similarity)
    if result:
        return result

    # Strategy 3: Fallback to text search
    result = match_form_by_text_search(user_text_l, form_index)
    return result
