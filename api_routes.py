from typing import List, Dict, Any, Callable
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from parse_form_helper import AdaptiveForm
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import Field 

class FormRequest(BaseModel):
    user_request: str = Field(description="The user's natural language request for a form (e.g., 'I need a trip expense report form').")


class MatchedForm(BaseModel):
    title: str
    score: float


class MatchedFormsResponse(BaseModel):
    results: List[MatchedForm]


def register_routes(
    api: FastAPI,
    form_index: Dict[str, Dict[str, Any]],
    embeddings_provider: Callable[[], Any],
    parse_form_from_text,
    cosine_similarity,
):
    @api.post("/matched_forms", response_model=MatchedFormsResponse)
    def matched_forms_endpoint(request: FormRequest):  # type: ignore[attr-defined]
        """
        Returns a list of matched forms from the data folder with score > 0.6.
        Each result contains 'title' and 'score'.
        """
        if not form_index:
            raise HTTPException(status_code=500, detail="No indexed forms available. Ensure data folder contains form files.")

        user_text = (request.user_request or "").strip()
        if not user_text:
            raise HTTPException(status_code=414, detail="לא נשלח טקסט בבקשה. נסה לנסח בקשה בעברית או באנגלית.")

        results: List[MatchedForm] = []
        embeddings = embeddings_provider()

        # Optionally expand the user request with semantically related terms via LLM
        def expand_query(text: str) -> str:
            api_key = os.getenv("GENERATIVE_AI_KEY")
            print("Expanding query using LLM...", text)
            if not api_key:
                return text
            try:
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
                prompt = (
                    "You are a helpful assistant. Given a short user request, "
                    "produce a compact comma-separated list of closely related words and phrases "
                    "in English and Hebrew that capture the same context (no explanations).\n\n"
                    f"Request: {text}\nRelated terms:"
                )
                resp = llm.invoke(prompt)
                related = ""
                try:
                    related = resp.content if hasattr(resp, "content") else str(resp)
                except Exception:
                    related = str(resp)
                # Append related terms to original text for richer embedding
                return f"{text}\nRelated: {related}"
            except Exception:
                return text

        augmented_text = expand_query(user_text)
        # Prepare user keywords from augmented text (simple tokenization)
        text_lower = augmented_text.lower()
        user_tokens = {t for t in [w.strip(" ,.:;()[]{}\n\t").lower() for w in text_lower.split()] if t}
        print("Augmented user text for matching:", user_tokens)

        def extract_form_keywords(content: str) -> List[str]:
            lines = content.splitlines()
            for i, line in enumerate(lines[:50]):
                if "keywords" in line.lower():
                    # concatenate next few lines until blank
                    collected: List[str] = []
                    for j in range(i + 1, min(i + 6, len(lines))):
                        if not lines[j].strip():
                            break
                        collected.append(lines[j])
                    blob = ", ".join(collected).lower()
                    # split by commas
                    return [kw.strip() for kw in blob.split(",") if kw.strip()]
            return []

        def keyword_overlap_score(form_keywords: List[str]) -> float:
            print("Calculating keyword overlap score...", user_text, form_keywords)
            if not form_keywords or not user_tokens:
                return 0.0
            kws = {k.strip().lower() for k in form_keywords if k.strip()}
            overlap = kws.intersection(user_tokens)
            print("Keyword overlap:", overlap)
            # Jaccard-like: overlap / sqrt(len(kws)*len(user_tokens)) to normalize
            denom = (len(kws) * len(user_tokens)) ** 0.5
            res = float(len(overlap) / denom)*10 if denom else 0.0
            print("Keyword overlap score:", len(overlap), denom, res)
            return res

        if embeddings:
            # Use the augmented text to compute the query embedding
            user_emb = embeddings.embed_query(augmented_text)
            for form_name, info in form_index.items():
                form_emb = info.get("embedding") or []
                # Embedding similarity
                emb_score = float(cosine_similarity(user_emb, form_emb))
                # Keyword-based score
                form_keywords = extract_form_keywords(info.get("content", ""))
                kw_score = keyword_overlap_score(form_keywords)
                # Combine scores (weighted); favor embeddings but include keywords for robustness
                print(f"Form: {form_name}, Embedding score: {emb_score}, Keyword score: {kw_score}")
                combined = 0.3 * emb_score + 0.7 * kw_score
                if combined > 0.2:
                    results.append(MatchedForm(title=form_name, score=round(combined, 2)))
            results.sort(key=lambda x: x.score, reverse=True)
            return MatchedFormsResponse(results=results)
        else:
            text = user_text.lower()
            for form_name in form_index:
                if form_name in text or text in form_name:
                    results.append(MatchedForm(title=form_name, score=0.0))
            return MatchedFormsResponse(results=results)

    @api.post("/generate_form", response_model=AdaptiveForm)
    def generate_form_endpoint(request: FormRequest):  # type: ignore[attr-defined]
        """
        Accepts a user request (in Hebrew or English) and returns a structured JSON schema for an adaptive form.
        Uses precomputed form embeddings (from `form_index`) to select the best match.
        Supports both RTL (Hebrew) and LTR (English) text.
        """
        if not form_index:
            raise HTTPException(status_code=500, detail="No indexed forms available. Ensure data folder contains form files.")

        try:
            user_text = (request.user_request or "").strip()
            if not user_text:
                raise HTTPException(status_code=414, detail="לא נשלח טקסט בבקשה. נסה לנסח בקשה בעברית או באנגלית.")

            food_keywords = ["אוכל", "meal", "food", "ארוחה"]
            if any(k in user_text.lower() for k in food_keywords):
                for form_name, info in form_index.items():
                    if "food" in form_name or "meal" in form_name:
                        matched_form = form_name
                        form_content = info["content"]
                        best_score = 1.0
                        parsed_form = parse_form_from_text(matched_form, form_content)
                        parsed_form.score = 1.0
                        return parsed_form

            embeddings = embeddings_provider()
            if embeddings:
                user_emb = embeddings.embed_query(user_text)
                best_score = -1.0
                best_form = None
                for form_name, info in form_index.items():
                    form_emb = info.get("embedding") or []
                    score = cosine_similarity(user_emb, form_emb)
                    if score > best_score:
                        best_score = score
                        best_form = (form_name, info["content"])
                if best_form is None:
                    raise HTTPException(status_code=414, detail="לא נמצא טופס מתאים לבקשה שלך. נסה לנסח מחדש או לבחור טופס קיים.")
                matched_form, form_content = best_form
            else:
                text = user_text.lower()
                matched_form = None
                for form_name, info in form_index.items():
                    if form_name in text or text in form_name:
                        matched_form = form_name
                        form_content = info["content"]
                        best_score = 0.0
                        break
                if not matched_form:
                    raise HTTPException(status_code=414, detail=f"לא נמצא טופס מתאים לבקשה שלך. נסה לנסח מחדש או לבחור טופס קיים. הטפסים הזמינים: {sorted(list(form_index.keys()))}")
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=500, detail="Failed to match form by semantic similarity.")

        try:
            parsed_form = parse_form_from_text(matched_form, form_content)
            parsed_form.score = round(best_score, 2)
            return parsed_form
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to parse form '{matched_form}': {str(e)}")
