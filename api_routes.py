from typing import List, Dict, Any, Callable
import os
from fastapi import FastAPI, HTTPException
from models import *
from langchain_google_genai import ChatGoogleGenerativeAI
from methods import find_matching_form, process_chat_message


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
            raise HTTPException(status_code=500,
                                detail="No indexed forms available. Ensure data folder contains form files.")

        user_text = (request.user_request or "").strip()
        # Normalize exclude list for comparison
        exclude_set = {e.strip().lower() for e in (request.exclude or []) if e and e.strip()}
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
                llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
                prompt = (
                    "You are a helpful assistant. Given a short user request, return a compact, "
                    "high-context expansion: a comma-separated list of domain-specific synonyms, each have ONLY one NOUN WORD!!!!, "
                    "related entities, intents, attributes,nouns and common acronyms in BOTH English!! and Hebrew!!. "
                    "Focus on meaningful terms only (no stopwords, no explanations, no numbering, NO verbs, ONLY!!! **ONE WORD** PHRASE). "
                    "Keep it under ~60 tokens, maximize semantic coverage.\n\n"
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

        def has_hebrew(text: str) -> bool:
            return any("\u0590" <= ch <= "\u05FF" for ch in text)

        def has_latin(text: str) -> bool:
            return any(('a' <= ch.lower() <= 'z') for ch in text)

        def extract_purpose_bilingual(content: str) -> tuple[str, str]:
            lines = content.splitlines()
            for i, line in enumerate(lines):
                low = line.strip().lower()
                if "purpose" in low or "מטרה" in low:
                    collected: List[str] = []
                    for j in range(i + 1, len(lines)):
                        nxt = lines[j]
                        if not nxt.strip():
                            break
                        collected.append(nxt.strip())
                    en_lines: List[str] = []
                    he_lines: List[str] = []
                    for t in collected:
                        if has_hebrew(t):
                            he_lines.append(t)
                        if has_latin(t) and not has_hebrew(t):
                            en_lines.append(t)
                    return (" ".join(en_lines).strip(), " ".join(he_lines).strip())
            return ("", "")

        def translate_text(text: str, target_lang: str) -> str:
            api_key = os.getenv("GENERATIVE_AI_KEY")
            if not text:
                return ""
            if not api_key:
                return text
            try:
                llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
                prompt = (
                    f"Translate the following text to {target_lang}. Return ONLY the translation, no explanations, no quotes.\n"
                    f"Text: {text}"
                )
                resp = llm.invoke(prompt)
                try:
                    return resp.content if hasattr(resp, "content") else str(resp)
                except Exception:
                    return str(resp)
            except Exception:
                return text

        def keyword_overlap_score(form_keywords: List[str]) -> float:
            print("Calculating keyword overlap score...", user_tokens, form_keywords)
            if not form_keywords or not user_tokens:
                return 0.0
            kws = {k.strip().lower() for k in form_keywords if k.strip()}
            overlap = kws.intersection(user_tokens)
            print("Keyword overlap:", overlap)
            # Jaccard-like: overlap / sqrt(len(kws)*len(user_tokens)) to normalize
            denom = (len(kws) * len(user_tokens)) ** 0.5
            res = float(len(overlap) / denom) * 10 if denom else 0.0
            print("Keyword overlap score:", len(overlap), denom, res)
            return res

        if embeddings:
            # Use the augmented text to compute the query embedding once
            user_emb = embeddings.embed_query(augmented_text)

            # Process each form in parallel to reduce latency
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def process_form(item):
                form_name, info = item
                try:
                    form_emb = info.get("embedding") or []
                    emb_score = float(cosine_similarity(user_emb, form_emb))
                    form_keywords = extract_form_keywords(info.get("content", ""))
                    kw_score = keyword_overlap_score(form_keywords)
                    combined = 0.3 * emb_score + 0.7 * kw_score
                    print(f"Form: {form_name}, Embedding score: {emb_score}, Keyword score: {kw_score}, Combined: {combined}")
                    if combined <= 0.6:
                        return None
                    # Apply exclusion filter
                    if form_name.strip().lower() in exclude_set:
                        return None

                    purpose_en, purpose_heb = extract_purpose_bilingual(info.get("content", ""))
                    desc_en = purpose_en or ""
                    desc_heb = purpose_heb or ""
                    if not desc_en and desc_heb:
                        desc_en = translate_text(desc_heb, "English")
                    if not desc_heb and desc_en:
                        desc_heb = translate_text(desc_en, "Hebrew")
                    title_heb = translate_text(form_name, "Hebrew")
                    return MatchedForm(
                        title=form_name,
                        score=round(combined, 2),
                        title_heb=title_heb,
                        description_en=desc_en,
                        description_heb=desc_heb,
                    )
                except Exception:
                    return None

            # Limit workers to avoid overwhelming external services (LLM/translation)
            max_workers = min(8, max(2, os.cpu_count() or 4))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_item = {executor.submit(process_form, item): item for item in form_index.items()}
                print("Submitted form matching tasks:", len(future_to_item))
                for future in as_completed(future_to_item):
                    mf = future.result()
                    if mf:
                        results.append(mf)
            print("Total matched forms found:", len(results))
            results.sort(key=lambda x: x.score, reverse=True)
            return MatchedFormsResponse(results=results)
        else:
            text = user_text.lower()
            for form_name in form_index:
                if form_name in text or text in form_name:
                    if form_name.strip().lower() in exclude_set:
                        continue
                    info = form_index.get(form_name, {})
                    purpose_en, purpose_heb = extract_purpose_bilingual(info.get("content", ""))
                    desc_en = purpose_en or ""
                    desc_heb = purpose_heb or ""
                    if not desc_en and desc_heb:
                        desc_en = translate_text(desc_heb, "English")
                    if not desc_heb and desc_en:
                        desc_heb = translate_text(desc_en, "Hebrew")
                    title_heb = translate_text(form_name, "Hebrew")
                    results.append(MatchedForm(
                        title=form_name,
                        score=0.0,
                        title_heb=title_heb,
                        description_en=desc_en,
                        description_heb=desc_heb,
                    ))
            return MatchedFormsResponse(results=results)

    @api.post("/start_chat", response_model=ChatResponse)
    def start_chat(request: StartChatRequest):  # type: ignore[attr-defined]
        """
        Accepts a user request (in Hebrew or English) and returns a structured JSON schema for an adaptive form.
        Uses precomputed form embeddings (from `form_index`) to select the best match.
        Supports both RTL (Hebrew) and LTR (English) text.
        """
        api_key = os.getenv("GENERATIVE_AI_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GENERATIVE_AI_KEY not configured")

        if not form_index:
            raise HTTPException(status_code=500,
                                detail="No indexed forms available. Ensure data folder contains form files.")

        form_name = (request.form_name or "").strip()
        if not form_name:
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        try:
            # Use the extracted logic to find matching form
            match_result = find_matching_form(form_name, form_index, embeddings_provider, cosine_similarity)

            matched_form, form_content = match_result

            parsed_form = parse_form_from_text(matched_form, form_content)
            print("Parsed form:", parsed_form)
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
            fields = parsed_form.fields or []
            # Ensure Hebrew labels are present; fallback to name if missing
            for f in fields:
                if not getattr(f, "label", None):
                    try:
                        # If name contains bilingual format, extract Hebrew after '/'
                        name = getattr(f, "name", "") or ""
                        print("Processing field name for label:", name)
                        if "/" in name:
                            parts = [p.strip() for p in name.split("/")]
                            if len(parts) > 1 and parts[1]:
                                f.label = parts[1]
                        # Otherwise keep existing or name
                        if not getattr(f, "label", None):
                            f.label = name
                    except Exception:
                        f.label = getattr(f, "name", "") or ""

            # Use extracted logic to process the chat message
            response_text, updated_fields, is_complete, updated_history = process_chat_message(
                user_message="",
                fields=fields,
                history=[],
                llm=llm
            )


            return ChatResponse(
                response=response_text,
                fields=updated_fields,
                is_complete=is_complete,
                history=updated_history,
                endpoint=getattr(parsed_form, "endpoint", None),
                form_type=getattr(parsed_form, "form_type", None)
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error in start_chat: {e}")

    @api.post("/chat", response_model=ChatResponse)
    def chat_endpoint(request: ChatMessage):
        """
        Handles conversational form filling.
        The AI extracts information from user messages and fills form fields.
        Returns updated form fields and asks for missing required information.
        """
        api_key = os.getenv("GENERATIVE_AI_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GENERATIVE_AI_KEY not configured")

        user_message = (request.message or "").strip()
        if not user_message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        try:
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
            fields = request.fields or []

            # Use extracted logic to process the chat message
            response_text, updated_fields, is_complete, updated_history = process_chat_message(
                user_message=user_message,
                fields=fields,
                history=request.history,
                llm=llm
            )

            return ChatResponse(
                response=response_text,
                fields=updated_fields,
                is_complete=is_complete,
                history=updated_history,
                endpoint=request.endpoint,
                form_type=request.form_type
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error in chat_endpoint: {e}")
