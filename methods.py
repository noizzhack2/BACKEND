"""Helper functions for form parsing from data/ files."""

from typing import Dict, Any, Callable, Tuple, Optional
from models import FormField, AdaptiveForm


def parse_form_from_text(form_name: str, form_content: str) -> AdaptiveForm:
    """
    Parse a form from predefined text content in data/ folder.
    Extracts title, description, and fields from the structured text file.
    Returns an AdaptiveForm object with all parsed information.
    """
    lines = form_content.split('\n')

    # Extract title (first non-empty line with "Form" in it)
    title = "Reimbursement Form"
    description = "Submit your reimbursement request using this form."
    endpoint_url = "/submit_form"
    fields = []

    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # Extract title from first line
        if i < 5 and "Form" in line_stripped:
            title = line_stripped
            # Try to extract English title before "/" if bilingual
            if "/" in title:
                title = title.split("/")[0].strip()
            break

    # Extract description (Purpose section)
    for i, line in enumerate(lines):
        if "Purpose" in line or "מטרה" in line:
            # Get the next non-empty line as description
            for j in range(i + 1, min(i + 5, len(lines))):
                if lines[j].strip() and ":" not in lines[j]:
                    desc_line = lines[j].strip()
                    if "/" in desc_line:
                        # Take only English part if bilingual
                        desc_line = desc_line.split("/")[0].strip()
                    description = desc_line
                    break
            break

    # Extract endpoint URL (look for explicit Endpoint line or first URL)
    for i, line in enumerate(lines):
        ls = line.strip()
        if ls.lower().startswith("endpoint url") or "כתובת קצה" in ls:
            # Next non-empty line should be the URL
            for j in range(i + 1, min(i + 4, len(lines))):
                candidate = lines[j].strip()
                if candidate:
                    endpoint_url = candidate
                    break
            break
    # Fallback: search for any http/https URL in the document if not set
    if endpoint_url == "/submit_form":
        import re
        m = re.search(r"https?://[^\s]+", form_content)
        if m:
            endpoint_url = m.group(0)

    # Extract fields from lines that have required/optional markers
    field_names = set()
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        lower_line = line_stripped.lower()
        # Include both required and optional fields from bullet lines
        if line_stripped.startswith("-") and (
                "required" in lower_line or "optional" in lower_line or "נדרש" in line_stripped or "אופציונלי" in line_stripped
        ):
            # Extract field name (usually before the opening parenthesis)
            if "(" in line_stripped:
                field_name = line_stripped.split("(")[0].strip().lstrip("- ").strip()
                # Extract English name if bilingual
                if "/" in field_name:
                    field_name = field_name.split("/")[0].strip()

                if field_name and len(field_name) > 2 and field_name not in field_names:
                    field_names.add(field_name)
                    field_type = "text"
                    # Determine required flag based on label
                    required_flag = ("required" in lower_line or "נדרש" in line_stripped)

                    # Determine field type
                    if any(word in line_stripped.lower() for word in ["date", "תאריך", "MM/DD"]):
                        field_type = "date"
                    elif any(word in line_stripped.lower() for word in ["select", "בחר", "dropdown"]):
                        field_type = "select"
                    elif any(word in line_stripped.lower() for word in ["checkbox", "תיבת סימון"]):
                        field_type = "checkbox"
                    elif any(word in line_stripped.lower() for word in ["numeric", "מספר", "number"]):
                        field_type = "number"
                    elif any(word in line_stripped.lower() for word in ["currency", "כספי", "amount"]):
                        field_type = "number"
                    elif any(word in line_stripped.lower() for word in ["text area", "שדה טקסט"]):
                        field_type = "textarea"
                    # Heuristic: notes fields should be textarea
                    elif any(word in lower_line for word in ["notes", "הערות"]):
                        field_type = "textarea"

                    # Peek ahead for API Field Name mapping near this bullet
                    api_field_name = None
                    for j in range(i + 1, min(i + 4, len(lines))):
                        nxt = lines[j].strip()
                        if not nxt:
                            continue
                        if nxt.lower().startswith("api field name:"):
                            api_field_name = nxt.split(":", 1)[1].strip()
                            break

                    # Assign Angular Material icon by field type
                    type_to_icon = {
                        "text": "description",
                        "number": "calculate",
                        "email": "email",
                        "textarea": "notes",
                        "checkbox": "check_box",
                        "date": "event",
                        "select": "list_alt",
                    }
                    icon_name = type_to_icon.get(field_type, "description")

                    fields.append(FormField(
                        name=field_name,
                        type=field_type,
                        label=field_name,
                        required=required_flag,
                        placeholder=f"Enter {field_name.lower()}",
                        current_value=None,
                        icon=icon_name,
                        api_field_name=api_field_name
                    ))

    # If no fields were extracted, create a generic field
    if not fields:
        fields.append(FormField(
            name="details",
            type="textarea",
            label="Form Details",
            required=True,
            placeholder="Enter your request details",
            current_value=None,
            icon="notes",
            api_field_name="details"
        ))
    return AdaptiveForm(
        title=title,
        description=description,
        fields=fields,
        endpoint=endpoint_url,
        instruction_file_name=form_name
    )


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


from typing import List, Dict, Any, Tuple
from datetime import datetime
import json
import re
from models import FormField


def prepare_form_fields_description(form_fields: List[FormField]) -> str:
    """
    Prepare a text description of form fields for the AI prompt.

    Args:
        form_fields: List of FormField objects

    Returns:
        Formatted string describing all form fields
    """
    fields_description = []
    for field in form_fields:
        status = f"FILLED: {field.current_value}" if field.current_value else "EMPTY"
        req_text = "REQUIRED" if field.required else "OPTIONAL"

        fields_description.append(
            f'  - name: "{field.name}", placeholder: "{field.placeholder}", type: {field.type}, '
            f'{req_text}, current_value: {status}'
        )

    return "\n".join(fields_description)


def prepare_conversation_context(history: List[Dict[str, str]]) -> str:
    """
    Format conversation history into a text context.

    Args:
        history: List of conversation messages with 'role' and 'content'

    Returns:
        Formatted conversation history string
    """
    if not history:
        return "No previous conversation"

    history_lines = []
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        history_lines.append(f"{role.capitalize()}: {content}")

    return "\n".join(history_lines)


def build_chat_prompt(
        user_message: str,
        form_fields: List[FormField],
        conversation_history: List[Dict[str, str]]
) -> str:
    """
    Build the AI prompt for form filling conversation.

    Args:
        user_message: The current user message
        form_fields: List of FormField objects
        conversation_history: Previous conversation messages

    Returns:
        Complete prompt string for the AI
    """
    today = datetime.now().strftime("%Y-%m-%d")
    fields_info = prepare_form_fields_description(form_fields)
    conversation_context = prepare_conversation_context(conversation_history)

    prompt = f"""You are a HEBREW friendly and helpful form-filling assistant. Your job is to:
                1. Extract ALL relevant information from the user's message
                2. Map extracted information to the appropriate form fields in the language of the user input
                3. Intelligently infer information (like calculating dates from relative terms like "yesterday", "today")
                4. Fill BOTH required and optional fields if information is available
                5. Your response should focus **SOLELY** on what is MISSING or needs correction for next steps
                6. IMPORTANT: **Do NOT** list the fields you have just filled or extracted in the currect text response. The user sees the form updating automatically. Only mention fields if there is an error with them.
                7. Be warm, concise, supportive, and natural in conversation

    Today's date: {today}

    Previous conversation:
    {conversation_context}

    Current form fields:
    {fields_info}

    User message: "{user_message}"

    First, extract values from the user's message. Then provide:
    1. A JSON object with extracted field values (use field "name" as key)
    2. A friendly, conversational response

    For dates: If user says "yesterday", calculate the actual date. If "today", use {today}.
    For locations: Normalize city names (e.g., "tel aviv" -> "Tel Aviv")
    For notes: Infer purpose or context from the message (e.g., "for an interview" -> "Interview")

    Return your response in this exact format:
    EXTRACTED_VALUES: {{"field_name": "value", "another_field": "value"}}
    RESPONSE: Your conversational message here

    Be natural and conversational in the RESPONSE. Acknowledge what was filled, and ask for missing required fields if any."""

    return prompt


def parse_ai_response(ai_text: str) -> Tuple[Dict[str, Any], str]:
    """
    Parse the AI response to extract field values and conversational text.

    Args:
        ai_text: Raw AI response text

    Returns:
        Tuple of (extracted_values dict, response_text string)
    """
    extracted_values = {}
    response_text = ""

    # Try to extract JSON from EXTRACTED_VALUES
    json_match = re.search(r'EXTRACTED_VALUES:\s*(\{[^}]*\})', ai_text, re.IGNORECASE)
    if json_match:
        try:
            extracted_values = json.loads(json_match.group(1))
        except:
            pass

    # Try to extract conversational response
    response_match = re.search(r'RESPONSE:\s*(.+)', ai_text, re.IGNORECASE | re.DOTALL)
    if response_match:
        response_text = response_match.group(1).strip()
    else:
        # Fallback: use entire response if format not followed
        response_text = ai_text

    return extracted_values, response_text


def update_form_fields(
        form_fields: List[FormField],
        extracted_values: Dict[str, Any]
) -> List[FormField]:
    """
    Update form fields with extracted values from AI.

    Args:
        form_fields: Original FormField objects
        extracted_values: Values extracted by AI

    Returns:
        Updated form fields with new values
    """
    updated_fields = []
    for field in form_fields:
        # Create a copy of the field with updated value
        field_dict = field.model_dump()

        # Only update if field is currently empty and we have a value
        if field.name in extracted_values and not field.current_value:
            field_dict["current_value"] = extracted_values[field.name]

        updated_fields.append(FormField(**field_dict))

    return updated_fields


def check_form_completion(form_fields: List[FormField]) -> bool:
    """
    Check if all required form fields are filled.

    Args:
        form_fields: List of FormField objects

    Returns:
        True if all required fields have values, False otherwise
    """
    return all(
        field.current_value is not None and field.current_value != ""
        or not field.required
        for field in form_fields
    )


def update_conversation_history(
        history: List[Dict[str, str]],
        user_message: str,
        assistant_response: str
) -> List[Dict[str, str]]:
    """
    Add new messages to conversation history.

    Args:
        history: Existing conversation history
        user_message: New user message
        assistant_response: New assistant response

    Returns:
        Updated conversation history
    """
    updated_history = history.copy()
    updated_history.append({"role": "user", "content": user_message})
    updated_history.append({"role": "assistant", "content": assistant_response})
    return updated_history


def process_chat_message(
        user_message: str,
        fields: List[FormField],
        conversation_history: List[Dict[str, str]],
        llm
) -> Tuple[str, List[FormField], bool, List[Dict[str, str]]]:
    """
    Process a chat message for conversational form filling.

    Args:
        user_message: The user's message
        fields: Current FormField objects
        conversation_history: Previous conversation messages
        llm: Language model instance for processing

    Returns:
        Tuple of (response_text, updated_fields, is_complete, updated_history)
    """
    # Build the prompt
    prompt = build_chat_prompt(user_message, fields, conversation_history)

    # Get AI response
    ai_response = llm.invoke(prompt)
    ai_text = ai_response.content if hasattr(ai_response, "content") else str(ai_response)

    # Parse AI response
    extracted_values, response_text = parse_ai_response(ai_text)

    # Update form fields
    updated_fields = update_form_fields(fields, extracted_values)

    # Check completion status
    is_complete = check_form_completion(updated_fields)

    # Update conversation history
    updated_history = update_conversation_history(conversation_history, user_message, response_text)

    return response_text, updated_fields, is_complete, updated_history
