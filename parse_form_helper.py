"""Helper functions for form parsing from data/ files."""

from typing import List, Optional, Any
from models import FormField as ModelFormField, AdaptiveForm as ModelAdaptiveForm

# def parse_form_from_text(form_name: str, form_content: str) -> AdaptiveForm:
    

def parse_form_from_text(form_name: str, form_content: str) -> ModelAdaptiveForm:
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
    fields: List[ModelFormField] = []
    
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
            for j in range(i+1, min(i+5, len(lines))):
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
            for j in range(i+1, min(i+4, len(lines))):
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
    
    # Extract fields: accept bilingual bullets or ones with markers
    field_names = set()
    print("Starting field extraction...")
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        print("Processing line for fields:", line_stripped)
        lower_line = line_stripped.lower()
        if not line_stripped.startswith("-"):
            continue
        has_marker = ("required" in lower_line or "optional" in lower_line or "נדרש" in line_stripped or "אופציונלי" in line_stripped)
        is_bilingual = ("/" in line_stripped)
        if not (has_marker or is_bilingual):
            continue

        # Remove parentheses markers and split on '/'
        import re
        cleaned = re.sub(r"\([^)]*\)", "", line_stripped)
        cleaned = cleaned.lstrip("- ").strip()
        parts = [p.strip() for p in cleaned.split("/", 1)]
        eng_name = parts[0] if parts else cleaned
        heb_name = parts[1] if len(parts) > 1 else None
        field_name = eng_name
        if not field_name or field_name in field_names:
            continue
        field_names.add(field_name)

        # Infer type
        field_type = "text"
        if any(word in lower_line for word in ["date", "תאריך", "mm/dd"]):
            field_type = "date"
        elif any(word in lower_line for word in ["select", "בחר", "dropdown"]):
            field_type = "select"
        elif any(word in lower_line for word in ["checkbox", "תיבת סימון"]):
            field_type = "checkbox"
        elif any(word in lower_line for word in ["numeric", "מספר", "number"]):
            field_type = "number"
        elif any(word in lower_line for word in ["currency", "כספי", "amount"]):
            field_type = "number"
        elif any(word in lower_line for word in ["text area", "שדה טקסט"]):
            field_type = "textarea"
        elif any(word in lower_line for word in ["notes", "הערות"]):
            field_type = "textarea"

        # Required flag
        required_flag = ("required" in lower_line or "נדרש" in line_stripped)

        # Peek ahead for API Field Name mapping near this bullet
        api_field_name = None
        for j in range(i+1, min(i+4, len(lines))):
            nxt = lines[j].strip()
            if not nxt:
                continue
            if nxt.lower().startswith("api field name:"):
                api_field_name = nxt.split(":", 1)[1].strip()
                break

        # Icon mapping
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

        fields.append(ModelFormField(
            name=field_name,
            label=heb_name if heb_name else field_name,
            heb_name=heb_name,
            type=field_type,
            current_value=None,
            required=required_flag,
            placeholder=f"Enter {field_name.lower()}",
            icon=icon_name,
            api_field_name=api_field_name
        ))

    if not fields:
        fields.append(ModelFormField(
            name="details",
            label="Form Details",
            type="textarea",
            current_value=None,
            required=True,
            placeholder="Enter your request details",
            icon="notes",
            api_field_name="details"
        ))
    return ModelAdaptiveForm(
        title=title,
        description=description,
        fields=fields,
        endpoint=endpoint_url,
        instruction_file_name=form_name
    )
