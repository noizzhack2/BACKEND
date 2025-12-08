"""Helper functions for form parsing from data/ files."""

from pydantic import BaseModel, Field
from typing import List
from typing import List, Literal, Optional, Dict, Any

class FormField(BaseModel):
    name: str = Field(description="A unique programmatic ID for the field (e.g., 'user_email').")
    label: str = Field(description="The user-facing label for the field (e.g., 'Your Full Name').")
    type: Literal["text", "number", "email", "textarea", "checkbox", "date", "select"] = Field(
        description="The HTML input type."
    )
    current_value: Optional[Any] = Field(default=None, description="The current value for the field, if any.")
    required: bool = Field(description="Whether the field is mandatory.")
    placeholder: Optional[str] = Field(default=None, description="Helper text shown inside the field.")
    icon: Optional[str] = Field(default=None, description="Angular Material icon name associated with the field.")
    api_field_name: Optional[str] = Field(default=None, description="Canonical API field name for backend payloads.")
    
# 2.2. Define the complete adaptive form structure
class AdaptiveForm(BaseModel):
    title: str = Field(description="A clear and concise title for the form.")
    description: str = Field(description="A brief explanation of the form's purpose.")
    fields: List[FormField] = Field(description="A list of all required input fields.")
    endpoint:str = Field(description="The API endpoint to submit the form data to.", default="/submit_form")
    instruction_file_name: str = Field(description="The source instruction file name (e.g., 'reimbursement_of_parking_expenses.txt').", default="")

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
                    for j in range(i+1, min(i+4, len(lines))):
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
