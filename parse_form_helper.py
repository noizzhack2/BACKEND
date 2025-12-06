"""Helper functions for form parsing from data/ files."""

from pydantic import BaseModel, Field
from typing import List

class FormField(BaseModel):
    name: str = Field(description="The name/identifier of the field.")
    type: str = Field(description="The type of input field (e.g., 'text', 'date', 'select', 'number', 'textarea', 'checkbox').")
    label: str = Field(description="The human-readable label for the field.")
    required: bool = Field(description="Whether the field is mandatory.")
    placeholder: str = Field(default="", description="Placeholder text for the field.")
    options: List[str] = Field(default_factory=list, description="For select fields, the list of available options.")
    initial_value: None = Field(default=None, description="The default value for the field, if any.")

class AdaptiveForm(BaseModel):
    title: str = Field(description="A clear and concise title for the form.")
    description: str = Field(description="A brief explanation of the form's purpose.")
    fields: List[FormField] = Field(description="A list of all required input fields.")

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
    
    # Extract fields from lines that have "required" or "נדרש"
    field_names = set()
    for line in lines:
        line_stripped = line.strip()
        if ("required" in line_stripped.lower() or "נדרש" in line_stripped) and line_stripped.startswith("-"):
            # Extract field name (usually before the opening parenthesis)
            if "(" in line_stripped:
                field_name = line_stripped.split("(")[0].strip().lstrip("- ").strip()
                # Extract English name if bilingual
                if "/" in field_name:
                    field_name = field_name.split("/")[0].strip()
                
                if field_name and len(field_name) > 2 and field_name not in field_names:
                    field_names.add(field_name)
                    field_type = "text"
                    
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
                    
                    fields.append(FormField(
                        name=field_name,
                        type=field_type,
                        label=field_name,
                        required=True,
                        placeholder=f"Enter {field_name.lower()}",
                        initial_value=None
                    ))
    
    # If no fields were extracted, create a generic field
    if not fields:
        fields.append(FormField(
            name="details",
            type="textarea",
            label="Form Details",
            required=True,
            placeholder="Enter your request details",
            initial_value=None
        ))
    return AdaptiveForm(
        title=title,
        description=description,
        fields=fields
    )
