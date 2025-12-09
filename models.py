"""Pydantic models for form structures."""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal, Optional


class FormField(BaseModel):
    name: str = Field(description="A unique programmatic ID for the field (e.g., 'user_email').")
    label: str = Field(description="The user-facing label for the field (e.g., 'Your Full Name').")
    heb_name: Optional[str] = Field(default=None, description="The field name in Hebrew for display and localization.")
    type: Literal["text", "number", "email", "textarea", "checkbox", "date", "select"] = Field(
        description="The HTML input type."
    )
    current_value: Optional[Any] = Field(default=None, description="The current value for the field, if any.")
    required: bool = Field(description="Whether the field is mandatory.")
    placeholder: Optional[str] = Field(default=None, description="Helper text shown inside the field.")
    icon: Optional[str] = Field(default=None, description="Angular Material icon name associated with the field.")
    api_field_name: Optional[str] = Field(default=None, description="Canonical API field name for backend payloads.")


class AdaptiveForm(BaseModel):
    title: str = Field(description="A clear and concise title for the form.")
    description: str = Field(description="A brief explanation of the form's purpose.")
    fields: List[FormField] = Field(description="A list of all required input fields.")
    endpoint: str = Field(description="The API endpoint to submit the form data to.", default="/submit_form")
    instruction_file_name: str = Field(
        description="The source instruction file name (e.g., 'reimbursement_of_parking_expenses.txt').", default="")


class FormRequest(BaseModel):
    user_request: str = Field(
        description="The user's natural language request for a form (e.g., 'I need a trip expense report form').")
    exclude: List[str] = Field(default_factory=list,
                               description="List of form titles to exclude from matching results.")


class StartChatRequest(BaseModel):
    form_name: str = Field(description="form name")


class MatchedForm(BaseModel):
    title: str
    score: float
    title_heb: str | None = None
    description_en: str | None = None
    description_heb: str | None = None


class MatchedFormsResponse(BaseModel):
    results: List[MatchedForm]


class ChatMessage(BaseModel):
    message: str = Field(description="The user's chat message or response")
    fields: List[FormField] = Field(default_factory=list,
                                    description="Current form fields being filled")
    history: List[Dict[str, str]] = Field(default_factory=list,
                                          description="Conversation history: list of {'role': 'user'|'assistant', 'content': '...'}")
    endpoint: Optional[str] = Field(default=None, description="Form submission endpoint carried from start_chat")


class ChatResponse(BaseModel):
    response: str = Field(
        description="The AI assistant's conversational response or question to continue filling the form")
    fields: List[FormField] = Field(
        description="Updated form fields with new values filled by AI based on user message")
    is_complete: bool = Field(default=False, description="True if all required fields are filled and valid")
    history: List[Dict[str, str]] = Field(description="Updated conversation history including the new exchange")
    endpoint: Optional[str] = Field(default=None, description="The API endpoint to submit the form data to")
