from typing import List, Dict, Any
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI


class ChatMessage(BaseModel):
    message: str = Field(description="The user's chat message or response")
    form_fields: List[Dict[str, Any]] = Field(default_factory=list,
                                              description="Current form fields being filled. Each field has 'id', 'label', 'value', 'type', 'required', etc.")
    history: List[Dict[str, str]] = Field(default_factory=list, description="Conversation history: list of {'role': 'user'|'assistant', 'content': '...'}")


class ChatResponse(BaseModel):
    response: str = Field(
        description="The AI assistant's conversational response or question to continue filling the form")
    form_fields: List[Dict[str, Any]] = Field(
        description="Updated form fields with new values filled by AI based on user message")
    is_complete: bool = Field(default=False, description="True if all required fields are filled and valid")
    history: List[Dict[str, str]] = Field(description="Updated conversation history including the new exchange")


def register_chat_routes(api: FastAPI):
    """Register chat-related API routes."""

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

            # Prepare current form state
            form_fields = request.form_fields or []

            # Import needed modules
            import json
            from datetime import datetime, timedelta

            # Get today's date for reference
            today = datetime.now().strftime("%Y-%m-%d")

            # Prepare form fields description for AI
            fields_description = []
            for field in form_fields:
                field_name = field.get("name", "")
                placeholder = field.get("placeholder", "")
                field_type = field.get("type", "text")
                required = field.get("required", False)
                current_value = field.get("value")
                validation = field.get("validation", {})

                status = f"FILLED: {current_value}" if current_value else "EMPTY"
                req_text = "REQUIRED" if required else "OPTIONAL"

                fields_description.append(
                    f'  - name: "{field_name}", placeholder: "{placeholder}", type: {field_type}, '
                    f'{req_text}, validation: {validation}, current_value: {status}'
                )

            fields_info = "\n".join(fields_description)

            # Build conversation history context
            conversation_context = ""
            if request.history:
                history_lines = []
                for msg in request.history:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    history_lines.append(f"{role.capitalize()}: {content}")
                conversation_context = "\n".join(history_lines)

            # Build comprehensive prompt for extraction and conversation
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
{conversation_context if conversation_context else "No previous conversation"}

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

            ai_response = llm.invoke(prompt)
            ai_text = ai_response.content if hasattr(ai_response, "content") else str(ai_response)

            # Parse the AI response to extract values and conversational response
            import re

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

            # Update form fields with extracted values
            updated_fields = []
            for field in form_fields:
                field_copy = field.copy()
                field_name = field_copy.get("name", "")

                # Only update if field is currently empty and we have a value
                if field_name in extracted_values and not field_copy.get("value"):
                    field_copy["value"] = extracted_values[field_name]

                updated_fields.append(field_copy)

            # Check if form is complete (all required fields filled)
            is_complete = all(
                field.get("value") is not None and field.get("value") != ""
                or not field.get("required", False)
                for field in updated_fields
            )

            # Update conversation history
            updated_history = request.history.copy()
            updated_history.append({"role": "user", "content": user_message})
            updated_history.append({"role": "assistant", "content": response_text})

            return ChatResponse(
                response=response_text,
                form_fields=updated_fields,
                is_complete=is_complete,
                history=updated_history
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process chat message: {str(e)}")
