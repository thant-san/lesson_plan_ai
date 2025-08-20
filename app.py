import streamlit as st
import openai
from openai import OpenAI
import PyPDF2
import io
import os
import json

# --- 1. Load OpenAI API configuration ---
# Prefer Streamlit secrets, fallback to environment variables
api_key = None
base_url = None
model_name = None

try:
    api_key = st.secrets.get("OPENAI_API_KEY")
    base_url = st.secrets.get("OPENAI_BASE_URL")
    model_name = st.secrets.get("OPENAI_MODEL")
except Exception:
    pass

if not api_key:
    api_key = os.getenv("OPENAI_API_KEY")
if not base_url:
    base_url = os.getenv("OPENAI_BASE_URL")
if not model_name:
    model_name = os.getenv("OPENAI_MODEL")

if not api_key:
    st.error("OpenAI API key not found. Set OPENAI_API_KEY in .streamlit/secrets.toml or as an environment variable.")
    st.stop()

# Default model depending on provider
if not model_name:
    model_name = "openai/gpt-5-chat-latest" if base_url else "gpt-3.5-turbo"

client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

# --- 2. Function to Extract Text from PDF ---
def extract_text_from_pdf(uploaded_file):
    """
    Extracts text content from an uploaded PDF file.
    Returns the extracted text as a string, or None if an error occurs.
    """
    if uploaded_file is not None:
        try:
            # Resolve PdfReadError across PyPDF2 versions
            try:
                from PyPDF2.errors import PdfReadError as _PdfReadError
            except Exception:
                try:
                    from PyPDF2.utils import PdfReadError as _PdfReadError
                except Exception:
                    class _PdfReadError(Exception):
                        pass

            # PyPDF2 expects a file-like object, io.BytesIO makes the uploaded file readable like one
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            return text
        except _PdfReadError:
            st.error("Invalid PDF file. Please upload a valid, unencrypted PDF.")
            return None
        except Exception as e:
            st.error(f"An error occurred while reading the PDF: {e}")
            return None
    return None

# --- 3. Function to Generate Lesson Plan ---
def generate_lesson_plan(text_content, study_duration_weeks, num_students, sections_per_week):
    """
    Uses an LLM to generate a structured lesson plan based on extracted PDF content,
    total study duration (in weeks), and the number of students.
    """
    if not text_content:
        return "No source content provided. Please upload a PDF with relevant curriculum or material."

    # Keep prompt concise but directive, provide structure for consistent output
    prompt = f"""
You are an expert pedagogy designer. Create a practical, well-structured lesson plan series using the provided source content.

Constraints and context:
- Total duration: {study_duration_weeks} week(s)
- Class size: {num_students} students
- Sections per week: {sections_per_week}
- Source content (use to derive objectives, topics, examples, and assessments):
---
{text_content[:6000]}
---

Requirements:
1) Provide an overview with goals and success criteria tailored to the class size.
2) Break down the plan by week with clear learning objectives, key topics, and vocabulary.
3) For each week, divide into exactly {sections_per_week} section(s). For each section include:
   - Activities (at least one teacher-led, one student-centered, one collaborative activity)
   - Materials/resources
   - Differentiation for mixed abilities and larger group management if applicable
   - Assessment (formative + one summative suggestion across the duration)
4) Add homework/extension ideas and optional enrichment.
5) Keep it concise and actionable. Use markdown headings and bullet points.
"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a senior instructional designer and master teacher."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1200,
            temperature=0.4,
        )
        return response.choices[0].message.content
    except (openai.APIError, openai.APIConnectionError, openai.RateLimitError, openai.BadRequestError) as e:
        st.error(f"OpenAI API Error: {e}. Check your API key and network connection.")
        return "Failed to generate a lesson plan due to an API error."
    except Exception as e:
        st.error(f"An unexpected error occurred with the LLM call: {e}")
        return "Failed to generate a lesson plan due to an unexpected error."


# --- 3b. Function to Generate Assessments ---
def generate_assessments(
    text_content,
    assessment_types,
    mcq_count=0,
    short_count=0,
    include_project=False,
    difficulty="Intermediate",
    include_answers=True,
    project_focus="",
    rubric_detail_level="Detailed",
):
    """
    Generate assessments (MCQ, Short Answer, Project) from the provided content and user preferences.
    """
    if not text_content:
        return "No source content provided. Please upload a PDF with relevant curriculum or material."

    # Build a focused prompt
    parts = []
    if "MCQ" in assessment_types and mcq_count > 0:
        parts.append(f"- MCQs: {mcq_count} questions with 4 options each, one correct answer. Vary difficulty around {difficulty}.")
    if "Short Answer" in assessment_types and short_count > 0:
        parts.append(f"- Short Answer: {short_count} concise prompts targeting key ideas at {difficulty} difficulty.")
    if include_project and "Project" in assessment_types:
        parts.append(f"- Project: A project brief focused on '{project_focus or 'core learning objectives'}' with a {rubric_detail_level.lower()} rubric.")

    include_key_text = "Include an answer key after each section." if include_answers else "Do not include answer keys."

    prompt = f"""
You are an experienced assessment designer. Create assessments based on the source material below.

Assessment specifications:
{chr(10).join(parts) if parts else '- No specific counts provided; propose a balanced set.'}
Overall difficulty: {difficulty}
Formatting: Use markdown headings for each assessment section and bullet lists for items. {include_key_text}

Source content:
---
{text_content[:6000]}
---
"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a meticulous assessment designer who writes clear, fair, and curriculum-aligned items."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1200,
            temperature=0.5,
        )
        return response.choices[0].message.content
    except (openai.APIError, openai.APIConnectionError, openai.RateLimitError, openai.BadRequestError) as e:
        st.error(f"OpenAI API Error: {e}. Check your API key and network connection.")
        return "Failed to generate assessments due to an API error."
    except Exception as e:
        st.error(f"An unexpected error occurred with the LLM call: {e}")
        return "Failed to generate assessments due to an unexpected error."


# --- 3c. Convert assessment text to structured schema via LLM ---
def convert_assessment_text_to_schema(assessment_text):
    """
    Ask the LLM to convert freeform assessment text into a strict JSON schema
    consumable by the Google Forms builder.

    Schema:
    {
      "title": string,
      "items": [
        {"type": "mcq", "question": string, "choices": [string, ...], "correctAnswer": string, "points": int},
        {"type": "short", "question": string, "correctAnswers": [string, ...], "points": int},
        {"type": "paragraph", "question": string, "points": int}
      ]
    }
    """
    system_msg = (
        "You are a converter that outputs only strict JSON (no markdown, no extra text)."
    )
    user_msg = f"Convert the following assessment to the required JSON schema. If points are not provided, default to 1.\n\n---\n{assessment_text}\n---\nReturn ONLY JSON."

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=800,
        )
        content = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        if content.startswith("```"):
            # remove leading/trailing fences with optional language tag
            content = content.strip()
            if content.startswith("```"):
                content = content[3:]
                # drop first line if it is a language
                first_newline = content.find('\n')
                if first_newline != -1:
                    content = content[first_newline + 1 :]
            if content.endswith("```"):
                content = content[: -3]

        # Attempt to locate JSON substring if extra text leaked
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1 and end > start:
                data = json.loads(content[start : end + 1])
            else:
                raise
        return data
    except Exception as e:
        st.error(f"Failed to convert assessment to JSON schema: {e}")
        return None


# --- 3d. Google APIs helpers (Forms + Gmail) ---
GOOGLE_FORMS_SCOPES = [
    "https://www.googleapis.com/auth/forms.body",
    "https://www.googleapis.com/auth/drive.file",
]
GOOGLE_GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
]


def get_google_credentials():
    try:
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
    except Exception as e:
        st.error("Google auth libraries not installed. Please install google-auth, google-auth-oauthlib, google-api-python-client.")
        return None

    creds = None
    token_path = os.getenv("GOOGLE_TOKEN_PATH", os.path.join(os.getcwd(), "token.json"))
    client_secret_path = (
        os.getenv("GOOGLE_OAUTH_CLIENT_SECRET_FILE")
        or os.path.join(os.getcwd(), ".streamlit", "google_oauth_client_secret.json")
    )

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, GOOGLE_FORMS_SCOPES + GOOGLE_GMAIL_SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(client_secret_path):
                st.error(f"Google OAuth client secret not found at: {client_secret_path}")
                return None
            flow = InstalledAppFlow.from_client_secrets_file(
                client_secret_path, GOOGLE_FORMS_SCOPES + GOOGLE_GMAIL_SCOPES
            )
            # Try local server first; fall back to console (device) flow if needed
            try:
                creds = flow.run_local_server(port=0)
            except Exception:
                creds = flow.run_console()
        # Save the credentials for next runs
        with open(token_path, "w") as token:
            token.write(creds.to_json())
    return creds


def build_google_services(creds):
    try:
        from googleapiclient.discovery import build
    except Exception:
        st.error("google-api-python-client not installed.")
        return None, None
    forms_service = build("forms", "v1", credentials=creds)
    gmail_service = build("gmail", "v1", credentials=creds)
    return forms_service, gmail_service


def create_google_form_from_schema(forms_service, schema):
    # Create the form shell
    form_title = schema.get("title") or "Auto-Generated Quiz"
    form = {"info": {"title": form_title}}
    created = forms_service.forms().create(body=form).execute()
    form_id = created["formId"]

    # Batch update: set quiz mode and add items
    requests = [
        {
            "updateSettings": {
                "settings": {"quizSettings": {"isQuiz": True}},
                "updateMask": "quizSettings.isQuiz",
            }
        }
    ]
    index = 0
    for item in schema.get("items", []):
        q_type = item.get("type")
        question = item.get("question", "Question")
        points = int(item.get("points", 1))

        if q_type == "mcq":
            choices = item.get("choices", [])
            correct = item.get("correctAnswer")
            requests.append(
                {
                    "createItem": {
                        "item": {
                            "title": question,
                            "questionItem": {
                                "question": {
                                    "required": True,
                                    "grading": {
                                        "pointValue": int(points),
                                        "correctAnswers": {"answers": [{"value": str(correct)}]} if correct else None,
                                    },
                                },
                                "choiceQuestion": {
                                    "type": "RADIO",
                                    "options": [{"value": str(c)} for c in choices],
                                },
                            },
                        },
                        "location": {"index": index},
                    }
                }
            )
        elif q_type == "short":
            correct_answers = item.get("correctAnswers", [])
            requests.append(
                {
                    "createItem": {
                        "item": {
                            "title": question,
                            "questionItem": {
                                "question": {
                                    "required": True,
                                    "grading": {
                                        "pointValue": int(points),
                                        "correctAnswers": {"answers": [{"value": str(a)} for a in correct_answers]} if correct_answers else None,
                                    },
                                },
                                "textQuestion": {"paragraph": False},
                            },
                        },
                        "location": {"index": index},
                    }
                }
            )
        else:  # paragraph/project or unknown
            requests.append(
                {
                    "createItem": {
                        "item": {
                            "title": question,
                            "questionItem": {
                                "question": {
                                    "required": False,
                                    "grading": {"pointValue": int(points)},
                                },
                                "textQuestion": {"paragraph": True},
                            },
                        },
                        "location": {"index": index},
                    }
                }
            )
        index += 1

    if requests:
        forms_service.forms().batchUpdate(formId=form_id, body={"requests": requests}).execute()

    # Get responder URL
    form_info = forms_service.forms().get(formId=form_id).execute()
    responder_uri = form_info.get("responderUri")
    return form_id, responder_uri


def send_email_with_link(gmail_service, recipients, subject, body_text):
    from email.mime.text import MIMEText
    import base64

    message = MIMEText(body_text)
    message["to"] = ", ".join(recipients)
    message["subject"] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    gmail_service.users().messages().send(userId="me", body={"raw": raw}).execute()
    return True

# --- 4. Streamlit User Interface ---
st.set_page_config(page_title="PDF ➜ Lesson Plan Generator", layout="centered")

st.title("📘 Teaching Toolkit from PDF")
st.markdown("Use the sidebar to upload content and choose between generating a lesson plan or assessments.")

# Sidebar controls
st.sidebar.header("Controls")
mode = st.sidebar.radio("Mode", ["Lesson Plan", "Assessment Generator", "AI Agent"], key="mode")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"], key="pdf_uploader")

# Placeholders for extracted text and result
extracted_text_placeholder = st.empty()
result_placeholder = st.empty()

if uploaded_file is not None:
    st.info("PDF uploaded successfully! Processing...")

    # Extract text when file is uploaded
    with st.spinner("Extracting text from PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)

    if not pdf_text:
        st.error("Failed to extract text from the PDF. Please try a different file.")
    else:
        with st.expander("Extracted Text Preview", expanded=False):
            st.text_area(
                "Full Text (truncated for preview):",
                pdf_text[:1200] + "..." if len(pdf_text) > 1200 else pdf_text,
                height=240,
                disabled=True,
            )

        if mode == "Lesson Plan":
            st.sidebar.subheader("Lesson Plan Settings")
            study_duration_weeks = st.sidebar.number_input(
                "Duration of study (weeks)",
                min_value=1,
                max_value=52,
                value=6,
                step=1,
                key="duration_weeks",
            )
            num_students = st.sidebar.number_input(
                "Number of students",
                min_value=1,
                max_value=500,
                value=25,
                step=1,
                key="num_students",
            )
            sections_per_week = st.sidebar.number_input(
                "Sections per week",
                min_value=1,
                max_value=14,
                value=2,
                step=1,
                key="sections_per_week",
            )

            if st.sidebar.button("Generate Lesson Plan", key="btn_generate_plan"):
                with st.spinner("Generating lesson plan with AI..."):
                    plan = generate_lesson_plan(pdf_text, study_duration_weeks, num_students, sections_per_week)
                result_placeholder.subheader("Lesson Plan")
                result_placeholder.markdown(plan)

        elif mode == "Assessment Generator":
            st.sidebar.subheader("Assessment Settings")
            assessment_types = st.sidebar.multiselect(
                "Assessment types",
                ["MCQ", "Short Answer", "Project"],
                default=["MCQ", "Short Answer"],
                key="assess_types",
            )
            difficulty = st.sidebar.selectbox(
                "Difficulty",
                ["Introductory", "Intermediate", "Advanced"],
                index=1,
                key="difficulty",
            )
            include_answers = st.sidebar.checkbox("Include answer key", value=True, key="include_answers")

            mcq_count = 0
            short_count = 0
            include_project = False
            project_focus = ""
            rubric_detail_level = "Detailed"

            if "MCQ" in assessment_types:
                mcq_count = st.sidebar.number_input("MCQ count", min_value=1, max_value=100, value=10, step=1, key="mcq_count")
            if "Short Answer" in assessment_types:
                short_count = st.sidebar.number_input("Short answer count", min_value=1, max_value=50, value=5, step=1, key="short_count")
            if "Project" in assessment_types:
                include_project = True
                project_focus = st.sidebar.text_input("Project topic/focus", value="Capstone applying key concepts", key="project_focus")
                rubric_detail_level = st.sidebar.selectbox("Rubric detail level", ["Brief", "Detailed"], index=1, key="rubric_level")

            if st.sidebar.button("Generate Assessment", key="btn_generate_assess"):
                with st.spinner("Generating assessments with AI..."):
                    content = generate_assessments(
                        text_content=pdf_text,
                        assessment_types=assessment_types,
                        mcq_count=mcq_count,
                        short_count=short_count,
                        include_project=include_project,
                        difficulty=difficulty,
                        include_answers=include_answers,
                        project_focus=project_focus,
                        rubric_detail_level=rubric_detail_level,
                    )
                result_placeholder.subheader("Assessments")
                result_placeholder.markdown(content)
        elif mode == "AI Agent":
            # Sidebar options for the agent
            st.sidebar.subheader("Agent Settings")
            attach_pdf = st.sidebar.checkbox("Attach PDF context to each question", value=True, key="agent_attach_pdf")
            st.sidebar.markdown("---")
            st.sidebar.subheader("Quiz Delivery (Google Form)")
            enable_delivery = st.sidebar.checkbox("Enable quiz delivery via Google Form", value=False, key="enable_delivery")
            recipient_emails_raw = st.sidebar.text_area("Student emails (comma-separated)", key="recipient_emails") if enable_delivery else ""
            email_subject = st.sidebar.text_input("Email subject", value="Your Quiz", key="email_subject") if enable_delivery else ""
            email_message = st.sidebar.text_area("Email message", value="Please complete the quiz at the link below:", key="email_message") if enable_delivery else ""
            pending_exists = bool(st.session_state.get("pending_quiz_text"))
            last_reply_exists = bool(st.session_state.get("last_assistant_reply"))
            if enable_delivery:
                st.sidebar.info("After the agent generates a quiz, review it and click Confirm & Send.")
                col_a, col_b = st.sidebar.columns(2)
                with col_a:
                    confirm_click = st.button("Confirm & Send Quiz", key="btn_confirm_send", disabled=not pending_exists)
                with col_b:
                    if st.button("Clear Pending", key="btn_clear_pending", disabled=not pending_exists):
                        st.session_state.pop("pending_quiz_text", None)
                        st.session_state.pop("pending_form_link", None)
                        st.experimental_rerun()
                # Allow marking the last assistant response as the pending quiz
                st.sidebar.button(
                    "Use last assistant response as quiz",
                    key="btn_use_last_as_quiz",
                    disabled=not last_reply_exists,
                    on_click=lambda: st.session_state.update({"pending_quiz_text": st.session_state.get("last_assistant_reply", "")}),
                )
                # Show a short preview of what will be sent
                if pending_exists:
                    st.sidebar.text_area(
                        "Pending quiz preview",
                        value=(st.session_state.get("pending_quiz_text") or "")[:800] + ("..." if len(st.session_state.get("pending_quiz_text") or "") > 800 else ""),
                        height=150,
                        disabled=True,
                    )
            if st.sidebar.button("Reset Chat", key="btn_reset_chat"):
                st.session_state.pop("agent_messages", None)

            # Initialize chat history
            if "agent_messages" not in st.session_state:
                st.session_state.agent_messages = [
                    {"role": "system", "content": "You are an AI teaching assistant. Answer clearly and concisely. Cite from the provided PDF context when relevant."}
                ]

            # Display existing chat
            for msg in st.session_state.agent_messages:
                if msg["role"] in ("user", "assistant"):
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"]) 

            # Chat input
            user_prompt = st.chat_input("Ask about the PDF, pedagogy, or create materials...")
            if user_prompt:
                st.session_state.agent_messages.append({"role": "user", "content": user_prompt})

                # Build request messages
                request_messages = list(st.session_state.agent_messages)
                if attach_pdf and pdf_text:
                    # Add/update a prior system message with context for this turn
                    request_messages.append({
                        "role": "system",
                        "content": f"PDF context (truncated):\n---\n{pdf_text[:6000]}\n---"
                    })

                with st.spinner("Thinking..."):
                    try:
                        resp = client.chat.completions.create(
                            model=model_name,
                            messages=request_messages,
                            max_tokens=800,
                            temperature=0.4,
                        )
                        assistant_reply = resp.choices[0].message.content
                    except Exception as e:
                        assistant_reply = f"Sorry, I encountered an error: {e}"

                st.session_state.agent_messages.append({"role": "assistant", "content": assistant_reply})
                st.session_state["last_assistant_reply"] = assistant_reply
                with st.chat_message("assistant"):
                    st.markdown(assistant_reply)

                # Prepare for optional delivery, but require explicit confirmation
                if enable_delivery and any(k in user_prompt.lower() for k in ["quiz", "assessment", "mcq", "short", "project"]):
                    st.session_state["pending_quiz_text"] = assistant_reply
                    st.info("Quiz prepared. Review it above. Use the sidebar to Confirm & Send.")

            # Handle confirmed sending in a separate step
            if enable_delivery and 'confirm_click' in locals() and confirm_click:
                with st.spinner("Creating Google Form and sending emails..."):
                    try:
                        # Use pending quiz if available, otherwise fallback to last assistant reply
                        quiz_source_text = st.session_state.get("pending_quiz_text") or st.session_state.get("last_assistant_reply")
                        if not quiz_source_text:
                            st.error("No quiz content available to send. Generate or mark content as quiz first.")
                            raise RuntimeError("No quiz content to send")
                        schema = convert_assessment_text_to_schema(quiz_source_text)
                        if not schema:
                            st.error("Could not convert the generated content into a quiz schema.")
                        else:
                            creds = get_google_credentials()
                            if not creds:
                                st.error("Google authentication is not configured.")
                            else:
                                forms_service, gmail_service = build_google_services(creds)
                                if not forms_service or not gmail_service:
                                    st.error("Google services failed to initialize.")
                                else:
                                    form_id, form_link = create_google_form_from_schema(forms_service, schema)
                                    st.session_state["pending_form_link"] = form_link
                                    st.success(f"Form created: {form_link}")
                                    recipients = [e.strip() for e in (recipient_emails_raw or "").split(',') if e.strip()]
                                    if recipients:
                                        send_email_with_link(
                                            gmail_service,
                                            recipients=recipients,
                                            subject=email_subject or "Your Quiz",
                                            body_text=f"{email_message or 'Please complete the quiz at the link below:'}\n\n{form_link}",
                                        )
                                        st.success(f"Email sent to: {', '.join(recipients)}")
                                    # Clear pending after sending
                                    st.session_state.pop("pending_quiz_text", None)
                    except Exception as e:
                        st.error(f"Failed to create form or send emails: {e}")

st.markdown("---")
st.caption("Upload a PDF, then use the sidebar to generate a lesson plan or assessments.")

