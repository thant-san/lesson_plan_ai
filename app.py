import streamlit as st
import openai
from openai import OpenAI
import PyPDF2
import io

# --- 1. Load OpenAI API Key from Streamlit Secrets ---
# Make sure you have a .streamlit/secrets.toml file with OPENAI_API_KEY="your_api_key_here"
try:
    client = OpenAI(base_url="https://api.aimlapi.com/v1",api_key=st.secrets["OPENAI_API_KEY"])
except KeyError:
    st.error("OpenAI API key not found. Please set it in .streamlit/secrets.toml")
    st.stop()

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
def generate_lesson_plan(text_content, study_duration_weeks, num_students):
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
- Source content (use to derive objectives, topics, examples, and assessments):
---
{text_content[:6000]}
---

Requirements:
1) Provide an overview with goals and success criteria tailored to the class size.
2) Break down the plan by week with clear learning objectives, key topics, and vocabulary.
3) For each week include:
   - Activities (at least one teacher-led, one student-centered, one collaborative activity)
   - Materials/resources
   - Differentiation for mixed abilities and larger group management if applicable
   - Assessment (formative + one summative suggestion across the duration)
4) Add homework/extension ideas and optional enrichment.
5) Keep it concise and actionable. Use markdown headings and bullet points.
"""

    try:
        response = client.chat.completions.create(
            model="openai/gpt-5-chat-latest",
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


# --- 4. Streamlit User Interface ---
st.set_page_config(page_title="PDF ➜ Lesson Plan Generator", layout="centered")

st.title("📘 Lesson Plan Generator from PDF")
st.markdown("Upload a curriculum or reading PDF, enter study duration and class size, then generate a structured lesson plan.")

# File uploader widget
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], key="pdf_uploader")

# Placeholders for extracted text and result
extracted_text_placeholder = st.empty()
lesson_plan_placeholder = st.empty()

if uploaded_file is not None:
    st.info("PDF uploaded successfully! Processing...")

    # Extract text when file is uploaded
    with st.spinner("Extracting text from PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
    
    if pdf_text:
        extracted_text_placeholder.subheader("Extracted Text Preview")
        extracted_text_placeholder.text_area(
            "Full Text (truncated for preview):",
            pdf_text[:1200] + "..." if len(pdf_text) > 1200 else pdf_text,
            height=260,
            disabled=True,
        )

        st.markdown("### Inputs")
        col1, col2 = st.columns(2)
        with col1:
            study_duration_weeks = st.number_input(
                "Duration of study (weeks)",
                min_value=1,
                max_value=52,
                value=6,
                step=1,
                key="duration_weeks",
            )
        with col2:
            num_students = st.number_input(
                "Number of students",
                min_value=1,
                max_value=500,
                value=25,
                step=1,
                key="num_students",
            )

        if st.button("Generate Lesson Plan", use_container_width=True, key="generate_plan_button"):
            with st.spinner("Generating lesson plan with AI..."):
                plan = generate_lesson_plan(pdf_text, study_duration_weeks, num_students)
            lesson_plan_placeholder.subheader("Lesson Plan")
            lesson_plan_placeholder.markdown(plan)
    else:
        st.error("Failed to extract text from the PDF. Please try a different file.")

st.markdown("---")
st.caption("Upload a PDF, set duration and class size, then generate a tailored lesson plan.")

