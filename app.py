import streamlit as st
import openai
import PyPDF2
import io # To handle file-like objects for PDF reading

# Set your API key from Streamlit secrets
# Make sure you have a .streamlit/secrets.toml file with OPENAI_API_KEY="your_api_key_here"
openai.api_key = st.secrets["OPENAI_API_KEY"]

def extract_text_from_pdf(uploaded_file):
    """
    Extracts text content from an uploaded PDF file.
    """
    if uploaded_file is not None:
        try:
            # Create a PdfReader object
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or "" # Handle potential None if page has no extractable text
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {e}. Please ensure it's a valid PDF and not password-protected.")
            return None
    return None

def generate_lesson_plan(topic, grade, duration, num_students, chapters_info, pdf_content=None):
    """
    Generates a structured lesson plan using a large language model,
    incorporating PDF content, number of students, and chapter details.
    """
    prompt = f"""
    You are an AI assistant specialized in creating detailed, engaging, and structured lesson plans.

    **Objective:** Create a comprehensive lesson plan for a {grade} grade class.

    **Core Information:**
    - **Topic:** "{topic}"
    - **Duration:** Approximately {duration} minutes
    - **Number of Students:** {num_students} students

    **Structure and Content Requirements:**

    The lesson plan MUST be structured using the following headings and sub-sections, adapted to the provided information:

    ## 1. Introduction & Hook (approx. 5-10 minutes)
    - A brief, engaging activity or question to grab students' attention, considering {num_students} students.

    ## 2. Main Lesson Content (structured by chapters)
    - Break down the main content into sections based on the following chapter information:
    """
    
    if chapters_info:
        for i, chapter in enumerate(chapters_info.splitlines()):
            if chapter.strip(): # Add only non-empty lines as chapters
                prompt += f"- **Chapter {i+1}: {chapter.strip()}**\n"
                prompt += "  - Detailed explanation of key concepts for this chapter.\n"
                prompt += "  - Relevant examples and demonstrations for this chapter.\n"
                prompt += "  - An interactive activity or group work suggestion for this chapter, suitable for the class size.\n"
    else:
        prompt += "- **Main Concepts:** Detailed explanation of key concepts, examples, and an interactive activity or group work.\n"
    
    prompt += """
    ## 3. Wrap-Up & Assessment
    - A summary of the main points covered.
    - A quick assessment (e.g., a short quiz, an exit ticket, or a discussion prompt) to check understanding.
    - Suggested homework or follow-up activity.

    """
    
    if pdf_content:
        prompt += f"""
    **Reference Material (from uploaded PDF):**
    Please draw information and specific details primarily from the following text to inform the lesson plan content:
    ---
    {pdf_content[:2500]} # Limiting PDF content to fit within typical LLM context window
    ---
    If the PDF content is very long, focus on the most relevant information for the topic and chapters provided.
    """
    else:
        prompt += """
    **Note:** No PDF content was provided, so generate the lesson plan based on general knowledge of the topic.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", # You can use "gpt-4" or other models for better quality
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant specialized in creating structured lesson plans for educators."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000, # Increased max_tokens to allow for longer lesson plans
            temperature=0.7 # Controls creativity vs. predictability
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"An error occurred while generating the lesson plan: {e}. Please try again.")
        st.info("Tip: If the error persists, the PDF might be too large or the prompt too complex for the current AI model. Try a shorter PDF or a simpler prompt.")
        return None

# --- Streamlit App Interface ---

st.set_page_config(page_title="💡 AI-Powered Lesson Plan Generator", layout="centered")

st.title("💡 AI-Powered Lesson Plan Generator")
st.markdown("Effortlessly create structured lesson plans by providing a topic, grade, duration, student count, and even source material from a PDF or specific chapter details.")

st.sidebar.header("Settings")
with st.sidebar:
    st.markdown("---")
    st.subheader("API Key")
    st.info("API key loaded from `.streamlit/secrets.toml`")
    st.markdown("---")


st.header("Lesson Details Input")

uploaded_file = st.file_uploader("Upload a PDF with Lesson Content (Optional)", type=["pdf"], key="pdf_uploader")
topic = st.text_input("📚 Topic:", placeholder="e.g., Photosynthesis, The American Revolution", key="topic_input")
grade = st.selectbox("🎓 Grade Level:", ["Kindergarten", "1st Grade", "2nd Grade", "3rd Grade", "4th Grade", 
                                        "5th Grade", "6th Grade", "7th Grade", "8th Grade", "9th Grade", 
                                        "10th Grade", "11th Grade", "12th Grade", "College Level"], key="grade_select")
duration = st.slider("⏰ Lesson Duration (in minutes):", min_value=15, max_value=120, step=5, value=45, key="duration_slider")
num_students = st.number_input("🧑‍🎓 Number of Students:", min_value=1, value=25, step=1, key="num_students_input")
chapters_info = st.text_area("✍️ Chapter/Section Details (Optional - one per line):", 
                             placeholder="e.g.,\nIntroduction to Cells\nCell Organelles\nCell Division", 
                             height=150, key="chapters_input")


st.markdown("---")
# Button to generate the lesson plan
if st.button("✨ Generate Lesson Plan", use_container_width=True, key="generate_button"):
    if not topic:
        st.error("❗ Please enter a **Topic** to generate a lesson plan.")
    else:
        pdf_text = None
        if uploaded_file:
            with st.spinner("Extracting text from PDF..."):
                pdf_text = extract_text_from_pdf(uploaded_file)
            if not pdf_text:
                st.warning("Could not extract text from the PDF. Generating lesson plan based on other inputs only.")
        
        if pdf_text and len(pdf_text) > 5000: # Simple check for very long PDFs
            st.warning("The extracted text from the PDF is very long. The AI will only use the beginning part to fit its context window.")
            # Truncate text for LLM to avoid context window issues
            pdf_text = pdf_text[:5000] # Adjust this limit based on your LLM's context window

        with st.spinner("Crafting your personalized lesson plan..."):
            lesson_plan_content = generate_lesson_plan(topic, grade, duration, num_students, chapters_info, pdf_text)
            
            if lesson_plan_content:
                st.markdown("---")
                st.subheader("Your Generated Lesson Plan")
                st.markdown(lesson_plan_content) # Render the markdown output from the LLM
            else:
                st.error("Failed to generate a lesson plan. Please check your inputs and try again.")

st.markdown("---")
st.caption("Powered by AI 🚀 | Ensure your API key is correctly set in `.streamlit/secrets.toml`")
