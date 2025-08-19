import streamlit as st
import openai # Or the library for your chosen LLM, e.g., google-generativeai

# Set your API key from Streamlit secrets
# Make sure you have a .streamlit/secrets.toml file with OPENAI_API_KEY="your_api_key_here"
openai.api_key = st.secrets["OPENAI_API_KEY"]

def generate_lesson_plan(topic, grade, duration):
    """
    Generates a structured lesson plan using a large language model.
    """
    prompt = f"""
    You are an AI assistant specialized in creating detailed and engaging lesson plans.
    Create a lesson plan for a {grade} grade class on the topic of "{topic}".
    The lesson should be approximately {duration} minutes long.

    The plan must be structured as follows:

    ## 1. Introduction & Hook (approx. 5-10 minutes)
    - A brief, engaging activity or question to grab students' attention.

    ## 2. Main Lesson Content
    - Detailed explanation of the key concepts.
    - Examples and demonstrations.
    - An interactive activity or group work.

    ## 3. Wrap-Up & Assessment
    - A summary of the main points.
    - A quick assessment, like a short quiz or a discussion prompt.
    - Suggested homework or follow-up activity.
    """
    try:
        # For Google Gemini, you'd replace this with the appropriate API call:
        # from google.generativeai.client import get_default_retriever_async_client
        # client = get_default_retriever_async_client()
        # response = await client.generate_content(
        #    model="gemini-pro", # Or another suitable Gemini model
        #    contents=[{"role": "user", "parts": [{"text": prompt}]}]
        # )
        # return response.text

        # For OpenAI (as used here):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", # Consider "gpt-4" for better quality if available
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000, # Adjust as needed for lesson plan length
            temperature=0.7 # Controls creativity vs. predictability (0.0-1.0)
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"An error occurred: {e}"

# --- Streamlit App Interface ---

st.set_page_config(page_title="AI Lesson Plan Generator", layout="centered")

st.title("💡 AI Lesson Plan Generator")
st.markdown("Effortlessly create structured lesson plans for any topic and grade level.")

st.sidebar.header("Settings")
with st.sidebar:
    st.markdown("---")
    st.subheader("API Key")
    # In a real deployment, use Streamlit Cloud secrets or environment variables
    # instead of direct input for production.
    # For local testing, ensure secrets.toml is set up as instructed.
    st.info("API key loaded from `.streamlit/secrets.toml`")
    st.markdown("---")


st.header("Lesson Details")
topic = st.text_input("📚 Topic:", placeholder="e.g., Photosynthesis, The American Revolution, Fractions", key="topic_input")
grade = st.selectbox("🎓 Grade Level:", ["Kindergarten", "1st Grade", "2nd Grade", "3rd Grade", "4th Grade", 
                                        "5th Grade", "6th Grade", "7th Grade", "8th Grade", "9th Grade", 
                                        "10th Grade", "11th Grade", "12th Grade", "College Level"], key="grade_select")
duration = st.slider("⏰ Lesson Duration (in minutes):", min_value=15, max_value=120, step=5, value=45, key="duration_slider")

# Button to generate the lesson plan
st.markdown("---")
if st.button("✨ Generate Lesson Plan", use_container_width=True, key="generate_button"):
    if topic:
        with st.spinner("Crafting your perfect lesson plan..."):
            lesson_plan = generate_lesson_plan(topic, grade, duration)
            
            st.markdown("---")
            st.subheader("Your Generated Lesson Plan")
            st.markdown(lesson_plan) # Use st.markdown to render the markdown output from the LLM
    else:
        st.error("❗ Please enter a topic to generate a lesson plan.")

st.markdown("---")
st.caption("Powered by AI 🚀")
