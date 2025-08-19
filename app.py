import streamlit as st
import openai
import PyPDF2
import io

# --- 1. Load OpenAI API Key from Streamlit Secrets ---
# Make sure you have a .streamlit/secrets.toml file with OPENAI_API_KEY="your_api_key_here"
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("OpenAI API key not found. Please set it in .streamlit/secrets.toml")
    st.stop() # Stop the app if API key is not found

# --- 2. Function to Extract Text from PDF ---
def extract_text_from_pdf(uploaded_file):
    """
    Extracts text content from an uploaded PDF file.
    Returns the extracted text as a string, or None if an error occurs.
    """
    if uploaded_file is not None:
        try:
            # PyPDF2 expects a file-like object, io.BytesIO makes the uploaded file readable like one
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            text = ""
            # Iterate through each page and extract text
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text: # Ensure text is not None
                    text += page_text
            return text
        except PyPDF2.utils.PdfReadError:
            st.error("Invalid PDF file. Please upload a valid, unencrypted PDF.")
            return None
        except Exception as e:
            st.error(f"An error occurred while reading the PDF: {e}")
            return None
    return None

# --- 3. Function to Get LLM Summary ---
def get_llm_summary(text_content):
    """
    Sends the extracted text content to OpenAI's GPT-3.5-turbo for a summary.
    """
    if not text_content:
        return "No text provided for summarization."

    prompt = f"""Summarize the following text briefly and concisely.
    
    Text:
    ---
    {text_content[:3000]} # Limit text length to fit within typical LLM context window for a quick test
    ---
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", # A good general-purpose model for summarization
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200, # Keep the summary short for this test
            temperature=0.3 # Lower temperature for more factual and less creative summary
        )
        return response['choices'][0]['message']['content']
    except openai.error.OpenAIError as e:
        st.error(f"OpenAI API Error: {e}. Check your API key and network connection.")
        return "Failed to get summary due to an API error."
    except Exception as e:
        st.error(f"An unexpected error occurred with the LLM call: {e}")
        return "Failed to get summary due to an unexpected error."


# --- 4. Streamlit User Interface ---
st.set_page_config(page_title="PDF Reader & LLM Summarizer", layout="centered")

st.title("📄 PDF Reader & AI Summarizer")
st.markdown("Upload a PDF to extract its text and get a quick summary from an AI.")

# File uploader widget
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], key="pdf_uploader")

# Placeholder for extracted text and summary
extracted_text_placeholder = st.empty()
llm_summary_placeholder = st.empty()

if uploaded_file is not None:
    st.info("PDF uploaded successfully! Processing...")

    # Extract text when file is uploaded
    with st.spinner("Extracting text from PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
    
    if pdf_text:
        extracted_text_placeholder.subheader("Extracted Text Preview:")
        # Display a truncated version of the extracted text for preview
        extracted_text_placeholder.text_area("Full Text (truncated for preview):", pdf_text[:1000] + "..." if len(pdf_text) > 1000 else pdf_text, height=300, disabled=True)

        # Button to trigger LLM summarization
        if st.button("Summarize Text with AI", use_container_width=True, key="summarize_button"):
            with st.spinner("Asking AI to summarize..."):
                summary = get_llm_summary(pdf_text)
            
            llm_summary_placeholder.subheader("AI Summary:")
            llm_summary_placeholder.write(summary)
    else:
        st.error("Failed to extract text from the PDF. Please try a different file.")

st.markdown("---")
st.caption("This is a step-by-step test for PDF reading and basic LLM integration.")

