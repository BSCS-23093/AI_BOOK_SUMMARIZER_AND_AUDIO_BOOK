import streamlit as st
import google.generativeai as genai
from gtts import gTTS
import tempfile
from langchain_ollama import OllamaLLM
from PyPDF2 import PdfReader
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter


def text_to_pdf(text):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica", 12)
    y_position = 750
    for line in text.split('\n'):
        if y_position < 50:
            c.showPage()
            y_position = 750
            c.setFont("Helvetica", 12)
        c.drawString(50, y_position, line[:80])
        y_position -= 15
    c.save()
    buffer.seek(0)
    return buffer

def text_to_speech(text):
    tts = gTTS(text=text, lang='en', slow=False)
    audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(audio_file.name)
    return audio_file.name

def get_detail_level_prompt(detail_level):
    templates = {
        "Very Brief": (
            "Create an extremely concise summary of the following text in 1-2 paragraphs only. "
            "Focus exclusively on the most essential points, main ideas, and key conclusions. "
            "Omit all supporting details, examples, and secondary points.\n\n{text}"
        ),
        "Brief": (
            "Provide a brief summary of the following text in about 3-4 paragraphs. "
            "Include the main arguments and key points only. "
            "Skip minor details and examples while preserving the core message.\n\n{text}"
        ),
        "Moderate": (
            "Summarize the following text with a moderate level of detail in about 5-7 paragraphs. "
            "Include the main points along with some important supporting details. "
            "Retain the key arguments and significant examples that illustrate central concepts.\n\n{text}"
        ),
        "Detailed": (
            "Create a detailed summary of the following text in about 8-10 paragraphs. "
            "Include main arguments, supporting points, important examples, and significant findings. "
            "Preserve the logical flow and structure of the original while condensing secondary information.\n\n{text}"
        ),
        "Very Detailed": (
            "Provide a comprehensive summary of the following text. "
            "Include most main points, supporting details, examples, and evidence from the original text. "
            "Preserve the nuance and complexity of arguments while still being more concise than the original.\n\n{text}"
        )
    }
    return templates.get(detail_level)


def process_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page_num in range(len(pdf.pages)):
        page = pdf.pages[page_num]
        text += page.extract_text()
    return text

def process_epub(file):
    book = epub.read_epub(file)
    text = ""
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_body_content(), 'html.parser')
            text += soup.get_text()
    return text

def summarize_with_gemini(api_key, text, detail_level):
    client = genai.Client(api_key=api_key)
    chat = client.chats.create(model="gemini-2.0-flash")
    detail_level_user = get_detail_level_prompt(detail_level)
    response = chat.send_message_stream(f"Summarize the following text with detail level {detail_level_user}:\n\n{text}")
    summary = ""
    for chunk in response:
        summary += chunk.text
    return summary

def summarize_with_ollama(text, detail_level):
    model = OllamaLLM(model="deepseek-r1")
    detail_level_user = get_detail_level_prompt(detail_level)
    prompt = f"Summarize the following text with detail level {detail_level_user}:\n\n{text}"
    response = model.generate([prompt])
    return response[0]['text'].strip()

def main():
    st.title("Book Summarizer and Audiobook Generator")
    api_provider = st.selectbox("Select API Provider", ["Gemini", "Local LLM (Ollama)"])
    detail_level = st.selectbox("Select Detail Level", ["Very Brief", "Brief", "Moderate", "Detailed", "Very Detailed"])
    if api_provider != "Local LLM (Ollama)":
        api_key = st.text_input("Enter your API key")
    uploaded_file = st.file_uploader("Upload a Book (PDF, EPUB, or TXT)", type=["pdf", "epub", "txt"])
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            text = process_pdf(uploaded_file)
        elif uploaded_file.type == "application/epub+zip":
            text = process_epub(uploaded_file)
        else:
            text = uploaded_file.read().decode("utf-8")
        #st.write("Original Text:")
        #st.write(text[:2000] + "...")
        if st.button("Generate Summary"):
            with st.spinner('Generating summary...'):
                if api_provider == "Gemini":
                    summary = summarize_with_gemini(api_key, text, detail_level)
                elif api_provider == "Local LLM (Ollama)":
                    summary = summarize_with_ollama(text, detail_level)
            st.write("Summary:")
            st.write(summary)
            summary_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
            with open(summary_file.name, "w") as f:
                f.write(summary)
            st.download_button(
                label="Download Summary (.txt)",
                data=summary,
                file_name="summary.txt",
                mime="text/plain"
            )
            pdf_data = text_to_pdf(summary)
            st.download_button(
                label="Download Summary (.pdf)",
                data=pdf_data,
                file_name="summary.pdf",
                mime="application/pdf"
            )
            # st.download_button(
            #     label="Download Summary (.pdf)",
            #     data=summary,
            #     file_name="summary.pdf",
            #     mime="application/pdf"
            # )

            with st.spinner('Converting to speech...'):
                audio_path = text_to_speech(summary)
            st.audio(audio_path)
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
                st.download_button(
                    label="Download Audiobook (.mp3)",
                    data=audio_bytes,
                    file_name="audiobook_summary.mp3",
                    mime="audio/mp3"
                )

if __name__ == "__main__":
    main()
