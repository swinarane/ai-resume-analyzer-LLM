import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import pdfplumber

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)


# Extract text from PDF
def extract_resume_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Analyze resume with LLM
def analyze_resume(resume_text, job_desc):
    prompt = f"""
You are an AI Resume Analyzer.

Resume:
{resume_text}

Job Description:
{job_desc}

Tasks:
1. Give skill match percentage
2. List missing skills
3. Give short improvement suggestions

Respond in clear bullet points.
"""
    response = llm.invoke(prompt)
    return response.content

# ---------------- STREAMLIT UI ---------------- #

st.set_page_config(page_title="AI Resume Analyzer", layout="centered")

st.title("📄 AI Resume Analyzer & Job Matcher")

resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_desc = st.text_area("Paste Job Description")

if st.button("Analyze Resume"):
    if resume_file and job_desc:
        with st.spinner("Analyzing resume..."):
            resume_text = extract_resume_text(resume_file)
            result = analyze_resume(resume_text, job_desc)
            st.success("Analysis Complete")
            st.markdown(result)
    else:
        st.warning("Please upload resume and paste job description.")

