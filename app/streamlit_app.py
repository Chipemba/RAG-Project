
import streamlit as st  
from functions import *
import random
import base64

# Set session state defaults
# ---------- Session defaults ----------
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = False


def build_vector_store():
    """
    Load docs and build the vector store. Sets session state.
    """
    documents = get_pdf_text()
    st.session_state.vector_store = create_vectorstore_from_texts(documents, api_key = st.session_state.api_key, filename="retrievedData")
    st.session_state.docs_loaded = True


def getAnswer(question):

    answer = query_document(vectorstore = st.session_state.vector_store, 
                                query = question,
                                api_key = st.session_state.api_key
                                )  
    return answer


# ---------- Page layout ----------
st.set_page_config(layout="wide", page_title="MotorFanz FAQ")
st.markdown("<h1 style='text-align: center;'>MotorFanz FAQ</h1>", unsafe_allow_html=True)

st.markdown("<p>Input your OpenAI API key</p>", unsafe_allow_html=True)
st.text_input(
    "OpenAI API key",
    type="password",
    key="api_key",
    label_visibility="collapsed",
    disabled=False
)

canBuild = bool(st.session_state.api_key)
buildClicked = st.button("Build Knowledge Base", disabled=not canBuild)

if buildClicked:
    if not st.session_state.api_key:
        st.error("Please provide your OpenAI API key.")
    else:
        with st.spinner("Indexing documents and building vector store..."):
            try:
                # Use the first file name as a hint, or default
                # fname_hint = uploaded_files[0].name if uploaded_files else "retrievedData"
                build_vector_store()
                st.success("Knowledge base is ready!")
            except Exception as e:
                st.session_state.vector_store = None
                st.session_state.docs_loaded = False
                st.error(f"Failed to build knowledge base: {e}")

col1, col2 = st.columns([0.6, 0.4], gap="small")


# Ppopulate the left column with questions from the questions.txt file and answer them with the RAG
with col1:

    st.markdown("<h2 style='text-align: center; text-decoration:underline;'>Frequently Asked Questions</h2>", unsafe_allow_html=True)
    questions= [
        (1, "For a beginner, is a 2-stroke or 4-stroke dirt bike better?"),
        (2, "What do engine “cc’s” mean on a dirt bike?"),
        (3, "What type of dirt bike is best for riding through forests?"),
        (4, "How much should I spend on my first dirt bike?"),
        (5, "How young can someone start riding a dirt bike?")
    ]

    faq_disabled = st.session_state.vector_store is None
    if faq_disabled:
        st.caption("Provide an API Key to build the knowledge base to enable FAQ answers.")

    for key, question in questions:
        if st.button(question, key=f"faq_{key}", disabled=faq_disabled):
            with st.spinner("Thinking..."):
                try:
                    answer = getAnswer(question)
                    st.write(answer)
                except Exception as e:
                    st.error(f"Could not fetch answer: {e}")


# Process the input
# if uploaded_file is not None:
with col2:
    st.markdown("<h2 style='text-align: center; text-decoration:underline;'>Chat with Motory</h2>", unsafe_allow_html=True)

