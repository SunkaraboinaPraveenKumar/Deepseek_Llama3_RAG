import os
import streamlit as st
from rag_utility import process_document_to_chromadb, answer_question

# Get working directory
working_dir = os.getcwd()

# Streamlit UI Title
st.title("🐋 DeepSeek-R1 vs 🦙 Llama-3")

# File Upload Section
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    save_path = os.path.join(working_dir, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Process the uploaded document
    process_documents = process_document_to_chromadb(uploaded_file.name)
    st.success("✅ Document Processed Successfully")

# User Input Section
user_question = st.text_area("Ask your question from the document")

# Answer Generation
if st.button("Answer"):
    answer = answer_question(user_question)

    # Get responses
    answer_deepseek = answer.get("answer_deepseek", "No response generated.")
    answer_llama3 = answer.get("answer_llama3", "No response generated.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### DeepSeek-r1 Response")
        st.markdown(
            f"""
            <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #222; color: white;">
                {answer_deepseek}
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown("### Llama-3 Response")
        st.markdown(
            f"""
            <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #222; color: white;">
                {answer_llama3}
            </div>
            """,
            unsafe_allow_html=True
        )
