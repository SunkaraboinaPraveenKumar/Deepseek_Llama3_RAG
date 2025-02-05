import os
import streamlit as st
from rag_utility_faiss import process_document_to_faiss

# Get working directory
working_dir = os.getcwd()

# File Upload Section
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    save_path = os.path.join(working_dir, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Process the uploaded document
    process_documents = process_document_to_faiss(uploaded_file.name)
    st.success("✅ Document Processed Successfully")
