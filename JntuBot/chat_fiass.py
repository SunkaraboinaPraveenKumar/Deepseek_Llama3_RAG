# ollama_gemma_faiss.py
import os
import json
import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# Load configuration
working_dir = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))
os.environ["OLLAMA_HOST"] = "http://localhost:11434"  # Ensure Ollama is running locally

# Initialize embedding model
embedding = HuggingFaceEmbeddings()


def load_documents():
    # Load the documents from the JSON file
    faiss_cpu_dir = f"{working_dir}/doc_vectorstore_faiss"
    documents_file = os.path.join(faiss_cpu_dir, "documents.json")

    if os.path.exists(documents_file):
        with open(documents_file, "r") as file:
            documents = json.load(file)
        return documents
    else:
        raise FileNotFoundError("documents.json not found. Ensure the file is available.")


# Load FAISS index
def load_faiss_index():
    persist_directory = f"{working_dir}/doc_vectorstore_faiss"
    faiss_index_file = os.path.join(persist_directory, "faiss.index")

    if os.path.exists(faiss_index_file):
        faiss_index = faiss.read_index(faiss_index_file)
        return faiss_index
    else:
        raise FileNotFoundError("FAISS index not found. Please process documents first.")


# Retrieve relevant documents from FAISS
def retrieve_from_faiss(user_question, k=2):
    faiss_index = load_faiss_index()

    # Embed the user's question
    question_embedding = embedding.embed_documents([user_question])
    question_vector = np.array(question_embedding).astype(np.float32)

    # Search for the top-k closest documents in FAISS
    D, I = faiss_index.search(question_vector, k=1)

    # Retrieve the documents (replace this with actual document retrieval logic)
    retrieved_docs = []
    for i in I[0]:
        # Assuming you have a list or database where documents are stored
        # Replace this line with actual code to fetch the document content
        document_content = get_document_content(i)  # This function should fetch the actual document content
        retrieved_docs.append(document_content)

    return retrieved_docs


def get_document_content(index):
    documents = load_documents()
    return documents[index]


# Initialize Ollama Gemma:2b model
ollama_llm = ChatOllama(model="gemma:2b")


# Chat stream with Ollama and FAISS context
# Chat stream with Ollama and FAISS context
def chat_with_gemma():
    print("Welcome to the Ollama Gemma:2b chat! Type 'exit' to quit.")
    faiss_index = load_faiss_index()

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Retrieve relevant documents from FAISS
        context_docs = retrieve_from_faiss(user_input)
        context = "\n".join(context_docs)
        # print(context)

        # Prepare the prompt with context
        prompt = (
            f"Given the following context:\n{context}"
            f"\n\nYour task is to answer the question based on the provided context. "
            f"Be Clear and Explanatory for about four to five lines."
            f"\n\nQuestion: {user_input}\nAnswer:"
        )

        # Stream the response from Ollama
        print("\nGemma: ", end="", flush=True)
        for chunk in ollama_llm.stream(prompt):
            # Extract and print only the content field
            print(chunk.content, end="", flush=True)
        print()


if __name__ == "__main__":
    chat_with_gemma()


