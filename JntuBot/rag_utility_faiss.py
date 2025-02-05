import os
import json
import faiss
import numpy as np
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

working_dir = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))
GROQ_API_KEY = config_data["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

embedding = HuggingFaceEmbeddings()

def process_document_to_faiss(file_name):
    persist_directory = f"{working_dir}/doc_vectorstore_faiss"

    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    # Load the document and split it into chunks
    loader = UnstructuredPDFLoader(f"{working_dir}/{file_name}")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Get embeddings for the texts
    embeddings = embedding.embed_documents([doc.page_content for doc in texts])

    # Convert embeddings to numpy array for FAISS
    vectors = np.array(embeddings).astype(np.float32)

    # Create FAISS index and add vectors
    faiss_index = faiss.IndexFlatL2(vectors.shape[1])  # L2 distance index
    faiss_index.add(vectors)

    # Save the FAISS index
    faiss_index_file = os.path.join(persist_directory, "faiss.index")
    faiss.write_index(faiss_index, faiss_index_file)

    # Save the document texts for later retrieval
    document_texts = [doc.page_content for doc in texts]
    with open(os.path.join(persist_directory, "documents.json"), "w") as f:
        json.dump(document_texts, f)

    return "Document processed and vectors added to FAISS."



def load_faiss_index():
    persist_directory = f"{working_dir}/doc_vectorstore_faiss"
    faiss_index_file = os.path.join(persist_directory, "faiss.index")

    if os.path.exists(faiss_index_file):
        faiss_index = faiss.read_index(faiss_index_file)
        return faiss_index
    else:
        return None


def retrieve_from_faiss(user_question):
    faiss_index = load_faiss_index()

    if faiss_index is None:
        return "No FAISS index found. Please process documents first."

    # Embed the user's question
    question_embedding = embedding.embed([user_question])
    question_vector = np.array(question_embedding).astype(np.float32)

    # Search for the top 2 closest documents in FAISS
    D, I = faiss_index.search(question_vector, k=2)  # k=2 for top 2 documents

    # Retrieve the documents and their content
    retrieved_docs = []
    for i in I[0]:
        retrieved_docs.append(f"Document {i} content here...")  # Update with actual document content retrieval logic

    return "\n".join(retrieved_docs)