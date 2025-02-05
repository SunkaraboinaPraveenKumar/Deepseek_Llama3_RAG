import os
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM

working_dir = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))
os.environ["OLLAMA_HOST"] = "http://localhost:11434"

embedding = HuggingFaceEmbeddings()


def load_embeddings():
    print("Loading embeddings...")
    persist_directory = f"{working_dir}/doc_vectorstore"
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    return vectordb


def chat():
    vectordb = load_embeddings()
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    ollama_llm = OllamaLLM(model="gemma:2b", streaming=True)

    qa_chain = RetrievalQA.from_chain_type(
        llm=ollama_llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    print("\nChatbot is ready! Type 'exit' to quit.\n")
    while True:
        user_question = input("You: ")
        if user_question.lower() == "exit":
            print("Exiting chat. Goodbye!")
            break

        response = qa_chain.invoke({"query": user_question})
        print("Bot:", response["result"], "\n")


if __name__ == "__main__":
    chat()