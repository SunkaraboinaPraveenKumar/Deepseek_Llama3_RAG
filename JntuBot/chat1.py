import os
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

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
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})
    ollama_llm = ChatOllama(model="gemma:2b")
    # ollama_llm = OllamaLLM(model="gemma:2b", streaming=True)

    print("\nChatbot is ready! Type 'exit' to quit.\n")
    while True:
        user_question = input("You: ")
        if user_question.lower() == "exit":
            print("Exiting chat. Goodbye!")
            break

        docs = retriever.invoke(user_question)  # Updated method to avoid deprecation
        context = "\n".join([doc.page_content for doc in docs])
        prompt = f"Context:\n{context}\n\nQuestion: {user_question}\nAnswer:"

        # response = ollama_llm.invoke(prompt)
        # print("Bot:", response, "\n")
        print("\nGemma: ", end="", flush=True)
        for chunk in ollama_llm.stream(prompt):
            # Extract and print only the content field
            print(chunk.content, end="", flush=True)
        print()


if __name__ == "__main__":
    chat()
