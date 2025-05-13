# -*- coding: utf-8 -*-
!pip install langchain_community langchain-chroma chromadb sentence-transformers pypdf
# Install required packages - consolidated

from langchain_community.document_loaders import PyPDFLoader
import copy

# Define PDF path - update this to your local path if not using Colab
PDF_PATH = "yourpdf"

# Load PDF
my_pdf_loader = PyPDFLoader(PDF_PATH)
print(my_pdf_loader)

my_pages_cut = my_pdf_loader.load()

# Clean up whitespace in documents
for i in range(len(my_pages_cut)):
  my_pages_cut[i].page_content = " ".join(my_pages_cut[i].page_content.split())

print(f"Loaded {len(my_pages_cut)} pages from PDF")
print(f"First page preview: {my_pages_cut[0].page_content[:100]}...")

from langchain.text_splitter import RecursiveCharacterTextSplitter

# Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,       # or 800–1200 tokens
    chunk_overlap=100,     # overlap to retain coherence between chunks
    separators=["\n\n", "\n", ".", "?", "!"]  # try to split at logical sentence/paragraph boundaries
)

# Split documents into chunks
pages_split = text_splitter.split_documents(my_pages_cut)
print(f"Split into {len(pages_split)} chunks")

from langchain.embeddings import HuggingFaceEmbeddings

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    encode_kwargs={'normalize_embeddings': True}
)

from langchain_chroma import Chroma  # New version

# Create vector store
chroma_store = Chroma.from_documents(
    documents=pages_split,  # Use the split pages instead of original
    embedding=embeddings,
    persist_directory="chroma_bge_store"
)

print("✅ Chroma vector store created and persisted!")

# Example of how to query the vector store

# Define query
query = "What is the difference between analysis and analytics?"

# Stage 1: Semantic Search - retrieve top-k most relevant documents
k_semantic = 20  # Number of documents to retrieve in first stage
semantic_results = chroma_store.similarity_search(query, k=k_semantic)

print(f"\nStage 1: Retrieved {len(semantic_results)} documents using semantic search")
print(f"Query: '{query}'")

# Display first 3 semantic results (preview)
for i, doc in enumerate(semantic_results[:3]):
    print(f"\n--- Semantic Result {i+1}/{k_semantic} ---\n{doc.page_content[:150]}...")

# Stage 2: Apply MMR to reduce redundancy from the semantic results
# Create a temporary Chroma store with just the semantic results
from langchain_core.documents import Document

# Extract the content and metadata from semantic results
semantic_docs = [
    Document(page_content=doc.page_content, metadata=doc.metadata)
    for doc in semantic_results
]

# Create a temporary vector store with just these documents
temp_vectorstore = Chroma.from_documents(
    documents=semantic_docs,
    embedding=embeddings,
    persist_directory="temp_mmr_store"
)

# Apply MMR on this smaller set
k_final = 5  # Final number of documents after MMR
mmr_results = temp_vectorstore.max_marginal_relevance_search(
    query,
    k=k_final,
    fetch_k=k_semantic,  # Consider all semantic results
    lambda_mult=0.7  # Balance between relevance (1.0) and diversity (0.0)
)

print(f"\nStage 2: Selected {len(mmr_results)} diverse documents using MMR")

# Display final results after MMR
for i, doc in enumerate(mmr_results):
    print(f"\n--- Final Result {i+1}/{k_final} ---\n{doc.page_content}")

# Clean up temporary vector store
import shutil
try:
    shutil.rmtree("temp_mmr_store")
    print("\nTemporary vector store cleaned up")
except:
    pass

# ... rest of your code ...

# --- Add LLM integration for generating responses ---
# ... existing code ...

# --- Add LLM integration for generating responses ---
import requests
import json
import os

# Set your OpenRouter API key
OPENROUTER_API_KEY = ""  # Replace with your actual API key

def generate_rag_response(query, k_semantic=10, k_final=3):
    print(f"\n📝 Generating response for: '{query}'")

    # Get relevant documents using MMR
    mmr_results = chroma_store.max_marginal_relevance_search(
        query,
        k=k_final,
        fetch_k=k_semantic,
        lambda_mult=0.7
    )

    # Create context from documents
    context = "\n\n".join([doc.page_content for doc in mmr_results])

    # Create prompt for OpenRouter
    prompt = f"""Answer the following question using only the provided context. If you cannot answer from the context, say "I cannot answer this from the provided information."

Context: {context}

Question: {query}"""

    # Call OpenRouter API
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer APIKEY",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8000",  # Update with your site URL
                "X-Title": "RAG-PDF-App",  # Update with your app name
            },
            data=json.dumps({
                "model": "microsoft/phi-4-reasoning-plus:free",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based only on the provided context."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
            })
        )

        # Parse the response
        response_json = response.json()
        answer = response_json['choices'][0]['message']['content']

        # Display results
        print("\n🤖 Generated Response:")
        print(answer)

        print("\n📚 Sources Used:")
        for i, doc in enumerate(mmr_results):
            print(f"\nSource {i+1}:")
            print(f"{doc.page_content[:200]}...")

        return answer

    except Exception as e:
        print(f"Error in response generation: {str(e)}")
        return "Sorry, I encountered an error while generating the response."

# ... rest of the code remains the same ...

# ... existing code ...

# Example usage
if __name__ == "__main__":
    while True:
        # Get user input
        user_query = input("\nEnter your question (or 'quit' to exit): ")

        # Check if user wants to quit
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        # Generate response
        try:
            generate_rag_response(user_query)
        except Exception as e:
            print(f"Error generating response: {e}")
