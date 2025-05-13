
# 🧠 RAG-Powered PDF Q&A System

This project is a Retrieval-Augmented Generation (RAG) pipeline that allows you to query a PDF document intelligently using semantic search, Maximal Marginal Relevance (MMR), and an LLM (Large Language Model) via the OpenRouter API.

## 🚀 Features

- Load and preprocess a PDF using `langchain_community`
- Split documents into chunks for vectorization
- Create vector embeddings using `BAAI/bge-large-en-v1.5`
- Store embeddings in a persistent Chroma vector store
- Perform semantic search and apply MMR for diverse, relevant retrieval
- Generate contextual answers using OpenRouter LLM API

## 📂 Project Structure

- `untitled22.py`: The main Python script for RAG pipeline
- `README.md`: This file

## 🛠️ Installation

```bash
pip install langchain_community langchain-chroma chromadb sentence-transformers pypdf python-dotenv
```

## 📄 Usage

1. Place your target PDF in the working directory.
2. Update the PDF path in the script:
   ```python
   PDF_PATH = "/content/your_pdf_file.pdf"
   ```
3. Set up your `.env` file (create this in the root folder):

```
OPENROUTER_API_KEY=your-api-key-here
```

4. Make sure `.env` is added to `.gitignore` to protect your API key.

5. Run the script:
```bash
python untitled22.py
```

6. Enter your query when prompted.

## 📌 Sample Query

```text
Enter your question (or 'quit' to exit): What is the difference between analysis and analytics?
```

## 🔐 Environment & Security

Never expose your API key publicly. Always store it using environment variables or in a `.env` file. This project expects the key as:
```python
os.getenv("OPENROUTER_API_KEY")
```

## 📚 Dependencies

- `langchain_community`
- `langchain-chroma`
- `chromadb`
- `sentence-transformers`
- `pypdf`
- `requests`
- `python-dotenv` (recommended for loading API keys)

## 📸 Screenshots / Demo (Optional)

Add a GIF or screenshot here to show your app in action!

## 🧠 Model Used

- Embeddings: `BAAI/bge-large-en-v1.5`
- LLM: `microsoft/phi-4-reasoning-plus:free` via OpenRouter

## 📥 Citation

Original notebook inspired by LangChain examples.

---

🛠 Built with love and curiosity. Happy building!
