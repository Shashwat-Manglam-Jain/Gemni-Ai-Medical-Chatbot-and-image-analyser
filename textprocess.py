from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os

# =====================================================
# 1 Load documents (supports both .txt and .pdf)
# =====================================================

# Define your data file (you can change or loop through multiple later)
file_path = r"static\medical_treatments.txt"

# Detect file type automatically
if file_path.lower().endswith(".pdf"):
    loader = PyPDFLoader(file_path)
elif file_path.lower().endswith(".txt"):
    loader = TextLoader(file_path, encoding="utf-8")
else:
    raise ValueError("Unsupported file type. Please use .pdf or .txt")

# Load and split text into documents
docs = loader.load()

# =====================================================
# 2 Chunk the documents for better retrieval
# =====================================================
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# =====================================================
# 3 Create embeddings
# =====================================================
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# =====================================================
# 4 Build FAISS index and save it
# =====================================================
db = FAISS.from_documents(chunks, emb)
db.save_local("faiss_index1")

print("✅ FAISS index built and saved successfully as 'faiss_index1'")
