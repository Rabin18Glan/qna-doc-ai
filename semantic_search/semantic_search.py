import os
import tempfile
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    WebBaseLoader,
    YoutubeLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def get_loader(file_path=None, file_type=None, url=None):
    if url:
        if "youtube.com" in url or "youtu.be" in url:
            return YoutubeLoader.from_youtube_url(url, add_video_info=True)
        else:
            return WebBaseLoader(url)
    elif file_path and file_type:
        if file_type == ".pdf":
            return PyPDFLoader(file_path)
        elif file_type == ".txt":
            return TextLoader(file_path)
        elif file_type in [".doc", ".docx"]:
            return UnstructuredWordDocumentLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    else:
        raise ValueError("No valid input provided to determine loader.")

def run(query, uploaded_files=None, urls=None):
    if not uploaded_files and not urls:
        return "Please upload files or provide URLs."

    all_documents = []

    # Process uploaded files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            uploaded_file.seek(0)  # Ensure the file can be read
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            try:
                loader = get_loader(file_path=tmp_path, file_type=file_ext)
                all_documents.extend(loader.load())
            finally:
                os.remove(tmp_path)  # This should run per file, not after the loop

    # Process URLs
    if urls:
        for url in urls:
            try:
                loader = get_loader(url=url)
                all_documents.extend(loader.load())
            except Exception as e:
                print(f"Failed to load {url}: {e}")

    if not all_documents:
        return "No content could be loaded from provided files or URLs."

    # Split, embed, and search
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(all_documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(all_splits, embeddings)
    results = db.similarity_search(query, k=3)

    return results[0].page_content if results else "No results found."
