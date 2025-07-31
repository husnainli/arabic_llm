from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os

def chunk_text(text, chunk_size=500, overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n", ".", "،", " "]
    )
    return splitter.split_text(text)

# def embed_chunks(chunks, persist_dir="chroma_db", model_name='intfloat/multilingual-e5-base'):
#     embedding = HuggingFaceEmbeddings(model_name=model_name)

#     # Ensure reproducible index
#     if os.path.exists(persist_dir):
#         vectorstore = Chroma(embedding_function=embedding)
#     else:
#         vectorstore = Chroma.from_texts(chunks, embedding=embedding, persist_directory=persist_dir)
#         vectorstore.persist()

#     return vectorstore

def embed_chunks(chunks, model_name='intfloat/multilingual-e5-base'):
    embedding = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_texts(chunks, embedding)
    return vectorstore



def retrieve_similar_chunks(vectorstore, query, k=4):
    return vectorstore.similarity_search(query, k=k)
