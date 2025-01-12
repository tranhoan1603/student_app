from typing import Union
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

class VectorDB:
    def __init__(self, documents=None, vector_db=FAISS, embedding=HuggingFaceEmbeddings()) -> None:
        self.embedding = embedding
        self.vector_db = vector_db(
            embedding_function=embedding,
            index=faiss.IndexFlatL2(768),
            docstore=InMemoryDocstore,
            index_to_docstore_id={}
        )
        self.db = self.build_db(documents)

    def build_db(self, documents):
        db = self.vector_db.from_documents(
            documents=documents,
            embedding=self.embedding
        )
        return db
    
    def get_retriever(self, search_type: str='mmr', search_kwargs: dict={'k': 10}):
        retriever = self.db.as_retriever(
            search_type=search_type, 
            search_kwargs=search_kwargs
        )
        return retriever