from typing import Union
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

class VectorDB:
    def __init__(self, documents=None, 
                 vector_db: Union[Chroma, FAISS] = Chroma, 
                 embedding=HuggingFaceEmbeddings()
                 ) -> None:
        self.embedding = embedding
        self.vector_db = vector_db()
        self.db = self.build_db(documents)

    def build_db(self, documents):
        db = self.vector_db.from_documents(
            documents=documents,
            embedding=self.embedding
        )
        return db
    
    def get_retriever(self, 
                      search_type: str='similarity', 
                      search_kwargs: dict={'k': 3}
                      ):
        retriever = self.db.as_retriever(
            search_type=search_type, 
            search_kwargs=search_kwargs
        )
        return retriever