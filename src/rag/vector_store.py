import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from src.config import CHROMA_PERSIST_DIR
from src.rag.embeddings import LocalEmbeddingEngine
from src.utils.logger import get_logger

logger = get_logger("VectorStore")

class ChromaVectorStore:
    """
    Manages persistent local ChromaDB collections for RAG retrieval.
    Operates without cloud subscriptions or external databases.
    """
    def __init__(self, collection_name: str = "career_knowledge_base"):
        self.collection_name = collection_name
        self.embedding_engine = LocalEmbeddingEngine()

        CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Connected to ChromaDB collection '{collection_name}' ({self.collection.count()} docs).")

    def count(self) -> int:
        return self.collection.count()

    def add_documents(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Embeds and upserts documents into the collection."""
        if not documents:
            return
        logger.info(f"Embedding and upserting {len(documents)} documents into '{self.collection_name}'...")
        embeddings = self.embedding_engine.embed_documents(documents)
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas or [{} for _ in documents]
        )

    def search(
        self,
        query: str,
        n_results: int = 5,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes a dense cosine similarity search for the given query.
        """
        if self.collection.count() == 0:
            return []

        query_embedding = self.embedding_engine.embed_query(query)
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, self.collection.count()),
            "include": ["documents", "metadatas", "distances"]
        }
        if where_filter:
            kwargs["where"] = where_filter

        results = self.collection.query(**kwargs)

        formatted = []
        if results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
            ids = results["ids"][0] if results.get("ids") else [""] * len(docs)

            for doc_id, doc, meta, dist in zip(ids, docs, metas, dists):
                # Cosine distance to similarity: 1 - distance
                similarity = max(0.0, 1.0 - dist)
                formatted.append({
                    "id": doc_id,
                    "content": doc,
                    "metadata": meta,
                    "similarity": round(similarity, 4)
                })
        return formatted

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Retrieves all documents stored in the collection for sparse indexing."""
        data = self.collection.get(include=["documents", "metadatas"])
        docs = []
        if data and data["documents"]:
            for doc_id, text, meta in zip(data["ids"], data["documents"], data["metadatas"] or []):
                docs.append({
                    "id": doc_id,
                    "content": text,
                    "metadata": meta
                })
        return docs
