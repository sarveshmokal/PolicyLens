"""
Embedding Agent for PolicyLens.

Third agent in the ingestion pipeline. Takes preprocessed text chunks
and converts them into 384-dimensional dense vectors using SBERT
(all-MiniLM-L6-v2), then stores them in ChromaDB for similarity search.

ChromaDB is configured with persistent storage so embeddings survive
application restarts. Each document's chunks are stored with metadata
(doc_id, page_number, source_file) to enable filtered retrieval
and citation generation.
"""

import os

import chromadb
from sentence_transformers import SentenceTransformer

from src.core.base_agent import BaseAgent
from src.core.config import settings, PROJECT_ROOT


class EmbeddingAgent(BaseAgent):
    """Embeds text chunks with SBERT and stores them in ChromaDB.

    Loads the SBERT model lazily on first use. Manages a single
    ChromaDB collection called 'policy_chunks' that holds all
    document embeddings with metadata for filtered retrieval.
    """

    def __init__(self, name: str = "embedding", description: str = ""):
        super().__init__(name=name, description=description)
        self._model = None
        self._chroma_client = None
        self._collection = None

        # Config
        self.model_name = settings.embedding.get("model_name", "all-MiniLM-L6-v2")
        self.batch_size = settings.embedding.get("batch_size", 64)
        self.collection_name = "policy_chunks"
        self.persist_dir = str(PROJECT_ROOT / "chroma_data")

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the SBERT model on first access.

        Loading a transformer model takes 2-3 seconds and ~200MB RAM.
        Lazy loading means this cost is paid only when embeddings are
        actually needed, not when the agent is registered.
        """
        if self._model is None:
            self.logger.info("Loading SBERT model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            self.logger.info(
                "Model loaded. Embedding dimension: %d",
                self._model.get_embedding_dimension(),
            )
        return self._model

    @property
    def collection(self) -> chromadb.Collection:
        """Lazy-load the ChromaDB collection on first access.

        Uses persistent storage so embeddings survive restarts.
        Creates the collection if it doesn't exist.
        """
        if self._collection is None:
            os.makedirs(self.persist_dir, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(
                path=self.persist_dir
            )
            self._collection = self._chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self.logger.info(
                "ChromaDB collection '%s' ready (%d existing documents)",
                self.collection_name,
                self._collection.count(),
            )
        return self._collection

    def process(self, input_data: dict) -> dict:
        """Embed text chunks and store them in ChromaDB.

        Args:
            input_data: Must contain 'chunks' list from PreprocessorAgent
                        and 'document' metadata dict.

        Returns:
            Dictionary with embedding stats and document info.
        """
        chunks = input_data["chunks"]
        document = input_data.get("document", {})
        doc_id = document.get("doc_id", "unknown")

        self.logger.info(
            "Embedding %d chunks from %s",
            len(chunks),
            document.get("filename", "unknown"),
        )

        # Check if this document is already embedded
        existing = self.collection.get(
            where={"doc_id": doc_id},
        )
        if existing["ids"] and not input_data.get("force_reprocess", False):
            self.logger.info(
                "Document %s already embedded (%d chunks). Skipping. "
                "Set force_reprocess=True to re-embed.",
                doc_id,
                len(existing["ids"]),
            )
            return {
                "document": document,
                "chunks_embedded": len(existing["ids"]),
                "skipped": True,
                "collection_total": self.collection.count(),
            }

        # If re-processing, delete old embeddings first
        if existing["ids"]:
            self.logger.info("Deleting %d old chunks for re-embedding", len(existing["ids"]))
            self.collection.delete(ids=existing["ids"])

        # Extract texts for batch embedding
        texts = [chunk["content"] for chunk in chunks]

        # Generate embeddings in batches
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = self.model.encode(
                batch,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            all_embeddings.extend(batch_embeddings.tolist())

        # Prepare data for ChromaDB
        ids = [chunk["chunk_id"] for chunk in chunks]
        metadatas = []
        for chunk in chunks:
            metadatas.append({
                "doc_id": doc_id,
                "page_number": chunk.get("page_number", 0),
                "chunk_index": chunk.get("chunk_index", 0),
                "word_count": chunk.get("word_count", 0),
                "source_file": document.get("filename", ""),
            })

        # Store in ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=all_embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        self.logger.info(
            "Embedded %d chunks. Collection total: %d",
            len(chunks),
            self.collection.count(),
        )

        return {
            "document": document,
            "chunks_embedded": len(chunks),
            "skipped": False,
            "embedding_dimension": self.model.get_embedding_dimension(),
            "collection_total": self.collection.count(),
        }

    def search(self, query: str, top_k: int = 10, doc_filter: str = None) -> list[dict]:
        """Search for similar chunks using a query string.

        This method is used by the RetrieverAgent for the dense
        component of hybrid search.

        Args:
            query: Natural language query to search for.
            top_k: Number of results to return.
            doc_filter: Optional doc_id to restrict search to one document.

        Returns:
            List of result dicts with chunk content, metadata, and scores.
        """
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        ).tolist()

        search_params = {
            "query_embeddings": query_embedding,
            "n_results": top_k,
        }

        if doc_filter:
            search_params["where"] = {"doc_id": doc_filter}

        results = self.collection.query(**search_params)

        # Flatten ChromaDB's nested result format into clean dicts
        formatted = []
        for i in range(len(results["ids"][0])):
            formatted.append({
                "chunk_id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "score": 1 - results["distances"][0][i],  # Convert distance to similarity
                "metadata": results["metadatas"][0][i],
            })

        return formatted

    def get_collection_stats(self) -> dict:
        """Return statistics about the current ChromaDB collection.

        Used by the dashboard for monitoring the vector store.
        """
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "embedding_model": self.model_name,
            "embedding_dimension": settings.embedding.get("dimension", 384),
            "persist_directory": self.persist_dir,
        }