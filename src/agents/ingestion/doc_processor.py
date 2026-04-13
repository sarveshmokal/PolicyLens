"""
Document Processor Agent for PolicyLens.

First agent in the ingestion pipeline. Opens PDF files using PyMuPDF,
extracts text content page by page, splits into overlapping chunks,
and produces structured output for downstream agents.

Chunking strategy: Recursive character splitting with configurable
chunk_size and overlap. This preserves paragraph boundaries where
possible while keeping chunks within the token budget for embedding.
"""

import hashlib
import os

import fitz  # PyMuPDF

from src.core.base_agent import BaseAgent
from src.core.config import settings


class DocProcessorAgent(BaseAgent):
    """Extracts text and metadata from PDF documents.

    Reads PDF files using PyMuPDF, extracts text page by page,
    then splits the full text into overlapping chunks suitable
    for embedding and retrieval.
    """

    def __init__(self, name: str = "doc_processor", description: str = ""):
        super().__init__(name=name, description=description)
        self.chunk_size = settings.chunking.get("chunk_size", 1000)
        self.chunk_overlap = settings.chunking.get("chunk_overlap", 200)
        self.min_chunk_size = settings.chunking.get("min_chunk_size", 100)

    def process(self, input_data: dict) -> dict:
        """Process a single PDF file into metadata and text chunks.

        Args:
            input_data: Must contain 'file_path' pointing to a PDF.

        Returns:
            Dictionary with 'document' metadata and 'chunks' list.
        """
        file_path = input_data["file_path"]
        self.logger.info("Processing PDF: %s", file_path)

        # Extract metadata and raw text
        metadata = self._extract_metadata(file_path)
        pages_text = self._extract_text(file_path)

        # Chunk the extracted text
        chunks = self._create_chunks(pages_text, metadata["doc_id"])

        self.logger.info(
            "Extracted %d pages, created %d chunks from %s",
            metadata["page_count"],
            len(chunks),
            metadata["filename"],
        )

        return {
            "document": metadata,
            "chunks": chunks,
        }

    def _extract_metadata(self, file_path: str) -> dict:
        """Extract document metadata from a PDF file.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Dictionary with doc_id, title, filename, page_count, file_size.
        """
        doc = fitz.open(file_path)
        filename = os.path.basename(file_path)

        # Generate a stable document ID from the filename
        doc_id = hashlib.md5(filename.encode()).hexdigest()[:12]

        # Try to get title from PDF metadata, fall back to filename
        pdf_metadata = doc.metadata
        title = pdf_metadata.get("title", "")
        if not title or title.strip() == "":
            # Derive title from filename: remove extension, replace underscores
            title = os.path.splitext(filename)[0].replace("_", " ")

        metadata = {
            "doc_id": doc_id,
            "title": title,
            "source_org": self._detect_source_org(filename),
            "filename": filename,
            "page_count": len(doc),
            "file_size_bytes": os.path.getsize(file_path),
        }

        doc.close()
        return metadata

    def _extract_text(self, file_path: str) -> list[dict]:
        """Extract text from each page of a PDF.

        Args:
            file_path: Path to the PDF file.

        Returns:
            List of dicts with 'page_number' and 'text' for each page.
        """
        doc = fitz.open(file_path)
        pages = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")

            # Skip pages with very little text (covers, blank pages)
            if len(text.strip()) < 50:
                continue

            pages.append({
                "page_number": page_num + 1,  # 1-indexed for human readability
                "text": text.strip(),
            })

        doc.close()
        self.logger.debug("Extracted text from %d non-empty pages", len(pages))
        return pages

    def _create_chunks(self, pages_text: list[dict], doc_id: str) -> list[dict]:
        """Split extracted page text into overlapping chunks.

        Uses recursive splitting: first tries to split on double newlines
        (paragraph boundaries), then single newlines, then spaces. This
        preserves natural text boundaries where possible.

        Args:
            pages_text: List of page dicts from _extract_text.
            doc_id: Parent document identifier.

        Returns:
            List of chunk dicts with chunk_id, content, page_number, etc.
        """
        chunks = []
        chunk_index = 0

        for page_data in pages_text:
            page_num = page_data["page_number"]
            text = page_data["text"]

            # Split this page's text into chunks
            page_chunks = self._recursive_split(text)

            for chunk_text in page_chunks:
                # Skip chunks that are too small to be useful
                if len(chunk_text.strip()) < self.min_chunk_size:
                    continue

                chunk_id = f"{doc_id}_chunk_{chunk_index:04d}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "content": chunk_text.strip(),
                    "page_number": page_num,
                    "chunk_index": chunk_index,
                    "token_count": len(chunk_text.split()),
                })
                chunk_index += 1

        return chunks

    def _recursive_split(self, text: str) -> list[str]:
        """Split text recursively using a hierarchy of separators.

        Tries to split on paragraph breaks first, preserving natural
        boundaries. Falls back to line breaks, then spaces if chunks
        are still too large.

        Args:
            text: Raw text string to split.

        Returns:
            List of text chunks, each within chunk_size limit.
        """
        if len(text) <= self.chunk_size:
            return [text]

        separators = ["\n\n", "\n", ". ", " "]
        chunks = []
        current_chunk = ""

        # Find the best separator that exists in the text
        separator = " "
        for sep in separators:
            if sep in text:
                separator = sep
                break

        parts = text.split(separator)

        for part in parts:
            # If adding this part would exceed chunk_size
            if len(current_chunk) + len(part) + len(separator) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    # Keep overlap from the end of the previous chunk
                    if self.chunk_overlap > 0 and len(current_chunk) > self.chunk_overlap:
                        current_chunk = current_chunk[-self.chunk_overlap:]
                    else:
                        current_chunk = ""

            current_chunk += (separator if current_chunk else "") + part

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _detect_source_org(self, filename: str) -> str:
        """Detect the publishing organization from the filename.

        Simple heuristic based on filename prefixes.

        Args:
            filename: PDF filename.

        Returns:
            Organization name or empty string.
        """
        filename_upper = filename.upper()
        org_map = {
            "OECD": "OECD",
            "IMF": "IMF",
            "WHO": "WHO",
            "UNCTAD": "UNCTAD",
            "EU_AI": "European Union",
        }
        for prefix, org in org_map.items():
            if filename_upper.startswith(prefix):
                return org
        return ""