"""
Preprocessor Agent for PolicyLens.

Second agent in the ingestion pipeline. Takes raw text chunks from
DocProcessorAgent and cleans them: normalizes whitespace, removes
headers/footers/page artifacts, and runs spaCy NER to identify
named entities (organizations, countries, dates, monetary values).

The cleaned, entity-tagged chunks are ready for embedding.
"""

import re

import spacy

from src.core.base_agent import BaseAgent


class PreprocessorAgent(BaseAgent):
    """Cleans raw text chunks and extracts named entities using spaCy.

    Applies text normalization, removes PDF artifacts, and runs
    NER to enrich chunks with entity metadata for downstream use
    in retrieval and synthesis.
    """

    def __init__(self, name: str = "preprocessor", description: str = ""):
        super().__init__(name=name, description=description)
        self.nlp = spacy.load("en_core_web_sm")
        # Disable heavy parser, add lightweight sentencizer instead
        self.nlp.select_pipes(enable=["ner", "tagger", "attribute_ruler", "lemmatizer"])
        self.nlp.add_pipe("sentencizer")

    def process(self, input_data: dict) -> dict:
        """Clean and enrich text chunks with NLP annotations.

        Args:
            input_data: Must contain 'chunks' list from DocProcessorAgent.

        Returns:
            Dictionary with 'chunks' (cleaned and enriched) and
            'entity_summary' with aggregate entity counts.
        """
        raw_chunks = input_data["chunks"]
        document = input_data.get("document", {})
        self.logger.info(
            "Preprocessing %d chunks from %s",
            len(raw_chunks),
            document.get("filename", "unknown"),
        )

        processed_chunks = []
        all_entities = []

        for chunk in raw_chunks:
            cleaned = self._clean_text(chunk["content"])

            # Skip chunks that become too short after cleaning
            if len(cleaned.split()) < 10:
                continue

            entities = self._extract_entities(cleaned)
            all_entities.extend(entities)

            processed_chunk = {
                **chunk,
                "content": cleaned,
                "entities": entities,
                "word_count": len(cleaned.split()),
                "sentence_count": len(list(self.nlp(cleaned).sents)),
            }
            processed_chunks.append(processed_chunk)

        entity_summary = self._summarize_entities(all_entities)

        self.logger.info(
            "Preprocessing complete: %d -> %d chunks, %d unique entities found",
            len(raw_chunks),
            len(processed_chunks),
            len(entity_summary),
        )

        return {
            "document": document,
            "chunks": processed_chunks,
            "entity_summary": entity_summary,
        }

    def _clean_text(self, text: str) -> str:
        """Normalize and clean raw PDF text.

        Removes common PDF artifacts: page numbers, headers/footers,
        excessive whitespace, and encoding remnants.

        Args:
            text: Raw text from PDF extraction.

        Returns:
            Cleaned text string.
        """
        # Remove page number patterns (standalone numbers, "Page X", etc.)
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"(?i)page\s+\d+", "", text)

        # Remove common PDF header/footer patterns
        text = re.sub(r"©.*?\d{4}", "", text)  # Copyright lines
        text = re.sub(r"https?://\S+", "", text)  # URLs (keep text, remove links)

        # Normalize whitespace
        text = re.sub(r"\t", " ", text)  # Tabs to spaces
        text = re.sub(r" {2,}", " ", text)  # Multiple spaces to single
        text = re.sub(r"\n{3,}", "\n\n", text)  # Multiple newlines to double
        text = re.sub(r"^\s+$", "", text, flags=re.MULTILINE)  # Blank lines

        # Fix common encoding artifacts
        text = text.replace("\u2019", "'")  # Right single quote
        text = text.replace("\u2018", "'")  # Left single quote
        text = text.replace("\u201c", '"')  # Left double quote
        text = text.replace("\u201d", '"')  # Right double quote
        text = text.replace("\u2013", "-")  # En dash
        text = text.replace("\u2014", "-")  # Em dash
        text = text.replace("\u2026", "...")  # Ellipsis

        return text.strip()

    def _extract_entities(self, text: str) -> list[dict]:
        """Extract named entities from text using spaCy.

        Focuses on entity types relevant to policy documents:
        organizations, geopolitical entities, dates, and monetary values.

        Args:
            text: Cleaned text to analyze.

        Returns:
            List of entity dicts with text, label, and start/end positions.
        """
        doc = self.nlp(text)

        # Entity types relevant to policy analysis
        relevant_types = {
            "ORG",    # Organizations (OECD, IMF, WHO)
            "GPE",    # Countries and cities
            "DATE",   # Dates and time periods
            "MONEY",  # Monetary values
            "NORP",   # Nationalities, religious/political groups
            "LAW",    # Laws and regulations
            "EVENT",  # Named events
            "PERCENT",  # Percentage values
        }

        entities = []
        seen = set()  # Deduplicate within the same chunk

        for ent in doc.ents:
            if ent.label_ in relevant_types:
                # Deduplicate: same text + same label = one entry
                key = (ent.text.strip(), ent.label_)
                if key not in seen:
                    seen.add(key)
                    entities.append({
                        "text": ent.text.strip(),
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                    })

        return entities

    def _summarize_entities(self, entities: list[dict]) -> dict:
        """Create a frequency summary of entities across all chunks.

        Args:
            entities: All entities extracted from all chunks.

        Returns:
            Dict mapping entity label to list of (text, count) tuples,
            sorted by frequency descending.
        """
        from collections import Counter

        label_counts = {}
        for entity in entities:
            label = entity["label"]
            text = entity["text"]
            if label not in label_counts:
                label_counts[label] = Counter()
            label_counts[label][text] += 1

        summary = {}
        for label, counter in label_counts.items():
            summary[label] = [
                {"text": text, "count": count}
                for text, count in counter.most_common(20)
            ]

        return summary