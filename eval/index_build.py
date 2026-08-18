"""Builds a throwaway, in-memory FAISS index from the sampled MSMARCO-XI
examples' candidate passages -- mirrors the target project's own
benchmark/ragbench.py pattern (same HNSW index type/params, same chunker,
same embedding model) so retrieval results reflect this project's actual
retrieval behavior, not a different pipeline. Never touches the target
project's live index/ directory.

Deliberately mixed-language: every candidate passage from every sampled
example goes in, English and Hindi both, tagged by language. This is what
makes the retrieval check's cross-lingual metric possible (see
eval/checks/retrieval.py) -- and it directly tests the thing this
project's embedding model was fine-tuned for (Hindi + English mixed
retrieval on this exact dataset), rather than a synthetic English-only or
Hindi-only setup.
"""
from dataclasses import dataclass

import numpy as np

from eval import target
from eval.dataset import EvalExample


@dataclass
class ChunkRecord:
    query_id: int
    lang: str          # "en" | "hi"
    is_selected: bool
    text: str


def build_index(examples: list[EvalExample]):
    target.load_target()
    import faiss
    from app.chunking import chunk_text
    from app.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_DIM, HNSW_EF_CONSTRUCTION, HNSW_EF_SEARCH, HNSW_M
    from app.embedder import embed, get_model

    texts: list[str] = []
    records: list[ChunkRecord] = []

    for ex in examples:
        selected_idx = ex.gt_passage_index  # None for unanswerable examples
        for lang, candidates in (("en", ex.candidates_en), ("hi", ex.candidates_hi)):
            for i, passage in enumerate(candidates):
                if not passage:
                    continue
                for chunk in chunk_text(passage, CHUNK_SIZE, CHUNK_OVERLAP):
                    texts.append(chunk)
                    records.append(
                        ChunkRecord(query_id=ex.query_id, lang=lang, is_selected=(i == selected_idx), text=chunk)
                    )

    get_model()  # ensure loaded before timing/embedding
    batch_size = 64
    vector_batches = [embed(texts[i : i + batch_size]) for i in range(0, len(texts), batch_size)]
    vectors = np.vstack(vector_batches) if vector_batches else np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    index = faiss.IndexHNSWFlat(EMBEDDING_DIM, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH
    if len(vectors):
        index.add(vectors)

    return index, records
