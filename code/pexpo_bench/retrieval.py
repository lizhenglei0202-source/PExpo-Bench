"""Hybrid retriever: dense (FAISS + BGE-M3) + BM25 + optional reranker.

Loads artifacts produced by knowledge_base/ingest.py:
  • faiss.index       (normalized dot-product = cosine)
  • chunks.parquet    (doc_id, section, text, ...)

Usage:
    from pexpo_bench.retrieval import Retriever
    r = Retriever.load("pexpo_bench/knowledge_base/index")
    passages = r.retrieve("WHO PM2.5 guideline", k=5)
"""
from __future__ import annotations

import hashlib
import pathlib
import re
from dataclasses import dataclass
from typing import Literal


@dataclass
class Passage:
    doc_id: str
    section: str
    text: str
    score: float
    chunk_id: str


_STOP = set('the a an of in on at to for and or but with by from as is are was were '
            'be been being this that these those it its as such if then than'.split())


def _is_low_info_text(text: str) -> bool:
    """Heuristic chunk-quality filter for retrieve-time skipping.

    Returns True if the chunk is unlikely to contain useful answer content:
      • too short (< 200 chars or < 30 content words)
      • references list (many "et al." / "doi:" / numbered citations)
      • no complete sentences in > 100 chars
    """
    if not isinstance(text, str): return True
    t = text.strip()
    if len(t) < 200: return True
    # Content-word count
    words = re.findall(r'\b[a-zA-Z]{2,}\b', t.lower())
    content_words = sum(1 for w in words if w not in _STOP)
    if content_words < 30: return True
    head = t[:1500]
    # References / bibliography
    if len(re.findall(r'\bet al[.,]?\b', head)) >= 4: return True
    if len(re.findall(r'\bdoi\s*:?\s*10\.', head, re.I)) >= 3: return True
    if len(re.findall(r'\b\d+\s*[A-Z][a-z]+\s+[A-Z]', head)) >= 5: return True
    # No sentence terminators in substantial text
    if len(re.findall(r'[.!?]\s+[A-Z]', head)) == 0 and len(head) > 100: return True
    return False


def _content_hash(text: str) -> str:
    """Stable hash of first 300 chars for dedup."""
    if not isinstance(text, str): return ""
    return hashlib.md5(text[:300].encode('utf-8')).hexdigest()[:12]


class Retriever:
    def __init__(self, index, chunks_df, embed_model, bm25=None, reranker=None,
                 alpha: float = 0.5,
                 dedup_content: bool = True,
                 filter_low_info: bool = True,
                 overfetch_factor: int = 8):
        self.index = index
        self.df = chunks_df
        self.embed_model = embed_model
        self.bm25 = bm25
        self.reranker = reranker
        self.alpha = alpha  # dense vs sparse blend weight
        self.dedup_content = dedup_content
        self.filter_low_info = filter_low_info
        self.overfetch_factor = overfetch_factor
        # Precompute low-info mask + content hash on the df once.
        if "_content_hash" not in chunks_df.columns:
            chunks_df["_content_hash"] = chunks_df["text"].astype(str).map(_content_hash)
        if "_low_info" not in chunks_df.columns and filter_low_info:
            chunks_df["_low_info"] = chunks_df["text"].astype(str).map(_is_low_info_text)

    # ---------------------- factory ----------------------
    @classmethod
    def load(cls, index_dir: str,
             embed_model_name: str = "BAAI/bge-m3",
             reranker_name: str | None = "BAAI/bge-reranker-v2-m3",
             use_bm25: bool = True,
             bm25_only: bool = False) -> "Retriever":
        """Load retriever. Pass bm25_only=True to skip FAISS+BGE entirely;
        useful when the dense index hasn't been (re)built but chunks.parquet
        already has the corpus (e.g. after appending authoritative PDFs)."""
        import pandas as pd

        index_dir = pathlib.Path(index_dir)
        df = pd.read_parquet(index_dir / "chunks.parquet")

        if bm25_only:
            from rank_bm25 import BM25Okapi
            bm25 = BM25Okapi([t.lower().split() for t in df["text"].tolist()])
            return cls(index=None, chunks_df=df, embed_model=None,
                       bm25=bm25, reranker=None)

        import faiss
        from sentence_transformers import SentenceTransformer
        index = faiss.read_index(str(index_dir / "faiss.index"))
        model = SentenceTransformer(embed_model_name, trust_remote_code=True)

        bm25 = None
        if use_bm25:
            from rank_bm25 import BM25Okapi
            bm25 = BM25Okapi([t.lower().split() for t in df["text"].tolist()])

        reranker = None
        if reranker_name:
            # Prefer sentence-transformers CrossEncoder (always available),
            # fall back to FlagEmbedding only if requested explicitly.
            try:
                from sentence_transformers import CrossEncoder
                reranker = CrossEncoder(reranker_name)
                # CrossEncoder uses .predict(); wrap to match FlagReranker.compute_score API
                _orig_predict = reranker.predict
                class _SBERTRerankerWrap:
                    def __init__(self, m): self.m = m
                    def compute_score(self, pairs, normalize=False):
                        scores = self.m.predict(pairs)
                        return scores.tolist() if hasattr(scores, 'tolist') else list(scores)
                reranker = _SBERTRerankerWrap(reranker)
            except Exception as e:
                try:
                    from FlagEmbedding import FlagReranker
                    reranker = FlagReranker(reranker_name, use_fp16=True)
                except Exception:
                    print(f"  [Retriever] reranker '{reranker_name}' failed to load: {e}")
                    reranker = None

        return cls(index, df, model, bm25, reranker)

    # ---------------------- core ----------------------
    def retrieve(self, query: str, k: int = 5,
                 mode: Literal["dense", "hybrid", "rerank", "bm25"] = "rerank",
                 candidates: int = 20) -> list[Passage]:
        """Retrieve top-k unique high-info passages.

         changes (transparent to callers):
          1. overfetch FAISS top-(k × overfetch_factor) instead of just k
          2. filter low-info chunks (references lists, < 200 chars, etc.)
          3. dedup by content hash (KB has 2.6× alias duplication)

        candidates is bumped to ensure enough pool for filtering+dedup.
        """
        # : overfetch to leave headroom for filter + dedup
        candidates = max(candidates, k * self.overfetch_factor)

        # BM25-only path
        if mode == "bm25" or self.index is None:
            if self.bm25 is None:
                raise RuntimeError("BM25 not built; load with use_bm25=True")
            scores = self.bm25.get_scores(query.lower().split())
            top_idx = scores.argsort()[-candidates:][::-1]
            pairs = [(int(i), float(scores[i])) for i in top_idx]
            return self._to_passages(self._apply_filters(pairs, k))

        # Dense
        qv = self.embed_model.encode([query], normalize_embeddings=True).astype("float32")
        dense_scores, dense_idx = self.index.search(qv, candidates)
        dense_scores, dense_idx = dense_scores[0], dense_idx[0]

        if mode == "dense":
            pairs = list(zip([int(i) for i in dense_idx], [float(s) for s in dense_scores]))
            return self._to_passages(self._apply_filters(pairs, k))

        # Hybrid blend
        scores: dict[int, float] = {int(i): float(s) * self.alpha
                                    for i, s in zip(dense_idx, dense_scores)}
        if self.bm25 is not None:
            bm_scores = self.bm25.get_scores(query.lower().split())
            bmax = bm_scores.max() if bm_scores.max() > 0 else 1.0
            top_bm = bm_scores.argsort()[-candidates:][::-1]
            for i in top_bm:
                scores[int(i)] = scores.get(int(i), 0.0) + (1 - self.alpha) * (bm_scores[i] / bmax)

        ranked = sorted(scores.items(), key=lambda x: -x[1])

        if mode == "hybrid" or self.reranker is None:
            return self._to_passages(self._apply_filters(ranked, k))

        # Cross-encoder rerank — use top-candidates pool, then filter
        rr_pool = ranked[:candidates]
        rr_pairs = [(query, self.df.iloc[i]["text"]) for i, _ in rr_pool]
        rerank_scores = self.reranker.compute_score(rr_pairs, normalize=True)
        reranked = sorted(zip([i for i, _ in rr_pool], rerank_scores),
                          key=lambda x: -x[1])
        return self._to_passages(self._apply_filters(reranked, k))

    def _apply_filters(self, pairs: list[tuple[int, float]], k: int
                       ) -> list[tuple[int, float]]:
        """: filter low-info chunks, dedup by content hash, take top-k."""
        if not (self.filter_low_info or self.dedup_content):
            return pairs[:k]
        seen_hashes: set[str] = set()
        kept: list[tuple[int, float]] = []
        for idx, score in pairs:
            row = self.df.iloc[int(idx)]
            if self.filter_low_info and row.get("_low_info", False):
                continue
            if self.dedup_content:
                h = row.get("_content_hash") or _content_hash(row["text"])
                if h in seen_hashes: continue
                seen_hashes.add(h)
            kept.append((idx, score))
            if len(kept) >= k: break
        return kept

    def _to_passages(self, pairs: list[tuple[int, float]]) -> list[Passage]:
        out = []
        cols = self.df.columns
        for idx, score in pairs:
            row = self.df.iloc[int(idx)]
            # `section` column doesn't always exist (chunks.parquet schema
            # evolved). Fall back to a page-derived label.
            if "section" in cols:
                section = row["section"]
            elif "page" in cols and row["page"] == row["page"]:  # not NaN
                section = f"page_{int(row['page'])}"
            else:
                section = ""
            out.append(Passage(
                doc_id=row["doc_id"], section=section,
                text=row["text"], score=float(score),
                chunk_id=row["chunk_id"],
            ))
        return out
