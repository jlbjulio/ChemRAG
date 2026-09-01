import json
import re
import unicodedata
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import faiss
import numpy as np

from chemistry.online import retrieve_online
from chemistry.schema import render_record_as_text
from load_document import PROJECT_ROOT
from local_embeddings import EMBEDDING_DIMENSION, create_query_embedding
from local_llm import NO_ANSWER, generate_answer
from local_reranker import RerankedResult, RetrievedResult, rerank_results


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FAISS_INDEX_PATH = PROCESSED_DIR / "index.faiss"
METADATA_PATH = PROCESSED_DIR / "metadata.json"

MIN_SIMILARITY = 0.85
DEBUG = False

_vector_store_loaded = False
_faiss_index = None
_metadata: list[dict[str, str]] = []


@dataclass(frozen=True)
class AnswerDetails:
    answer: str
    sources: list[str]
    notices: list[str]


def load_vector_store() -> tuple[Any | None, list[dict[str, str]]]:
    global _vector_store_loaded, _faiss_index, _metadata

    if _vector_store_loaded:
        return _faiss_index, _metadata

    _vector_store_loaded = True

    if not FAISS_INDEX_PATH.exists() or not METADATA_PATH.exists():
        return None, []

    index = faiss.read_index(str(FAISS_INDEX_PATH))
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    if index.d != EMBEDDING_DIMENSION:
        raise ValueError(
            "The FAISS index has the wrong dimension: "
            f"{index.d}; expected {EMBEDDING_DIMENSION}."
        )

    if index.ntotal != len(metadata):
        raise ValueError(
            "The number of FAISS vectors does not match the metadata."
        )

    _faiss_index = index
    _metadata = metadata
    return _faiss_index, _metadata


def retrieve_local_chunks(question: str) -> list[RetrievedResult]:
    index, metadata = load_vector_store()

    if index is None or not metadata:
        return []

    query_embedding = create_query_embedding(question)
    query_vector = np.asarray([query_embedding], dtype=np.float32)
    _, scores, indices = index.range_search(
        query_vector,
        MIN_SIMILARITY,
    )
    results = [
        (float(score), metadata[int(index_id)])
        for score, index_id in zip(scores, indices)
    ]
    results.sort(key=lambda result: result[0], reverse=True)
    return results


def retrieve_chunks_with_notices(
    question: str,
) -> tuple[list[RetrievedResult], list[str]]:
    online_result = retrieve_online(question)

    if DEBUG:
        print(
            "\nDetected online query: "
            f"{online_result.intent.kind}={online_result.intent.value}"
        )

        for warning in online_result.warnings:
            print(f"Online source warning: {warning}")

    if online_result.records:
        return [
            (
                1.0,
                {
                    "source": (
                        f"{record.source} {record.source_id}"
                    ),
                    "provider": record.source,
                    "source_url": record.source_url,
                    "text": render_record_as_text(record),
                },
            )
            for record in online_result.records
        ], []

    return retrieve_local_chunks(question), []


def retrieve_chunks(question: str) -> list[RetrievedResult]:
    results, _ = retrieve_chunks_with_notices(question)
    return results


def build_context(results: list[RerankedResult]) -> str:
    return "\n\n".join(
        f"Evidence [S{index}]\nSource: {item['source']}\n{item['text']}"
        for index, (_, _, item) in enumerate(results, start=1)
    )

def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)

    return "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    ).casefold().strip(" .\n\t")


def is_no_answer(answer: str) -> bool:
    return normalize_text(answer) == normalize_text(NO_ANSWER)


def _requested_properties(question: str) -> list[str]:
    patterns = (
        r"\b(?:report|provide|list|give|show|tell me|reporta|informa|dame|"
        r"lista|muestra)\b\s+(.+?)(?:[.?!]|$)",
        r"\b(?:what (?:is|are)|which (?:is|are)|cu[aá]l(?:es)? "
        r"(?:es|son))\b\s+(.+?)(?:[.?!]|$)",
    )
    match = next(
        (
            candidate
            for pattern in patterns
            if (
                candidate := re.search(
                    pattern,
                    question,
                    re.IGNORECASE,
                )
            )
        ),
        None,
    )

    if not match:
        return []

    segment = re.split(
        r",\s*(?:and\s+|y\s+)?(?:distinguish|differentiate|distingue)",
        match.group(1),
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    parts = re.split(r"\s*,\s*|\s+and\s+|\s+y\s+", segment)
    cleaned = []

    for part in parts:
        label = re.sub(
            r"^(?:(?:and|y)\s+)?(?:only\s+)?(?:the\s+)?"
            r"(?:(?:solo|solamente)\s+)?(?:(?:el|la|los|las)\s+)?"
            r"(?:(?:available|disponible)\s+)?",
            "",
            part.strip(),
            flags=re.IGNORECASE,
        )
        label = re.sub(
            r"\s+(?:of|for|de|del|para)\s+(?:(?:the|el|la)\s+)?"
            r"[A-Za-zÀ-ÿ0-9₀-₉().+-]+$",
            "",
            label,
            flags=re.IGNORECASE,
        ).strip()

        if label:
            cleaned.append(label)

    return cleaned if len(cleaned) >= 2 else []


def _property_key(label: str) -> str:
    normalized = normalize_text(label)

    if "thermodynamic stability" in normalized or "convex hull" in normalized:
        return "thermodynamic_stability"

    if "estabilidad termodinamica" in normalized or "casco convexo" in normalized:
        return "thermodynamic_stability"

    if "formation energy" in normalized or "energia de formacion" in normalized:
        return "formation_energy"

    if "band gap" in normalized or "brecha de banda" in normalized:
        return "band_gap"

    if "space group" in normalized or "grupo espacial" in normalized:
        return "space_group"

    if any(
        term in normalized
        for term in (
            "crystal structure",
            "crystal prototype",
            "prototype",
            "estructura cristalina",
            "prototipo",
        )
    ):
        return "crystal_structure"

    return re.sub(
        r"\b(?:reported|available|calculated|experimental|reportado|"
        r"disponible|calculado|experimental)\b",
        "",
        normalized,
    ).strip()


def _context_properties(context: str) -> dict[str, list[tuple[str, str | None]]]:
    properties: dict[str, list[tuple[str, str | None]]] = {}
    labeled_evidence_pattern = re.compile(
        r"^\s*[-*]?\s*(.+?)\s+\(evidence type:\s*([^)]+)\)"
        r":\s*(.+?)\s*$",
        re.IGNORECASE,
    )
    line_pattern = re.compile(r"^\s*[-*]?\s*([^:]+):\s*(.+?)\s*$")
    value_evidence_pattern = re.compile(
        r"\s*\(evidence type:\s*([^;)]+)(?:;[^)]*)?\)\s*$",
        re.IGNORECASE,
    )

    for line in context.splitlines():
        evidence_match = labeled_evidence_pattern.match(line)

        if evidence_match:
            label, label_evidence, value = evidence_match.groups()
        else:
            match = line_pattern.match(line)

            if not match:
                continue

            label, value = match.groups()
            label_evidence = None

        key = _property_key(label)
        value_evidence = value_evidence_pattern.search(value)
        evidence_type = label_evidence

        if value_evidence:
            evidence_type = value_evidence.group(1).strip()
            value = value_evidence_pattern.sub("", value).strip()

        entry = (value, evidence_type.strip() if evidence_type else None)
        properties.setdefault(key, [])

        if entry not in properties[key]:
            properties[key].append(entry)

    return properties


def enforce_structured_scope(
    question: str,
    context: str,
    answer: str,
) -> str:
    requested = _requested_properties(question)

    if not requested:
        return answer

    properties = _context_properties(context)
    distinguish = bool(
        re.search(
            r"distinguish|differentiate|calculated|experimental|disting",
            question,
            re.IGNORECASE,
        )
    )
    spanish = bool(re.search(r"[¿¡áéíóúñ]", question, re.IGNORECASE))
    missing = "No disponible" if spanish else "Not available"
    lines = []
    found_property = False

    for label in requested:
        entries = properties.get(_property_key(label), [])

        if not entries:
            value = missing
        else:
            found_property = True
            value, evidence_type = entries[0]

            if distinguish and evidence_type:
                value = f"{value} ({evidence_type})"

        lines.append(f"- {label.strip().capitalize()}: {value}")

    return "\n".join(lines) if found_property else NO_ANSWER


def answer_question_with_details(question: str) -> AnswerDetails:
    relevant_results, notices = retrieve_chunks_with_notices(question)

    if not relevant_results:
        return AnswerDetails(NO_ANSWER, [], notices)

    if DEBUG:
        print("\nRetrieved candidates:")

        for score, item in relevant_results:
            title = item["text"].splitlines()[0]
            print(f"- {score:.4f}: {title}")

    reranking_started = perf_counter()
    reranked_results = rerank_results(question, relevant_results)
    reranking_time = perf_counter() - reranking_started

    if DEBUG:
        print(f"\nReranked candidates ({reranking_time:.3f} s):")

        for reranker_score, retrieval_score, item in reranked_results:
            title = item["text"].splitlines()[0]
            print(
                f"- reranker={reranker_score:.4f}, "
                f"retrieval={retrieval_score:.4f}: {title}"
            )

    context = build_context(reranked_results)
    generation_started = perf_counter()
    answer = generate_answer(question, context)
    answer = enforce_structured_scope(question, context, answer)
    generation_time = perf_counter() - generation_started

    if DEBUG:
        print(f"\nQwen generation time: {generation_time:.3f} s")

    if is_no_answer(answer):
        return AnswerDetails(NO_ANSWER, [], notices)

    sources = list(
        dict.fromkeys(
            item["source"] for _, _, item in reranked_results
        )
    )
    return AnswerDetails(answer, sources, notices)


def answer_question(question: str) -> tuple[str, list[str]]:
    result = answer_question_with_details(question)
    return result.answer, result.sources
