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
from local_llm import (
    NO_ANSWER,
    generate_answer,
    refine_answer,
)
from local_reranker import RerankedResult, RetrievedResult, rerank_results


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FAISS_INDEX_PATH = PROCESSED_DIR / "index.faiss"
METADATA_PATH = PROCESSED_DIR / "metadata.json"

MIN_SIMILARITY = 0.85
CHUNKS_PER_BATCH = 5
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


def split_into_batches(
    results: list[RerankedResult],
    batch_size: int = CHUNKS_PER_BATCH,
) -> list[list[RerankedResult]]:
    if batch_size <= 0:
        raise ValueError("The batch size must be greater than zero.")

    return [
        results[start:start + batch_size]
        for start in range(0, len(results), batch_size)
    ]


def build_context(results: list[RerankedResult]) -> str:
    return "\n\n".join(
        f"Evidence [S{index}]\nSource: {item['source']}\n{item['text']}"
        for index, (_, _, item) in enumerate(results, start=1)
    )


def split_into_source_batches(
    results: list[RerankedResult],
    batch_size: int = CHUNKS_PER_BATCH,
) -> list[list[RerankedResult]]:
    if batch_size <= 0:
        raise ValueError("The batch size must be greater than zero.")

    representatives = []
    remaining = []
    seen_providers = set()

    for result in results:
        _, _, item = result
        provider = item.get("provider", item["source"])

        if provider not in seen_providers and len(representatives) < batch_size:
            representatives.append(result)
            seen_providers.add(provider)
        else:
            remaining.append(result)

    batches = [representatives] if representatives else []
    return [*batches, *split_into_batches(remaining, batch_size)]


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)

    return "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    ).casefold().strip(" .\n\t")


def is_no_answer(answer: str) -> bool:
    return normalize_text(answer) == normalize_text(NO_ANSWER)


def _requested_list(question: str) -> list[str]:
    match = re.search(
        r"\b(?:report|provide|list|give|reporta|informa|dame|lista)\b"
        r"\s+(.+?)(?:[.?!]|$)",
        question,
        re.IGNORECASE,
    )

    if not match:
        return []

    segment = re.sub(
        r"^(?:only\s+)?(?:the\s+)?(?:available\s+)?",
        "",
        match.group(1).strip(),
        flags=re.IGNORECASE,
    )
    segment = re.sub(
        r"\s+for\s+[A-Za-z0-9₀-₉]+$",
        "",
        segment,
        flags=re.IGNORECASE,
    )
    parts = re.split(r"\s*,\s*|\s+and\s+|\s+y\s+", segment)
    cleaned = [
        re.sub(
            r"^(?:(?:and|y)\s+)?(?:(?:the|la|el|los|las)\s+)?",
            "",
            part.strip(),
            flags=re.I,
        )
        for part in parts
        if part.strip()
    ]
    return cleaned if len(cleaned) >= 2 else []


def _evidence_type(label: str, value: str, context: str) -> str | None:
    normalized_label = normalize_text(label)
    normalized_value = normalize_text(
        re.sub(r"\([^)]*\)$", "", value).strip()
    )

    for line in context.splitlines():
        normalized_line = normalize_text(line)

        if (
            normalized_label not in normalized_line
            and normalized_value not in normalized_line
        ):
            continue

        match = re.search(
            r"evidence type:\s*([^;)]+)",
            line,
            re.IGNORECASE,
        )

        if match:
            return match.group(1).strip()

    return None


def enforce_requested_scope(
    question: str,
    context: str,
    answer: str,
) -> str:
    requested_items = _requested_list(question)

    if not requested_items:
        return answer

    candidate_items = []

    for line in answer.splitlines():
        match = re.match(r"\s*[-*]?\s*([^:]+):\s*(.+?)\s*$", line)

        if match:
            candidate_items.append((match.group(1).strip(), match.group(2).strip()))

    if not candidate_items:
        return answer

    distinction_requested = bool(
        re.search(
            r"distinguish|differentiate|calculated|experimental|disting",
            question,
            re.IGNORECASE,
        )
    )
    missing_text = (
        "No disponible."
        if re.search(r"\b(?:de|del|para|y|informa|dame)\b", question, re.I)
        else "Not available."
    )
    scoped_lines = []

    for requested in requested_items:
        normalized_requested = normalize_text(requested)
        selected = next(
            (
                (label, value)
                for label, value in candidate_items
                if normalize_text(label) in normalized_requested
                or normalized_requested in normalize_text(label)
            ),
            None,
        )

        if selected is None:
            value = missing_text
        else:
            _, value = selected

            if re.search(
                r"not provided|not available|no disponible",
                value,
                re.IGNORECASE,
            ):
                value = missing_text

            if distinction_requested and not re.search(
                r"not provided|not available|no disponible",
                value,
                re.IGNORECASE,
            ):
                evidence_type = _evidence_type(requested, value, context)

                if evidence_type and evidence_type.casefold() not in value.casefold():
                    value = f"{value} ({evidence_type})"

        scoped_lines.append(f"- {requested.strip().capitalize()}: {value}")

    return "\n".join(scoped_lines)


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

    batches = split_into_source_batches(reranked_results)

    if all("provider" in item for _, _, item in reranked_results):
        context = build_context(batches[0])
        draft = generate_answer(question, context)
        answer = refine_answer(question, context, draft)
        answer = enforce_requested_scope(question, context, answer)

        if is_no_answer(answer):
            return AnswerDetails(NO_ANSWER, [], notices)

        sources = list(
            dict.fromkeys(
                item["source"] for _, _, item in batches[0]
            )
        )
        return AnswerDetails(answer, sources, notices)

    for batch_number, batch in enumerate(batches, start=1):
        if DEBUG:
            print(
                f"\nReviewing batch {batch_number}/{len(batches)} "
                f"({len(batch)} chunks)..."
            )

        context = build_context(batch)
        generation_started = perf_counter()
        partial_answer = generate_answer(question, context)
        generation_time = perf_counter() - generation_started

        if DEBUG:
            print(f"Qwen generation time: {generation_time:.3f} s")

        if is_no_answer(partial_answer):
            continue

        partial_answer = refine_answer(
            question,
            context,
            partial_answer,
        )
        partial_answer = enforce_requested_scope(
            question,
            context,
            partial_answer,
        )

        sources = list(
            dict.fromkeys(item["source"] for _, _, item in batch)
        )

        return AnswerDetails(partial_answer, sources, notices)

    return AnswerDetails(NO_ANSWER, [], notices)


def answer_question(question: str) -> tuple[str, list[str]]:
    result = answer_question_with_details(question)
    return result.answer, result.sources
