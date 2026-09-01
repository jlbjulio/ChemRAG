import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

import search_chunks  # noqa: E402
from model_runtime import configure_utf8_console  # noqa: E402


def run_scenario(answer_chunk: int) -> bool:
    fake_results = [
        (
            1.0 - (number / 1000),
            {
                "source": f"document_{number}.txt",
                "text": f"CHUNK {number}: irrelevant information.",
            },
        )
        for number in range(1, 19)
    ]

    fake_results[answer_chunk - 1][1]["text"] = (
        f"CHUNK {answer_chunk}: the secret code is ORION."
    )

    reviewed_contexts: list[str] = []

    original_retrieve_chunks = search_chunks.retrieve_chunks_with_notices
    original_rerank_results = search_chunks.rerank_results
    original_generate_answer = search_chunks.generate_answer
    original_refine_answer = search_chunks.refine_answer
    original_debug = search_chunks.DEBUG

    def fake_retrieve_chunks(
        question: str,
    ) -> tuple[list[search_chunks.RetrievedResult], list[str]]:
        return fake_results, []

    def fake_rerank_results(
        question: str,
        results: list[search_chunks.RetrievedResult],
    ) -> list[search_chunks.RerankedResult]:
        return [
            (
                float(len(results) - position),
                faiss_score,
                item,
            )
            for position, (faiss_score, item)
            in enumerate(results)
        ]

    def fake_generate_answer(
        question: str,
        context: str,
    ) -> str:
        reviewed_contexts.append(context)

        if "secret code is ORION" in context:
            return "The secret code is ORION."

        return search_chunks.NO_ANSWER

    def fake_refine_answer(
        question: str,
        context: str,
        draft: str,
    ) -> str:
        return draft


    try:
        search_chunks.DEBUG = False
        search_chunks.retrieve_chunks_with_notices = fake_retrieve_chunks
        search_chunks.rerank_results = fake_rerank_results
        search_chunks.generate_answer = fake_generate_answer
        search_chunks.refine_answer = fake_refine_answer

        answer, sources = search_chunks.answer_question(
            "What is the secret code?"
        )
    finally:
        search_chunks.retrieve_chunks_with_notices = original_retrieve_chunks
        search_chunks.rerank_results = original_rerank_results
        search_chunks.generate_answer = original_generate_answer
        search_chunks.refine_answer = original_refine_answer
        search_chunks.DEBUG = original_debug

    expected_batches = math.ceil(
        answer_chunk / search_chunks.CHUNKS_PER_BATCH
    )

    success = (
        len(reviewed_contexts) == expected_batches
        and "ORION" in answer
        and f"document_{answer_chunk}.txt" in sources
    )

    print(
        f"Chunk {answer_chunk}: "
        f"{len(reviewed_contexts)} batch(es) reviewed"
    )

    return success


def main() -> None:
    configure_utf8_console()
    early_success = run_scenario(answer_chunk=2)
    late_success = run_scenario(answer_chunk=18)

    if not early_success or not late_success:
        raise SystemExit(
            "FAIL: progressive batching did not work."
        )

    print(
        "PASS: Qwen stops early and can still reach a late answer."
    )


if __name__ == "__main__":
    main()
