import json
import sys
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
CASES_PATH = PROJECT_ROOT / "evals" / "generation_cases.json"

sys.path.insert(0, str(SRC_DIR))

import search_chunks  # noqa: E402
from model_runtime import configure_utf8_console  # noqa: E402

search_chunks.DEBUG = False


def normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)

    return "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    ).lower()


def main():
    configure_utf8_console()
    cases = json.loads(
        CASES_PATH.read_text(encoding="utf-8")
    )

    passed = 0

    for case in cases:
        question = case["question"]

        answer, sources = search_chunks.answer_question(
            question
        )

        normalized_answer = normalize(answer)

        if case["should_refuse"]:
            success = normalize(
                search_chunks.NO_ANSWER
            ) in normalized_answer
        else:
            expected_terms = [
                normalize(term)
                for term in case["expected_terms"]
            ]

            success = all(
                term in normalized_answer
                for term in expected_terms
            )

        if success:
            passed += 1
            status = "PASS"
        else:
            status = "FAIL"

        print(f"\n[{status}] {question}")
        print(f"Answer: {answer}")

        if sources:
            print(f"Sources: {', '.join(sources)}")

    total = len(cases)
    accuracy = passed / total

    print("\n----------------------------")
    print(f"Result: {passed}/{total}")
    print(f"Accuracy: {accuracy:.1%}")


if __name__ == "__main__":
    main()
