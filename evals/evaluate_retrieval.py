import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
CASES_PATH = PROJECT_ROOT / "evals" / "retrieval_cases.json"

sys.path.insert(0, str(SRC_DIR))

from search_chunks import (  # noqa: E402
    MIN_SIMILARITY,
    retrieve_chunks,
)
from model_runtime import configure_utf8_console  # noqa: E402


def main():
    configure_utf8_console()
    cases = json.loads(
        CASES_PATH.read_text(encoding="utf-8")
    )

    passed = 0

    for case in cases:
        question = case["question"]
        results = retrieve_chunks(question)
        should_find = case["should_find"]

        if should_find:
            expected_text = case["expected_text"].lower()

            found_expected_text = any(
                expected_text in item["text"].lower()
                for _, item in results
            )

            success = found_expected_text
        else:
            success = not results

        if success:
            passed += 1
            status = "PASS"
        else:
            status = "FAIL"

        print(f"\n[{status}] {question}")

        if results:
            best_score, best_item = results[0]
            title = best_item["text"].splitlines()[0]

            print(f"Best result: {title}")
            print(f"Retrieval score: {best_score:.4f}")
        else:
            print(
                "No result met the local similarity threshold of "
                f"{MIN_SIMILARITY:.2f}"
            )

    total = len(cases)
    accuracy = passed / total

    print("\n----------------------------")
    print(f"Result: {passed}/{total}")
    print(f"Accuracy: {accuracy:.1%}")


if __name__ == "__main__":
    main()
