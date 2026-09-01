import json
import sys
import unicodedata
from pathlib import Path
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
CASES_PATH = PROJECT_ROOT / "evals" / "finetuning_cases.json"

sys.path.insert(0, str(SRC_DIR))

from local_llm import (  # noqa: E402
    NO_ANSWER,
    generate_answer,
    generate_base_answer,
    get_local_llm,
)
from model_runtime import configure_utf8_console  # noqa: E402


def normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)

    return "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    ).casefold().strip(" .\n\t")


def evaluate_answer(answer: str, case: dict) -> bool:
    normalized_answer = normalize(answer)

    if case["should_refuse"]:
        refusal_phrases = (
            NO_ANSWER,
            "not available in the retrieved evidence",
            "no esta disponible en la evidencia recuperada",
        )
        return any(
            normalize(phrase) in normalized_answer
            for phrase in refusal_phrases
        )

    expected_present = all(
        normalize(term) in normalized_answer
        for term in case["expected_terms"]
    )
    forbidden_terms = [
        "PubChem",
        "ChEMBL",
        "OQMD",
        "COD",
        "C2DB",
        "RDKit",
        "source identifier",
        *case.get("forbidden_terms", []),
    ]
    forbidden_absent = all(
        normalize(term) not in normalized_answer
        for term in forbidden_terms
    )
    return expected_present and forbidden_absent


def main() -> None:
    configure_utf8_console()
    cases = json.loads(
        CASES_PATH.read_text(encoding="utf-8")
    )

    print("Loading one shared Qwen instance...")
    get_local_llm()

    base_passed = 0
    lora_passed = 0
    base_time = 0.0
    lora_time = 0.0

    for number, case in enumerate(cases, start=1):
        base_started = perf_counter()
        base_answer = generate_base_answer(
            case["question"],
            case["context"],
        )
        base_time += perf_counter() - base_started

        lora_started = perf_counter()
        lora_answer = generate_answer(
            case["question"],
            case["context"],
        )
        lora_time += perf_counter() - lora_started

        base_success = evaluate_answer(base_answer, case)
        lora_success = evaluate_answer(lora_answer, case)

        base_passed += int(base_success)
        lora_passed += int(lora_success)

        print(f"\nCase {number}: {case['question']}")
        print(
            f"Base [{'PASS' if base_success else 'FAIL'}]: "
            f"{base_answer}"
        )
        print(
            f"LoRA [{'PASS' if lora_success else 'FAIL'}]: "
            f"{lora_answer}"
        )

    total = len(cases)

    print("\n----------------------------")
    print(
        f"Qwen base: {base_passed}/{total} "
        f"({base_passed / total:.1%})"
    )
    print(
        f"Qwen + LoRA: {lora_passed}/{total} "
        f"({lora_passed / total:.1%})"
    )
    print(f"Base time: {base_time:.2f} s")
    print(f"LoRA time: {lora_time:.2f} s")
    print(
        "LoRA difference: "
        f"{lora_passed - base_passed:+d} case(s)"
    )


if __name__ == "__main__":
    main()
