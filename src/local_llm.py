from pathlib import Path
from typing import Any

from model_runtime import configure_quiet_model_loading


configure_quiet_model_loading()

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADAPTER_DIR = (
    PROJECT_ROOT / "models" / "qwen3-0.6b-chemistry-lora"
)
MODEL_ID = "Qwen/Qwen3-0.6B"

NO_ANSWER = "I could not find that information in the retrieved sources."
SYSTEM_PROMPT = (
    "You are an organic and inorganic chemistry assistant. Answer in the "
    "same language as the user's question. Use only the retrieved evidence. "
    "Understand the exact request before answering. A simple question should "
    "receive a direct answer; a broad or explicitly detailed question should "
    "receive a complete, well-organized explanation. Include every requested "
    "item and exclude facts that were not requested. Reconcile differing "
    "records when possible and clearly distinguish calculated properties from "
    "reported crystal data. When multiple properties are requested, write one "
    "bullet per requested property, preserve the user's labels, and do not add "
    "unrequested properties. For crystal structure, report a named prototype, "
    "phase, or symmetry description when available; do not substitute lattice "
    "parameters or unit-cell volume unless the user asks for them. If only "
    "some requested items are available, answer those and explicitly identify "
    "each unavailable item. Use readable numeric precision without changing "
    "the scientific meaning. Do not mention databases, sources, record IDs, "
    "licenses, retrieval details, or internal field names in the answer. Never "
    "infer a missing property; state clearly which requested information is "
    "not available in the evidence. If the answer is entirely absent, reply "
    "exactly: "
    f"{NO_ANSWER}"
)
class LocalLLM:
    def __init__(self) -> None:
        if not ADAPTER_DIR.is_dir():
            raise FileNotFoundError(
                "The chemistry LoRA adapter was not found at "
                f"{ADAPTER_DIR}. Run training/train_lora.py first."
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.tokenizer: Any = AutoTokenizer.from_pretrained(ADAPTER_DIR)
        base_model: Any = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=dtype,
        )
        self.model: Any = PeftModel.from_pretrained(
            base_model,
            ADAPTER_DIR,
        )

        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def _generate_messages(self, messages: list[dict[str, str]]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated_tokens = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

    def generate(self, question: str, context: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Retrieved context:\n{context}\n\n"
                    f"Question:\n{question}\n\n"
                    "Final-answer contract:\n"
                    "- Answer only what the question explicitly requests.\n"
                    "- Never include an unrequested property, even when it "
                    "appears in the context.\n"
                    "- For a multi-property question, use exactly one bullet "
                    "for each requested property and no other bullets.\n"
                    "- Mark a requested property as unavailable when the "
                    "context does not contain it.\n"
                    "- If the question asks to distinguish evidence types, "
                    "label each value as calculated, reported crystal data, "
                    "or experimental only when the context supports that "
                    "label.\n"
                    "- Return only the answer, without sources or retrieval "
                    "commentary."
                ),
            },
        ]
        return self._generate_messages(messages)

    def generate_without_adapter(
        self,
        question: str,
        context: str,
    ) -> str:
        if isinstance(self.model, PeftModel):
            with self.model.disable_adapter():
                return self.generate(question, context)

        return self.generate(question, context)


_local_llm: LocalLLM | None = None


def get_local_llm() -> LocalLLM:
    global _local_llm

    if _local_llm is None:
        _local_llm = LocalLLM()

    return _local_llm


def generate_answer(question: str, context: str) -> str:
    return get_local_llm().generate(question, context)


def generate_base_answer(question: str, context: str) -> str:
    return get_local_llm().generate_without_adapter(question, context)
