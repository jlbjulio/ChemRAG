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
LORA_INFERENCE_SCALE = 0.1

NO_ANSWER = "I could not find that information in the retrieved sources."
SYSTEM_PROMPT = (
    "You are an organic and inorganic chemistry assistant. Answer in the "
    "same language as the user's question. Use only the retrieved evidence. "
    "Understand the exact request before answering. A simple question should "
    "receive a direct answer; a broad or explicitly detailed question should "
    "receive a complete, well-organized explanation. Include every requested "
    "item and exclude facts that were not requested. Reconcile differing "
    "records when possible and clearly distinguish calculated properties from "
    "reported crystal data. For crystal structure, report a named prototype, "
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
EDITOR_PROMPT = (
    "You are the final editor of a scientific answer. Return only the final "
    "answer in the same language as the question. Check the draft against the "
    "retrieved evidence. Include every item the user requested and remove every "
    "fact the user did not request. Never mention databases, providers, source "
    "names, record IDs, licenses, or retrieval operations. Do not claim that an "
    "unrequested property is missing. If a requested item is absent, identify "
    "that item as unavailable. Preserve units and distinguish calculated values "
    "from reported crystal or experimental data whenever the question requests "
    "that distinction. In that case, explicitly label DFT/PBE-derived values as "
    "calculated and label crystallographic records as reported crystal data; do "
    "not call them experimental unless the evidence says so. Follow the evidence "
    "type labels in the context. A provider name is "
    "not a scientific method. For crystal "
    "structure, prefer an explicit prototype or phase; do not repeat the space "
    "group as the structure when both were requested and a prototype is present. "
    "When the question requests multiple listed items, write exactly one bullet "
    "per requested item, copy its label from the question, and do not add other "
    "bullets. Do not invent facts."
)


class LocalLLM:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.has_adapter = ADAPTER_DIR.exists()
        tokenizer_source = ADAPTER_DIR if self.has_adapter else MODEL_ID

        self.tokenizer: Any = AutoTokenizer.from_pretrained(tokenizer_source)
        base_model: Any = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=dtype,
        )
        self.model: Any

        if self.has_adapter:
            self.model = PeftModel.from_pretrained(
                base_model,
                ADAPTER_DIR,
            )
            self._scale_adapter(LORA_INFERENCE_SCALE)
        else:
            self.model = base_model

        self.model.to(self.device)
        self.model.eval()

    def _scale_adapter(self, factor: float) -> None:
        """Blend LoRA behavior conservatively with Qwen instruction-following."""
        for module in self.model.modules():
            scaling = getattr(module, "scaling", None)

            if not isinstance(scaling, dict):
                continue

            for adapter_name, current_scale in list(scaling.items()):
                if isinstance(current_scale, (int, float)):
                    scaling[adapter_name] = current_scale * factor

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
                    f"Question:\n{question}"
                ),
            },
        ]
        return self._generate_messages(messages)

    def refine(self, question: str, context: str, draft: str) -> str:
        messages = [
            {"role": "system", "content": EDITOR_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Draft answer:\n{draft}\n\n"
                    f"Retrieved evidence:\n{context}\n\n"
                    f"Question:\n{question}\n\n"
                    "Rewrite the final answer using only the evidence."
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


def refine_answer(question: str, context: str, draft: str) -> str:
    return get_local_llm().refine(question, context, draft)
