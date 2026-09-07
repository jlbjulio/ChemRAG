import logging
import warnings
from pathlib import Path
from typing import cast

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings(
    "ignore",
    message="You are sending unauthenticated requests to the HF Hub.*",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "training" / "data"
ADAPTER_DIR = (
    PROJECT_ROOT / "models" / "qwen3-0.6b-chemistry-lora"
)
MODEL_ID = "Qwen/Qwen3-0.6B"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch did not detect an NVIDIA GPU. "
            "Check the CUDA installation before training."
        )

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Base model: {MODEL_ID}")

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(DATA_DIR / "train.jsonl"),
            "validation": str(DATA_DIR / "validation.jsonl"),
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    training_config = SFTConfig(
        output_dir=str(
            PROJECT_ROOT / "outputs" / "qwen3-chemistry"
        ),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=25,
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=512,
        completion_only_loss=True,
        optim="adamw_torch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    if trainer.model is None:
        raise RuntimeError(
            "SFTTrainer did not initialize the model."
        )

    peft_model = cast(PeftModel, trainer.model)
    peft_model.print_trainable_parameters()
    trainer.train()

    trainer.save_model(str(ADAPTER_DIR))
    tokenizer.save_pretrained(ADAPTER_DIR)

    print(f"LoRA adapter saved to: {ADAPTER_DIR}")


if __name__ == "__main__":
    main()
