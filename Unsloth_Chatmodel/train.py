import os
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import get_chat_template, train_on_responses_only

# 1. Configuration
max_seq_length = 2048
dtype = None  # None for auto-detection (Float16/Bfloat16)
load_in_4bit = True  # Set to True to save maximum VRAM

# 2. Load Model & Tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-0.5B-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# 3. Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# 4. Apply Qwen-2.5 Chat Template
tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen-2.5",
)

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [
        tokenizer.apply_chat_template(
            convo, 
            tokenize=False, 
            add_generation_prompt=False
        ) 
        for convo in convos
    ]
    return {"text": texts}

# 5. Load and Format Dataset
dataset = load_dataset("json", data_files="data/train_data.json", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

# 6. Trainer Setup
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=1,          # Set to None and use num_train_epochs=1-3 for full training
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)

# 7. Train only on assistant answers (skips masking user/system queries)
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

# 8. Start Training
print("Starting training...")
trainer_stats = trainer.train()

# 9. Save LoRA weights
model.save_pretrained("qwen_nutrition_lora")
tokenizer.save_pretrained("qwen_nutrition_lora")
print("Training completed and model saved to 'qwen_nutrition_lora'!")
