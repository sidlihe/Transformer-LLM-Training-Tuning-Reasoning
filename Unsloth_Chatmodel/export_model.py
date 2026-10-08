from unsloth import FastLanguageModel

print("Loading saved LoRA model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="qwen_nutrition_lora",  # loads your saved checkpoint
    max_seq_length=2048,
    load_in_4bit=True,
)

print("Merging LoRA with base model and saving 16-bit standalone model...")
model.save_pretrained_merged(
    "qwen_nutrition_merged_16bit",
    tokenizer,
    save_method="merged_16bit",
)

print("Successfully merged and saved to folder: qwen_nutrition_merged_16bit!")
