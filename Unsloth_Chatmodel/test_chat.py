import torch
from unsloth import FastLanguageModel

# Load the fine-tuned LoRA model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="qwen_nutrition_lora",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

# Define your prompt using the same system instruction
messages = [
    {
        "role": "system",
        "content": (
            "You are a friendly and knowledgeable Indian nutrition and fitness assistant. "
            "You give practical, easy-to-follow advice about food, protein, workouts, "
            "Indian meals, and healthy habits. Keep your replies warm, natural and helpful."
        ),
    },
    {
        "role": "user",
        "content": "Can you suggest a high-protein vegetarian Indian diet plan for weight loss?",
    },
]

# Apply chat template
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

# Generate response
outputs = model.generate(
    input_ids=input_ids,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    use_cache=True,
)

response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
print("\n--- Assistant Reply ---")
print(response)
