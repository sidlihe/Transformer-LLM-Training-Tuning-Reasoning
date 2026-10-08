# 🥗 Indian Nutrition & Fitness Chatbot (Qwen-2.5-0.5B Fine-Tuning)

This project documents the end-to-end pipeline for fine-tuning **Qwen2.5-0.5B-Instruct** using **Unsloth (LoRA/QLoRA)** inside WSL2 (Ubuntu), exporting the standalone weights, and running inference/testing on Windows (both GPU & CPU).

---

## 📁 Project Structure

```text
Transformer-LLM-Training-Tuning-Reasoning/
├── Data/
│   ├── tbl_user_chat_202610081150.csv      # Raw database dump
│   └── train_data.json                     # Prepared JSON conversation dataset
├── Transformer_notebook/
│   ├── test_gpu.ipynb                      # GPU inference in Windows (PyTorch/Transformers)                    # CPU inference in Windows
├── Unsloth_Chatmodel/
├── README.md
│   ├── train.py                            # Training script for Unsloth
│   ├── export_model.py                     # Script to merge LoRA into standalone model
│   ├── qwen_nutrition_lora/                # Trained LoRA adapter weights (~40MB)
├── .gitignore
├──  qwen_nutrition_merged_16bit/        # Standalone fused 16-bit model (~1GB)