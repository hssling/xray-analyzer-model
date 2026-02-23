import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import os
from huggingface_hub import login

# 1. Configuration targeting Chest X-Rays
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct" 
DATASET_ID = "hssling/Chest-XRay-10k-Control"
OUTPUT_DIR = "./omni-xray-adapter"
HF_HUB_REPO = "hssling/omni-xray-adapter" 

def main():
    # Attempt to authenticate with Hugging Face via Kaggle Secrets
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        hf_token = user_secrets.get_secret("HF_TOKEN")
        login(token=hf_token)
        print("Successfully logged into Hugging Face Hub using Kaggle Secrets.")
    except Exception as e:
        print("Could not log in via Kaggle Secrets.", e)

    print(f"Loading processor and model: {MODEL_ID}")
    
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    
    # 4-bit Quantization
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    print("Applying LoRA parameters...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    
    print(f"Loading dataset: {DATASET_ID}")
    dataset = load_dataset(DATASET_ID, split="train[:1000]") # Using subset of 10k for speed
    
    def format_data(example):
        findings = example.get("findings") or example.get("text") or example.get("description") or "Chest X-Ray findings."
        messages = [
            {
                "role": "system",
                "content": "You are Omni-XRay AI, a highly advanced radiologist. Analyze the provided radiograph."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Analyze this Chest X-Ray and provide the clinical findings in a structured format."}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": str(findings)}
                ]
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text, "image": example["image"]}
    
    formatted_dataset = dataset.map(format_data, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=150, 
        save_strategy="steps",
        save_steps=50,
        fp16=True,
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        report_to="none"
    )

    def collate_fn(examples):
        texts = [ex["text"] for ex in examples]
        images = [ex["image"] for ex in examples]
        batch = processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt"
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch

    print("Starting fine-tuning...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        data_collator=collate_fn
    )

    trainer.train()
    
    print(f"Saving fine-tuned adapter to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    
    print(f"Pushing model weights to Hugging Face Hub: {HF_HUB_REPO}...")
    try:
        trainer.model.push_to_hub(HF_HUB_REPO)
        processor.push_to_hub(HF_HUB_REPO)
        print(f"✅ Success! Your model is now live at: https://huggingface.co/{HF_HUB_REPO}")
    except Exception as e:
        print(f"❌ Failed to push to Hugging Face Hub. Error: {e}")

if __name__ == "__main__":
    main()
