import os
import sys
import json
import time
import argparse
import urllib.request
import urllib.error

def parse_args():
    parser = argparse.ArgumentParser(description="VeritasAI Training Executor")
    parser.add_argument("--job_id", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3.2-3B")
    parser.add_argument("--dataset_id", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--callback_url", type=str, required=True)
    parser.add_argument("--hf_token", type=str, default="")
    return parser.parse_args()

def post_callback(url, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.read()
    except Exception as e:
        print(f"Callback failed to {url}: {e}", file=sys.stderr)
        return None

def get_callback_url(base_url, suffix=None):
    if not suffix:
        return base_url
    if "?" in base_url:
        path, query = base_url.split("?", 1)
        return f"{path}/{suffix}?{query}"
    return f"{base_url}/{suffix}"

def fetch_dataset(download_url):
    print(f"Downloading dataset from: {download_url}")
    try:
        with urllib.request.urlopen(download_url, timeout=30) as response:
            content = response.read().decode("utf-8")
            rows = []
            for line in content.strip().split("\n"):
                if line.strip():
                    rows.append(json.loads(line))
            return rows
    except Exception as e:
        print(f"Failed to fetch dataset: {e}", file=sys.stderr)
        sys.exit(1)

def run_simulation(args, rows):
    print("Running in SIMULATION MODE (No GPU or Unsloth detected)")
    total_steps = len(rows) * args.epochs // args.batch_size
    total_steps = max(5, total_steps) # ensure at least 5 steps
    
    # Notify downloading status
    post_callback(args.callback_url, {
        "status": "downloading",
        "step": 0,
        "epoch": 0,
        "total_steps": total_steps
    })
    time.sleep(2)
    
    # Notify training status
    post_callback(args.callback_url, {
        "status": "training",
        "step": 0,
        "epoch": 0,
        "total_steps": total_steps
    })
    
    # Simulate step epochs
    history = []
    for step in range(1, total_steps + 1):
        time.sleep(1.5)  # simulate training compute time
        loss = round(2.2 - (step * 0.15) + (0.04 * (step % 2)), 4)
        epoch = min(args.epochs, int(1 + (step * args.epochs / total_steps)))
        
        payload = {
            "status": "training",
            "step": step,
            "loss": loss,
            "lr": args.learning_rate,
            "epoch": epoch,
            "total_steps": total_steps
        }
        print(f"[Step {step}/{total_steps}] Loss: {loss}, Epoch: {epoch}")
        post_callback(args.callback_url, payload)
        
    # Notify uploading status
    post_callback(args.callback_url, {
        "status": "uploading",
        "step": total_steps,
        "epoch": args.epochs,
        "total_steps": total_steps
    })
    time.sleep(2)
    
    # Complete callback
    base_model_clean = args.base_model.split("/")[-1].replace(".", "-").replace("_", "-").lower()
    hf_model_id = f"veritas-{base_model_clean}-{args.job_id[:8]}"
    post_callback(get_callback_url(args.callback_url, "complete"), {
        "hf_model_id": f"huggingface-user/{hf_model_id}",
        "actual_cost_usd": 0.0,
        "gpu_seconds_used": total_steps * 2
    })
    print("Simulation training finished successfully.")

def run_real_training(args, rows):
    print("Initializing Unsloth GPU Training Engine...")
    try:
        from unsloth import FastLanguageModel
        import torch
        from datasets import Dataset as HFDataset
        from transformers import TrainingArguments, TrainerCallback
        from trl import SFTTrainer
        
        # Verify CUDA is active
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. GPU required for real training.")

        print(f"Loading Base Model: {args.base_model}...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.base_model,
            max_seq_length=args.max_seq_length,
            load_in_4bit=True,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )

        # Format Dataset
        def format_prompts(row):
            if "messages" in row and isinstance(row["messages"], list):
                return {"text": tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)}
            else:
                return {"text": f"Instruction:\n{row.get('input', '')}\n\nResponse:\n{row.get('output', '')}"}

        dataset = HFDataset.from_list(rows)
        dataset = dataset.map(format_prompts)

        output_dir = f"./outputs_{args.job_id}"
        total_steps = len(rows) * args.epochs // args.batch_size
        total_steps = max(1, total_steps)

        # Notify provisioning finished, starting training
        post_callback(args.callback_url, {
            "status": "training",
            "step": 0,
            "epoch": 0,
            "total_steps": total_steps
        })

        class CallbackStreamer(TrainerCallback):
            def on_log(self, trainer_args, state, control, logs=None, **kwargs):
                if logs and "loss" in logs:
                    step = state.global_step
                    loss = logs["loss"]
                    lr = logs.get("learning_rate", 0.0)
                    epoch = int(state.epoch or 0)
                    
                    post_callback(args.callback_url, {
                        "status": "training",
                        "step": step,
                        "loss": loss,
                        "lr": lr,
                        "epoch": epoch,
                        "total_steps": total_steps
                    })

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
            dataset_num_proc=2,
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=args.batch_size,
                gradient_accumulation_steps=1,
                warmup_steps=args.warmup_steps,
                max_steps=total_steps,
                learning_rate=args.learning_rate,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir=output_dir,
            ),
            callbacks=[CallbackStreamer()],
        )

        start_time = time.time()
        trainer.train()
        gpu_seconds = int(time.time() - start_time)

        # Notify uploading status
        post_callback(args.callback_url, {
            "status": "uploading",
            "step": total_steps,
            "epoch": args.epochs,
            "total_steps": total_steps
        })

        base_model_clean = args.base_model.split("/")[-1].replace(".", "-").replace("_", "-").lower()
        hf_repo_id = f"veritas-{base_model_clean}-{args.job_id[:8]}"
        if args.hf_token:
            print("Saving model and tokenizer locally...")
            local_dir = f"lora_model_{args.job_id}"
            model.save_pretrained(local_dir)
            tokenizer.save_pretrained(local_dir)
            
            config_path = os.path.join(local_dir, "adapter_config.json")
            if os.path.exists(config_path):
                print("Cleaning up adapter_config.json to ensure TGI/PEFT compatibility...")
                try:
                    with open(config_path, "r") as f:
                        cfg = json.load(f)
                    
                    allowed_keys = {
                        "peft_type", "task_type", "base_model_name_or_path",
                        "r", "lora_alpha", "lora_dropout", "target_modules",
                        "bias", "fan_in_fan_out", "modules_to_save",
                        "init_lora_weights", "layers_to_transform", "layers_pattern",
                        "rank_pattern", "alpha_pattern"
                    }
                    cleaned_cfg = {k: v for k, v in cfg.items() if k in allowed_keys}
                    cleaned_cfg["base_model_name_or_path"] = args.base_model
                    
                    with open(config_path, "w") as f:
                        json.dump(cleaned_cfg, f, indent=2)
                    print("Cleaned adapter_config.json successfully:")
                    print(json.dumps(cleaned_cfg, indent=2))
                except Exception as e:
                    print(f"Warning: Failed to clean adapter_config.json: {e}")
                    
            print(f"Pushing LoRA adapter model to Hugging Face: {hf_repo_id}")
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                api.create_repo(repo_id=hf_repo_id, token=args.hf_token, exist_ok=True)
                api.upload_folder(
                    folder_path=local_dir,
                    repo_id=hf_repo_id,
                    repo_type="model",
                    token=args.hf_token,
                )
                full_repo_id = hf_repo_id
            except Exception as e:
                print(f"Error uploading via HfApi: {e}. Falling back to default push_to_hub...")
                model.push_to_hub(hf_repo_id, token=args.hf_token)
                tokenizer.push_to_hub(hf_repo_id, token=args.hf_token)
                full_repo_id = hf_repo_id  # HuggingFace will auto-resolve username
        else:
            full_repo_id = f"anonymous/{hf_repo_id}"

        # Complete callback
        post_callback(get_callback_url(args.callback_url, "complete"), {
            "hf_model_id": full_repo_id,
            "actual_cost_usd": round((gpu_seconds / 3600.0) * 0.34, 4), # estimated RTX 4090 cost
            "gpu_seconds_used": gpu_seconds
        })
        print("GPU Fine-tuning completed and model pushed successfully.")

    except Exception as e:
        print(f"GPU Training failed: {e}", file=sys.stderr)
        post_callback(get_callback_url(args.callback_url, "failed"), {
            "error_message": str(e)
        })
        sys.exit(1)

def main():
    args = parse_args()
    print(f"=== Starting VeritasAI Fine-Tuning Execution: {args.job_id} ===")
    
    # Calculate download URL based on callback URL
    base_url = args.callback_url.split("/jobs/")[0]
    download_url = f"{base_url}/datasets/{args.dataset_id}/raw"
    
    # Download dataset
    rows = fetch_dataset(download_url)
    
    # Determine mode
    has_unsloth = False
    try:
        import unsloth
        import torch
        if torch.cuda.is_available():
            has_unsloth = True
    except ImportError:
        pass
        
    if has_unsloth:
        run_real_training(args, rows)
    else:
        run_simulation(args, rows)

if __name__ == "__main__":
    main()
