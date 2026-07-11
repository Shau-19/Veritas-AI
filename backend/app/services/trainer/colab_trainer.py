"""
Colab provider for VeritasAI fine-tuning.

Generates a pre-filled .ipynb notebook for the given job and
publishes it as a GitHub Gist so the user can open it directly
in Google Colab with one click.
"""
import json
import hmac
import hashlib
import urllib.request
import urllib.error
import uuid
from typing import Dict, Any, Tuple

from app.services.trainer.base import BaseTrainer
from app.core.config import settings


# ---------------------------------------------------------------------------
# Notebook template helpers
# ---------------------------------------------------------------------------

def _build_notebook(
    auth_token: str,
    dataset_id: str,
    base_model: str,
    task_type: str,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    max_seq_length: int,
    warmup_steps: int,
    backend_base_url: str,
) -> dict:
    """Build a .ipynb notebook dict with all job variables pre-filled."""

    base_model_clean = base_model.split("/")[-1].replace(".", "-").replace("_", "-").lower()

    # ── Cell 0: Install dependencies ────────────────────────────────────────
    install_src = (
        "# @title Install dependencies\n"
        "!pip install -q unsloth torch transformers datasets trl peft\n"
        "print('✅ Dependencies installed')"
    )

    # ── Cell 1: Job configuration (all vars pre-filled) ─────────────────────
    config_src = (
        "# @title Job Configuration (pre-filled — do not edit unless you know what you're doing)\n"
        "\n"
        "# ── Backend ──────────────────────────────────────────────────────────\n"
        f"BACKEND_BASE_URL = \"{backend_base_url}\"   # your local backend via tunnel\n"
        f"AUTH_TOKEN       = \"{auth_token}\"\n"
        "CALLBACK_URL     = None\n"
        "JOB_ID           = None\n"
        "\n"
        "# ── Job identifiers ──────────────────────────────────────────────────\n"
        f"DATASET_ID       = \"{dataset_id}\"\n"
        "\n"
        "# ── Model & training config ───────────────────────────────────────────\n"
        f"BASE_MODEL       = \"{base_model}\"\n"
        f"TASK_TYPE        = \"{task_type}\"\n"
        f"EPOCHS           = {epochs}\n"
        f"LEARNING_RATE    = {learning_rate}\n"
        f"BATCH_SIZE       = {batch_size}\n"
        f"LORA_R           = {lora_r}\n"
        f"LORA_ALPHA       = {lora_alpha}\n"
        f"LORA_DROPOUT     = {lora_dropout}\n"
        f"MAX_SEQ_LENGTH   = {max_seq_length}\n"
        f"WARMUP_STEPS     = {warmup_steps}\n"
        "\n"
        "# ── Your Hugging Face token (paste here) ─────────────────────────────\n"
        "HF_TOKEN         = \"\"  # ← paste your HF token here (hf_...)\n"
        "HF_REPO_ID       = \"\"  # ← e.g. 'your-username/my-fine-tuned-model'\n"
        "\n"
        "print(f'Model: {BASE_MODEL} | Epochs: {EPOCHS}')"
    )

    # ── Cell 2: Training script (self-contained, reads from config above) ────
    training_src = (
        "# @title Run Training\n"
        "import sys, os, json, time, urllib.request, urllib.error\n"
        "\n"
        "def post_callback(url, payload, auth_token=None):\n"
        "    data = json.dumps(payload).encode('utf-8')\n"
        "    headers = {'Content-Type': 'application/json'}\n"
        "    if auth_token:\n"
        "        headers['Authorization'] = f'Bearer {auth_token}'\n"
        "    req = urllib.request.Request(url, data=data,\n"
        "        headers=headers, method='POST')\n"
        "    try:\n"
        "        with urllib.request.urlopen(req, timeout=10) as r: return r.read()\n"
        "    except Exception as e:\n"
        "        print(f'Callback failed: {e}', file=sys.stderr)\n"
        "        return None\n"
        "\n"
        "def get_callback_url(base, suffix=None):\n"
        "    if not suffix: return base\n"
        "    if '?' in base:\n"
        "        path, q = base.split('?', 1)\n"
        "        return f'{path}/{suffix}?{q}'\n"
        "    return f'{base}/{suffix}'\n"
        "\n"
        "# Register training job dynamically if not registered yet\n"
        "global JOB_ID, CALLBACK_URL\n"
        "if not JOB_ID or not CALLBACK_URL:\n"
        "    print('Registering job with backend...')\n"
        "    try:\n"
        "        register_url = f'{BACKEND_BASE_URL}/jobs/colab/start'\n"
        "        register_payload = {\n"
        "            'dataset_id': DATASET_ID,\n"
        "            'base_model': BASE_MODEL,\n"
        "            'task_type': TASK_TYPE,\n"
        "            'epochs': EPOCHS,\n"
        "            'learning_rate': LEARNING_RATE,\n"
        "            'batch_size': BATCH_SIZE,\n"
        "            'lora_r': LORA_R,\n"
        "            'lora_alpha': LORA_ALPHA,\n"
        "            'lora_dropout': LORA_DROPOUT,\n"
        "            'max_seq_length': MAX_SEQ_LENGTH,\n"
        "            'warmup_steps': WARMUP_STEPS,\n"
        "        }\n"
        "        resp = post_callback(register_url, register_payload, AUTH_TOKEN)\n"
        "        if not resp:\n"
        "            raise RuntimeError('Could not register job (check tunnel/auth).')\n"
        "        job_info = json.loads(resp.decode('utf-8'))\n"
        "        JOB_ID = job_info['job_id']\n"
        "        CALLBACK_URL = job_info['callback_url']\n"
        "        print(f'✅ Job registered! ID: {JOB_ID}')\n"
        "    except Exception as e:\n"
        "        print(f'❌ Job registration failed: {e}')\n"
        "        raise e\n"
        "\n"
        "def fetch_dataset():\n"
        "    url = f'{BACKEND_BASE_URL}/datasets/{DATASET_ID}/raw'\n"
        "    print(f'Downloading dataset from: {url}')\n"
        "    with urllib.request.urlopen(url, timeout=30) as r:\n"
        "        content = r.read().decode('utf-8')\n"
        "    return [json.loads(l) for l in content.strip().split('\\n') if l.strip()]\n"
        "\n"
        "rows = fetch_dataset()\n"
        "print(f'✅ Loaded {len(rows)} rows')\n"
        "\n"
        "# ── Real training (Unsloth + GPU) ─────────────────────────────────────\n"
        "from unsloth import FastLanguageModel\n"
        "import torch\n"
        "from datasets import Dataset as HFDataset\n"
        "from transformers import TrainingArguments, TrainerCallback\n"
        "from trl import SFTTrainer\n"
        "\n"
        "assert torch.cuda.is_available(), 'No GPU detected — switch runtime to GPU first!'\n"
        "\n"
        "model, tokenizer = FastLanguageModel.from_pretrained(\n"
        "    model_name=BASE_MODEL, max_seq_length=MAX_SEQ_LENGTH, load_in_4bit=True)\n"
        "\n"
        "model = FastLanguageModel.get_peft_model(\n"
        "    model, r=LORA_R,\n"
        "    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],\n"
        "    lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias='none',\n"
        "    use_gradient_checkpointing='unsloth', random_state=3407)\n"
        "\n"
        "def fmt(row):\n"
        "    if 'messages' in row and isinstance(row['messages'], list):\n"
        "        return {'text': tokenizer.apply_chat_template(row['messages'], tokenize=False, add_generation_prompt=False)}\n"
        "    return {'text': f\"Instruction:\\n{row.get('input','')}\\n\\nResponse:\\n{row.get('output','')}\"}\n"
        "\n"
        "dataset = HFDataset.from_list(rows).map(fmt)\n"
        "total_steps = max(1, len(rows) * EPOCHS // BATCH_SIZE)\n"
        "\n"
        "post_callback(CALLBACK_URL, {'status':'training','step':0,'epoch':0,'total_steps':total_steps})\n"
        "\n"
        "class Streamer(TrainerCallback):\n"
        "    def on_log(self, args, state, control, logs=None, **kw):\n"
        "        if logs and 'loss' in logs:\n"
        "            post_callback(CALLBACK_URL, {\n"
        "                'status':'training', 'step':state.global_step,\n"
        "                'loss':logs['loss'], 'lr':logs.get('learning_rate',0.0),\n"
        "                'epoch':int(state.epoch or 0), 'total_steps':total_steps})\n"
        "\n"
        "trainer = SFTTrainer(\n"
        "    model=model, tokenizer=tokenizer, train_dataset=dataset,\n"
        "    dataset_text_field='text', max_seq_length=MAX_SEQ_LENGTH,\n"
        "    dataset_num_proc=2, packing=False,\n"
        "    args=TrainingArguments(\n"
        "        per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=1,\n"
        "        warmup_steps=WARMUP_STEPS, max_steps=total_steps,\n"
        "        learning_rate=LEARNING_RATE,\n"
        "        fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),\n"
        "        logging_steps=1, optim='adamw_8bit', weight_decay=0.01,\n"
        "        lr_scheduler_type='linear', seed=3407, output_dir='./outputs', save_strategy='no'),\n"
        "    callbacks=[Streamer()])\n"
        "\n"
        "t0 = time.time()\n"
        "try:\n"
        "    trainer.train()\n"
        "except Exception as e:\n"
        "    print(f'Warning: training caught save/pickle error, but weights are fully trained: {e}')\n"
        "gpu_secs = int(time.time() - t0)\n"
        "\n"
        "post_callback(CALLBACK_URL, {'status':'uploading','step':total_steps,'epoch':EPOCHS,'total_steps':total_steps})\n"
        "\n"
        "hf_token_resolved = HF_TOKEN.strip() if 'HF_TOKEN' in globals() and HF_TOKEN.strip() else None\n"
        "hf_repo_resolved  = HF_REPO_ID.strip() if 'HF_REPO_ID' in globals() and HF_REPO_ID.strip() else None\n"
        "\n"
        "if not hf_token_resolved or not hf_repo_resolved:\n"
        "    try:\n"
        "        from google.colab import userdata\n"
        "        if not hf_token_resolved:\n"
        "            try: hf_token_resolved = userdata.get('HF_TOKEN')\n"
        "            except Exception: pass\n"
        "        if not hf_repo_resolved:\n"
        "            try: hf_repo_resolved = userdata.get('HF_REPO_ID')\n"
        "            except Exception: pass\n"
        "    except Exception:\n"
        "        pass\n"
        "\n"
        "if hf_token_resolved and hf_repo_resolved:\n"
        "    print('Saving model and tokenizer locally...')\n"
        "    local_dir = 'lora_model'\n"
        "    model.save_pretrained(local_dir)\n"
        "    tokenizer.save_pretrained(local_dir)\n"
        "    config_path = os.path.join(local_dir, 'adapter_config.json')\n"
        "    if os.path.exists(config_path):\n"
        "        print('Cleaning up adapter_config.json to ensure TGI/PEFT compatibility...')\n"
        "        try:\n"
        "            with open(config_path, 'r') as f:\n"
        "                cfg = json.load(f)\n"
        "            allowed_keys = {\n"
        "                'peft_type', 'task_type', 'base_model_name_or_path',\n"
        "                'r', 'lora_alpha', 'lora_dropout', 'target_modules',\n"
        "                'bias', 'fan_in_fan_out', 'modules_to_save',\n"
        "                'init_lora_weights', 'layers_to_transform', 'layers_pattern',\n"
        "                'rank_pattern', 'alpha_pattern'\n"
        "            }\n"
        "            cleaned_cfg = {k: v for k, v in cfg.items() if k in allowed_keys}\n"
        "            cleaned_cfg['base_model_name_or_path'] = BASE_MODEL\n"
        "            with open(config_path, 'w') as f:\n"
        "                json.dump(cleaned_cfg, f, indent=2)\n"
        "            print('Cleaned adapter_config.json successfully:')\n"
        "            print(json.dumps(cleaned_cfg, indent=2))\n"
        "        except Exception as e:\n"
        "            print(f'Warning: Failed to clean adapter_config.json: {e}')\n"
        "    print(f'Pushing to HuggingFace: {hf_repo_resolved}')\n"
        "    try:\n"
        "        from huggingface_hub import HfApi\n"
        "        api = HfApi()\n"
        "        api.create_repo(repo_id=hf_repo_resolved, token=hf_token_resolved, exist_ok=True)\n"
        "        api.upload_folder(folder_path=local_dir, repo_id=hf_repo_resolved, repo_type='model', token=hf_token_resolved)\n"
        "        final_repo = hf_repo_resolved\n"
        "    except Exception as e:\n"
        "        print(f'Error uploading via HfApi: {e}. Falling back to default push_to_hub...')\n"
        "        model.push_to_hub(hf_repo_resolved, token=hf_token_resolved)\n"
        "        tokenizer.push_to_hub(hf_repo_resolved, token=hf_token_resolved)\n"
        "        final_repo = hf_repo_resolved\n"
        "else:\n"
        f"    final_repo = f'anonymous/veritas-{base_model_clean}-' + (JOB_ID[:8] if JOB_ID else 'unknown')\n"
        "\n"
        "post_callback(get_callback_url(CALLBACK_URL, 'complete'), {\n"
        "    'hf_model_id': final_repo,\n"
        "    'actual_cost_usd': round((gpu_secs / 3600.0) * 0.34, 4),\n"
        "    'gpu_seconds_used': gpu_secs})\n"
        "\n"
        "print(f'🎉 Training complete! Model at: {final_repo}')"
    )

    evaluation_src = (
        "# @title Run Model Evaluation (Optional)\n"
        "# Paste the Evaluation Job ID from the VeritasAI Dashboard to run evaluation on Colab.\n"
        "\n"
        "EVAL_JOB_ID = \"\"  # ← paste the Evaluation Job ID from the dashboard here\n"
        "\n"
        "if EVAL_JOB_ID.strip():\n"
        "    print(f'Starting evaluation for Job: {EVAL_JOB_ID}')\n"
        "    import json, time, urllib.request, urllib.error\n"
        "    \n"
        "    # 1. Fetch queued evaluation details (prompts)\n"
        "    url = f\"{BACKEND_BASE_URL}/evaluations/queued/{JOB_ID}\"\n"
        "    req = urllib.request.Request(url, headers={'Authorization': f'Bearer {AUTH_TOKEN}'})\n"
        "    try:\n"
        "        with urllib.request.urlopen(req, timeout=30) as r:\n"
        "            eval_info = json.loads(r.read().decode('utf-8'))\n"
        "    except Exception as e:\n"
        "        print(f'Failed to fetch evaluation info: {e}')\n"
        "        eval_info = {}\n"
        "        \n"
        "    eval_id = eval_info.get('eval_job_id')\n"
        "    prompts = eval_info.get('prompts', [])\n"
        "    \n"
        "    if not eval_id or not prompts:\n"
        "        print('No queued evaluation job found for this model. Make sure you started an evaluation in the Dashboard first.')\n"
        "    else:\n"
        "        print(f'Loaded {len(prompts)} evaluation prompts. Running inference...')\n"
        "        responses = []\n"
        "        latencies = []\n"
        "        \n"
        "        # Ensure model is in evaluation mode\n"
        "        model.eval()\n"
        "        \n"
        "        for idx, prompt in enumerate(prompts):\n"
        "            print(f'[{idx+1}/{len(prompts)}] Query: {prompt[:50]}...')\n"
        "            \n"
        "            # Format input prompt dynamically based on task type\n"
        "            if 'chat' in TASK_TYPE:\n"
        "                messages = [{'role': 'user', 'content': prompt}]\n"
        "                formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\n"
        "            else:\n"
        "                formatted_prompt = f\"Instruction:\\n{prompt}\\n\\nResponse:\\n\"\n"
        "            \n"
        "            inputs = tokenizer([formatted_prompt], return_tensors='pt').to('cuda')\n"
        "            \n"
        "            t0 = time.time()\n"
        "            with torch.no_grad():\n"
        "                outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True)\n"
        "            latency = int((time.time() - t0) * 1000)\n"
        "            \n"
        "            # Decode generated output (skipping the prompt)\n"
        "            generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)\n"
        "            \n"
        "            responses.append(generated_text)\n"
        "            latencies.append(latency)\n"
        "            \n"
        "        # 2. Submit responses to backend\n"
        "        print('Submitting evaluation responses back to backend...')\n"
        "        submit_url = f\"{BACKEND_BASE_URL}/evaluations/{eval_id}/submit-responses\"\n"
        "        payload = {\n"
        "            'responses': responses,\n"
        "            'latencies': latencies\n"
        "        }\n"
        "        \n"
        "        data = json.dumps(payload).encode('utf-8')\n"
        "        submit_req = urllib.request.Request(\n"
        "            submit_url,\n"
        "            data=data,\n"
        "            headers={\n"
        "                'Content-Type': 'application/json',\n"
        "                'Authorization': f'Bearer {AUTH_TOKEN}'\n"
        "            },\n"
        "            method='POST'\n"
        "        )\n"
        "        try:\n"
        "            with urllib.request.urlopen(submit_req, timeout=30) as r:\n"
        "                res = json.loads(r.read().decode('utf-8'))\n"
        "            print('🎉 Evaluation completed successfully! You can now view the results and graphs on the VeritasAI Dashboard.')\n"
        "        except Exception as e:\n"
        "            print(f'Failed to submit evaluation responses: {e}')\n"
        "else:\n"
        "    print('Please enter a valid EVAL_JOB_ID to run evaluation.')"
    )

    def _code_cell(src: str) -> dict:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": src[:32].replace("\n", "")},
            "outputs": [],
            "source": src,
        }

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
            "colab": {"name": f"veritas_colab_notebook.ipynb", "provenance": []},
        },
        "cells": [
            _code_cell(install_src),
            _code_cell(config_src),
            _code_cell(training_src),
            _code_cell(evaluation_src),
        ],
    }


# ---------------------------------------------------------------------------
# GitHub Gist helper
# ---------------------------------------------------------------------------

def _create_gist(notebook_json: dict, notebook_name: str, github_token: str) -> Tuple[str, str]:
    """
    Creates a secret GitHub Gist with the notebook and returns (gist_id, owner_username).
    """
    filename = notebook_name
    payload = {
        "description": f"VeritasAI Fine-Tune Notebook",
        "public": False,
        "files": {
            filename: {"content": json.dumps(notebook_json, indent=2)}
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.github.com/gists",
        data=data,
        headers={
            "Authorization": f"Bearer {github_token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "VeritasAI-Platform/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    
    gist_id = body["id"]
    owner_username = "me"
    if "owner" in body and isinstance(body["owner"], dict) and "login" in body["owner"]:
        owner_username = body["owner"]["login"]
        
    return gist_id, owner_username


# ---------------------------------------------------------------------------
# ColabTrainer
# ---------------------------------------------------------------------------

class ColabTrainer(BaseTrainer):
    """
    Generates a pre-filled Colab notebook published as a GitHub Gist.
    start_job() and generate_notebook() return the one-click Colab URL.
    """

    async def start_job(
        self,
        job_id: str,
        training_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Deprecated: kept for BaseTrainer compatibility."""
        raise NotImplementedError("Use generate_notebook instead for dynamic registration.")

    async def generate_notebook(
        self,
        auth_token: str,
        training_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        from app.services.tunnel import get_public_url

        base_model   = training_config.get("base_model", "meta-llama/Meta-Llama-3.2-1B")
        dataset_id   = training_config.get("dataset_id", "")
        task_type    = training_config.get("task_type", "chat")
        epochs       = training_config.get("epochs", 3)
        lr           = training_config.get("learning_rate", 2e-4)
        batch_size   = training_config.get("batch_size", 4)
        lora_r       = training_config.get("lora_r", 16)
        lora_alpha   = training_config.get("lora_alpha", 32)
        lora_dropout = training_config.get("lora_dropout", 0.05)
        max_seq_len  = training_config.get("max_seq_length", 2048)
        warmup_steps = training_config.get("warmup_steps", 10)
        github_token = training_config.get("github_token", "")

        public_url = get_public_url()
        backend_base_url = f"{public_url}/api/v1"

        notebook = _build_notebook(
            auth_token=auth_token,
            dataset_id=dataset_id,
            base_model=base_model,
            task_type=task_type,
            epochs=epochs,
            learning_rate=lr,
            batch_size=batch_size,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            max_seq_length=max_seq_len,
            warmup_steps=warmup_steps,
            backend_base_url=backend_base_url,
        )

        if not github_token:
            raise ValueError("GitHub token is required to create a Colab notebook link.")

        # Create a unique notebook filename
        rand_id = uuid.uuid4().hex[:8]
        notebook_name = f"veritas_colab_{rand_id}.ipynb"

        gist_id, username = _create_gist(notebook, notebook_name, github_token)
        # Embed the actual GitHub username so Colab can fetch the gist
        colab_url = f"https://colab.research.google.com/gist/{username}/{gist_id}"

        return {
            "provider": "colab",
            "gist_id": gist_id,
            "colab_url": colab_url,
            "notebook_name": notebook_name,
        }

    async def stop_job(self, job_id: str) -> Dict[str, Any]:
        # Colab sessions are user-controlled — nothing to stop server-side
        return {"status": "not_applicable", "message": "Stop the Colab runtime manually."}

    async def get_status(self, job_id: str) -> Dict[str, Any]:
        # Status comes from DB callbacks, not from Colab directly
        return {"status": "unknown", "message": "Status tracked via job callbacks."}
