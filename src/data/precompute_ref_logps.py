"""
Precompute frozen reference policy (B0) log-probabilities for DPO training dataset.
Saves ref_chosen_logp and ref_rejected_logp directly into e_pref_train_with_ref.jsonl.
This completely eliminates dynamic adapter toggling during training, guaranteeing 100% hardware stability.
"""
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
import sys
import json
import torch
torch.set_num_threads(2)
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from configs.base_config import MODEL_ID, TORCH_DTYPE, ATTN_IMPLEMENTATION, OUTPUT_DATA_DIR


def compute_sequence_logp(model, tokenizer, prompt: str, completion: str) -> float:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    input_ids = torch.tensor([prompt_ids + completion_ids], device=model.device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    shift_logits = logits[0, :-1, :]
    shift_labels = input_ids[0, 1:]
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss_per_token = loss_fct(shift_logits, shift_labels)

    prompt_len = len(prompt_ids)
    completion_token_losses = loss_per_token[prompt_len - 1:]
    total_logp = -completion_token_losses.sum().item()
    return total_logp


def precompute_reference_logps(
    input_path: str = None,
    output_path: str = None,
    max_samples: int = None,
    overwrite: bool = False
):
    if input_path is None:
        input_path = os.path.join(OUTPUT_DATA_DIR, "e_pref_train.jsonl")
    if output_path is None:
        output_path = os.path.join(OUTPUT_DATA_DIR, "e_pref_train_with_ref.jsonl")

    print(f"Loading input preference dataset: {input_path}")
    dataset = load_dataset("json", data_files=input_path, split="train")
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    if os.path.exists(output_path) and not overwrite:
        # Check if complete
        with open(output_path, "r", encoding="utf-8") as f:
            existing_lines = sum(1 for _ in f)
        if existing_lines >= len(dataset):
            print(f"Precomputed reference logps already complete ({existing_lines} pairs) at: {output_path}")
            return output_path
        else:
            print(f"Existing file has {existing_lines} pairs, but expected {len(dataset)}. Recomputing...")

    print(f"Loading reference model: {MODEL_ID} in {TORCH_DTYPE} with {ATTN_IMPLEMENTATION} on CUDA...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=TORCH_DTYPE,
        attn_implementation=ATTN_IMPLEMENTATION
    ).to("cuda")
    model.eval()

    tmp_path = output_path + ".tmp"
    completed_records = []
    if os.path.exists(tmp_path):
        with open(tmp_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    completed_records.append(json.loads(line))
        print(f"Resuming from {len(completed_records)} already computed pairs in {tmp_path}...")

    start_idx = len(completed_records)
    print(f"Computing reference log-probabilities for remaining {len(dataset) - start_idx} pairs...")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(tmp_path, "a", encoding="utf-8") as f_out:
        for idx in tqdm(range(start_idx, len(dataset)), desc="Precomputing Ref Logps"):
            item = dataset[idx]
            prompt = item["prompt"]
            chosen = item["chosen"]
            rejected = item["rejected"]

            ref_chosen_logp = compute_sequence_logp(model, tokenizer, prompt, chosen)
            ref_rejected_logp = compute_sequence_logp(model, tokenizer, prompt, rejected)

            record = dict(item)
            record["ref_chosen_logp"] = ref_chosen_logp
            record["ref_rejected_logp"] = ref_rejected_logp

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()

    # Move tmp to final destination
    if os.path.exists(output_path):
        os.remove(output_path)
    os.replace(tmp_path, output_path)

    print(f"Successfully saved {len(dataset)} pairs with reference logps to: {output_path}")
    return output_path


if __name__ == "__main__":
    precompute_reference_logps()
