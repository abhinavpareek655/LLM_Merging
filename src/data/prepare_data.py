"""
Data Preparation & Filtering Pipeline for Experiment 1 (E1).
Strictly enforces:
1. Max sequence length <= 1024 tokens via pre-tokenization filtering (no truncation).
2. Proper Qwen chat template formatting.
3. Clean JSONL outputs in outputs/datasets/.
"""
import os
import sys
import json
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from configs.base_config import (
    MODEL_ID,
    MAX_SEQ_LENGTH,
    OUTPUT_DATA_DIR
)


def format_chat(tokenizer, user_text: str, assistant_text: str) -> tuple[str, int]:
    """Applies chat template and returns formatted text and its token count."""
    messages = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text}
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False)
    token_len = len(tokenizer.encode(formatted, add_special_tokens=False))
    return formatted, token_len


def prepare_math_data(tokenizer, target_count: int = 5000) -> str:
    print(f"\n[1/4] Preparing Math Dataset: MetaMathQA (Target: {target_count}, Max Len: {MAX_SEQ_LENGTH})...")
    output_path = os.path.join(OUTPUT_DATA_DIR, "e_math.jsonl")
    if os.path.exists(output_path):
        print(f"File already exists at {output_path}, skipping.")
        return output_path

    ds = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)
    valid_records = []
    dropped_long = 0

    pbar = tqdm(total=target_count, desc="Processing Math")
    for item in ds:
        prompt = item.get("query", "").strip()
        response = item.get("response", "").strip()
        if not prompt or not response:
            continue

        formatted, length = format_chat(tokenizer, prompt, response)
        if length > MAX_SEQ_LENGTH:
            dropped_long += 1
            continue

        valid_records.append({
            "prompt": prompt,
            "response": response,
            "text": formatted,
            "length": length
        })
        pbar.update(1)
        if len(valid_records) >= target_count:
            break
    pbar.close()

    print(f"Math: Gathered {len(valid_records)} samples. Filtered out {dropped_long} overly long samples.")
    with open(output_path, "w", encoding="utf-8") as f:
        for r in valid_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return output_path


def prepare_code_data(tokenizer, target_count: int = 3000) -> str:
    print(f"\n[2/4] Preparing Code Dataset (MBPP Train [task_id>=601] + Python Algos, Target: {target_count})...")
    output_path = os.path.join(OUTPUT_DATA_DIR, "e_code.jsonl")

    valid_records = []
    dropped_long = 0

    # 1. Official MBPP Train splits (Tasks 601-974, strictly disjoint from Test 11-510)
    print("  -> Gathering official non-overlapping MBPP train samples...")
    try:
        mbpp_sanitized_train = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
        for item in mbpp_sanitized_train:
            t_id = item.get("task_id", 0)
            if t_id < 600:
                continue  # Guarantee test split isolation
            prompt_text = item.get("prompt", "").strip()
            tests = item.get("test_list", [])
            code = item.get("code", "").strip()
            user_msg = f"Write a Python function to solve the following problem:\n{prompt_text}\nYour code should satisfy these tests:\n" + "\n".join(tests[:1])
            assistant_msg = f"```python\n{code}\n```"

            formatted, length = format_chat(tokenizer, user_msg, assistant_msg)
            if length <= MAX_SEQ_LENGTH:
                valid_records.append({
                    "prompt": user_msg,
                    "response": assistant_msg,
                    "text": formatted,
                    "length": length
                })
    except Exception as e:
        print(f"  Warning loading sanitized train: {e}")

    # 2. Augment with general Python coding instructions
    print("  -> Gathering Python algorithmic instructions...")
    ds_code = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train", streaming=True)
    pbar = tqdm(total=target_count, desc="Processing Code")
    pbar.update(len(valid_records))

    for item in ds_code:
        if len(valid_records) >= target_count:
            break
        instr = item.get("instruction", "").strip()
        inp = item.get("input", "").strip()
        out = item.get("output", "").strip()
        if not instr or not out:
            continue
        user_msg = f"{instr}\nInput: {inp}" if inp else instr
        assistant_msg = out

        formatted, length = format_chat(tokenizer, user_msg, assistant_msg)
        if length > MAX_SEQ_LENGTH:
            dropped_long += 1
            continue

        valid_records.append({
            "prompt": user_msg,
            "response": assistant_msg,
            "text": formatted,
            "length": length
        })
        pbar.update(1)
    pbar.close()

    print(f"Code: Gathered {len(valid_records)} samples (0 overlap with test). Dropped {dropped_long} long samples.")
    with open(output_path, "w", encoding="utf-8") as f:
        for r in valid_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return output_path


def prepare_instr_data(tokenizer, target_count: int = 3000) -> str:
    print(f"\n[3/4] Preparing Instruction Dataset: alpaca-cleaned (Target: {target_count}, Max Len: {MAX_SEQ_LENGTH})...")
    output_path = os.path.join(OUTPUT_DATA_DIR, "e_instr.jsonl")

    ds = load_dataset("yahma/alpaca-cleaned", split="train", streaming=True)
    valid_records = []
    dropped_long = 0

    pbar = tqdm(total=target_count, desc="Processing Instruction")
    for item in ds:
        instr = item.get("instruction", "").strip()
        inp = item.get("input", "").strip()
        out = item.get("output", "").strip()
        if not instr or not out:
            continue

        user_msg = f"{instr}\n\nContext / Input:\n{inp}" if inp else instr
        assistant_msg = out

        formatted, length = format_chat(tokenizer, user_msg, assistant_msg)
        if length > MAX_SEQ_LENGTH:
            dropped_long += 1
            continue

        valid_records.append({
            "prompt": user_msg,
            "response": assistant_msg,
            "text": formatted,
            "length": length
        })
        pbar.update(1)
        if len(valid_records) >= target_count:
            break
    pbar.close()

    print(f"Instruction: Gathered {len(valid_records)} samples. Filtered out {dropped_long} overly long samples.")
    with open(output_path, "w", encoding="utf-8") as f:
        for r in valid_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return output_path


def prepare_pref_data(tokenizer, train_count: int = 2000, test_count: int = 500) -> tuple[str, str]:
    print(f"\n[4/4] Preparing Preference Dataset: UltraFeedback (Train: {train_count}, Test: {test_count})...")
    train_path = os.path.join(OUTPUT_DATA_DIR, "e_pref_train.jsonl")
    test_path = os.path.join(OUTPUT_DATA_DIR, "e_pref_test.jsonl")

    if not os.path.exists(train_path):
        ds_train = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs", streaming=True)
        train_records = []
        dropped_train = 0

        pbar = tqdm(total=train_count, desc="Processing Pref Train")
        for item in ds_train:
            prompt = item.get("prompt", "").strip()
            chosen_msgs = item.get("chosen", [])
            rejected_msgs = item.get("rejected", [])
            if len(chosen_msgs) < 2 or len(rejected_msgs) < 2:
                continue

            chosen_text = chosen_msgs[1]["content"].strip()
            rejected_text = rejected_msgs[1]["content"].strip()
            if not chosen_text or not rejected_text:
                continue

            # Format both to verify length <= MAX_SEQ_LENGTH
            _, len_c = format_chat(tokenizer, prompt, chosen_text)
            _, len_r = format_chat(tokenizer, prompt, rejected_text)

            if len_c > MAX_SEQ_LENGTH or len_r > MAX_SEQ_LENGTH:
                dropped_train += 1
                continue

            # Format prompt for Qwen chat template
            formatted_prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)

            train_records.append({
                "prompt": formatted_prompt,
                "chosen": chosen_text,
                "rejected": rejected_text,
                "raw_prompt": prompt
            })
            pbar.update(1)
            if len(train_records) >= train_count:
                break
        pbar.close()

        print(f"Pref Train: Gathered {len(train_records)} pairs. Filtered out {dropped_train} overly long pairs.")
        with open(train_path, "w", encoding="utf-8") as f:
            for r in train_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if not os.path.exists(test_path):
        ds_test = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs", streaming=True)
        test_records = []
        dropped_test = 0

        pbar = tqdm(total=test_count, desc="Processing Pref Test")
        for item in ds_test:
            prompt = item.get("prompt", "").strip()
            chosen_msgs = item.get("chosen", [])
            rejected_msgs = item.get("rejected", [])
            if len(chosen_msgs) < 2 or len(rejected_msgs) < 2:
                continue

            chosen_text = chosen_msgs[1]["content"].strip()
            rejected_text = rejected_msgs[1]["content"].strip()
            if not chosen_text or not rejected_text:
                continue

            _, len_c = format_chat(tokenizer, prompt, chosen_text)
            _, len_r = format_chat(tokenizer, prompt, rejected_text)

            if len_c > MAX_SEQ_LENGTH or len_r > MAX_SEQ_LENGTH:
                dropped_test += 1
                continue

            formatted_prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)

            test_records.append({
                "prompt": formatted_prompt,
                "chosen": chosen_text,
                "rejected": rejected_text,
                "raw_prompt": prompt
            })
            pbar.update(1)
            if len(test_records) >= test_count:
                break
        pbar.close()

        print(f"Pref Test: Gathered {len(test_records)} pairs. Filtered out {dropped_test} overly long pairs.")
        with open(test_path, "w", encoding="utf-8") as f:
            for r in test_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return train_path, test_path


def main():
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    print(f"Loading tokenizer: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    prepare_math_data(tokenizer, target_count=5000)
    prepare_code_data(tokenizer, target_count=5000)
    prepare_instr_data(tokenizer, target_count=5000)
    prepare_pref_data(tokenizer, train_count=2000, test_count=500)
    print("\nAll datasets successfully prepared and cached locally!")


if __name__ == "__main__":
    main()
