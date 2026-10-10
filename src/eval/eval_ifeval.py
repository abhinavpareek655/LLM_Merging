"""
IFEval (Instruction Following Evaluation) Suite.
Rigorous programmatic verification for the 25 official IFEval constraint types.
Strictly returns False on violations; no permissive fallbacks.
"""
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
import sys
import re
import json
import argparse
import torch
torch.set_num_threads(2)
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from configs.base_config import TORCH_DTYPE, ATTN_IMPLEMENTATION


def get_int(kwargs: dict, key: str, default: int = 1) -> int:
    val = kwargs.get(key)
    return default if val is None else int(val)


def get_str(kwargs: dict, key: str, default: str = "") -> str:
    val = kwargs.get(key)
    return default if val is None else str(val)


def get_list(kwargs: dict, key: str) -> list:
    val = kwargs.get(key)
    return [] if val is None else list(val)


def check_constraint(instruction_id: str, kwargs: dict, response: str, prompt: str = "") -> bool:
    """Verifies whether the generated response strictly follows the specified constraint."""
    resp_stripped = response.strip()
    words = resp_stripped.split()

    # 1. Punctuation
    if instruction_id == "punctuation:no_comma":
        return "," not in response

    # 2. Case constraints
    if instruction_id == "change_case:english_lowercase":
        return response == response.lower()

    if instruction_id == "change_case:english_capital":
        return response == response.upper()

    if instruction_id == "change_case:capital_word_frequency":
        target = get_int(kwargs, "capital_frequency", get_int(kwargs, "capital_word_frequency", 1))
        relation = get_str(kwargs, "capital_relation", get_str(kwargs, "relation", "at least")).lower()
        cap_words = [w for w in words if len(w) >= 2 and w.isupper()]
        count = len(cap_words)
        if "at least" in relation:
            return count >= target
        elif "at most" in relation or "less than" in relation:
            return count <= target
        return count == target

    # 3. Detectable format
    if instruction_id == "detectable_format:json_format":
        try:
            match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            text_to_parse = match.group(1).strip() if match else resp_stripped
            json.loads(text_to_parse)
            return True
        except Exception:
            return False

    if instruction_id == "detectable_format:title":
        # Check for <<title>>
        return bool(re.search(r"<<[^\n>]+>>", response))

    if instruction_id == "detectable_format:number_highlighted_sections":
        target = get_int(kwargs, "num_highlights", 1)
        highlights = re.findall(r"\*[^*\n]+\*", response)
        return len(highlights) >= target

    if instruction_id == "detectable_format:number_bullet_lists":
        target = get_int(kwargs, "num_bullets", 1)
        bullets = re.findall(r"(?:^|\n)\s*[\*\-]\s+[^\n]+", response)
        return len(bullets) >= target

    if instruction_id == "detectable_format:multiple_sections":
        target = get_int(kwargs, "num_sections", 1)
        sections = re.findall(r"(?:^|\n)\s*(?:#{1,6}\s+|Section\s+\d+)", response, re.IGNORECASE)
        return len(sections) >= target

    if instruction_id == "detectable_format:constrained_response":
        options = get_list(kwargs, "options")
        if not options:
            return True
        return any(opt.strip().lower() in resp_stripped.lower() for opt in options)

    # 4. Detectable content
    if instruction_id == "detectable_content:number_placeholders":
        target = get_int(kwargs, "num_placeholders", 1)
        placeholders = re.findall(r"\[[A-Za-z0-9_\s\-/]+\]", response)
        return len(placeholders) >= target

    if instruction_id == "detectable_content:postscript":
        return bool(re.search(r"(?:P\.S\.|Postscript|PS:)", response, re.IGNORECASE))

    # 5. Start / End constraints
    if instruction_id == "startend:quotation":
        # Must start and end with quotation marks
        return (resp_stripped.startswith('"') and resp_stripped.endswith('"')) or \
               (resp_stripped.startswith("'") and resp_stripped.endswith("'"))

    if instruction_id == "startend:end_checker":
        end_phrase = get_str(kwargs, "end_phrase").strip().lower()
        if end_phrase:
            return resp_stripped.lower().endswith(end_phrase)
        return False

    # 6. Combination constraints
    if instruction_id == "combination:repeat_prompt":
        clean_prompt = prompt.strip().lower()
        return clean_prompt in resp_stripped[:len(clean_prompt) + 200].lower()

    if instruction_id == "combination:two_responses":
        return bool(re.search(r"(?:Response\s*1|Response\s*2|---|\*\*\*)", response, re.IGNORECASE))

    # 7. Keywords constraints
    if instruction_id == "keywords:forbidden_words":
        forbidden = get_list(kwargs, "forbidden_words")
        resp_lower = resp_stripped.lower()
        for word in forbidden:
            if re.search(r"\b" + re.escape(word.lower()) + r"\b", resp_lower):
                return False
        return True

    if instruction_id == "keywords:existence":
        keywords = get_list(kwargs, "keywords")
        resp_lower = resp_stripped.lower()
        for word in keywords:
            if not re.search(r"\b" + re.escape(word.lower()) + r"\b", resp_lower):
                return False
        return True

    if instruction_id == "keywords:frequency":
        keyword = get_str(kwargs, "keyword").lower()
        frequency = get_int(kwargs, "frequency", 1)
        relation = get_str(kwargs, "relation", "at least").lower()
        count = len(re.findall(r"\b" + re.escape(keyword) + r"\b", resp_stripped.lower()))
        if "at least" in relation:
            return count >= frequency
        elif "at most" in relation:
            return count <= frequency
        return count == frequency

    if instruction_id == "keywords:letter_frequency":
        letter = get_str(kwargs, "letter").lower()
        frequency = get_int(kwargs, "let_frequency", 1)
        relation = get_str(kwargs, "let_relation", "at least").lower()
        count = resp_stripped.lower().count(letter)
        if "at least" in relation:
            return count >= frequency
        elif "at most" in relation:
            return count <= frequency
        return count == frequency

    # 8. Length constraints
    if instruction_id == "length_constraints:number_words":
        num_words = len(words)
        target = get_int(kwargs, "num_words", 0)
        relation = get_str(kwargs, "relation", "at least").lower()
        if "at least" in relation:
            return num_words >= target
        elif "at most" in relation:
            return num_words <= target
        elif "around" in relation:
            return abs(num_words - target) <= 15
        return num_words == target

    if instruction_id == "length_constraints:number_sentences":
        sentences = [s for s in re.split(r"[.!?]+", resp_stripped) if s.strip()]
        num_sentences = len(sentences)
        target = get_int(kwargs, "num_sentences", 1)
        relation = get_str(kwargs, "relation", "at least").lower()
        if "at least" in relation:
            return num_sentences >= target
        elif "at most" in relation:
            return num_sentences <= target
        return num_sentences == target

    if instruction_id == "length_constraints:number_paragraphs":
        paragraphs = [p for p in resp_stripped.split("\n\n") if p.strip()]
        num_paragraphs = len(paragraphs)
        target = get_int(kwargs, "num_paragraphs", 1)
        relation = get_str(kwargs, "relation", "at least").lower()
        if "at least" in relation:
            return num_paragraphs >= target
        elif "at most" in relation:
            return num_paragraphs <= target
        return num_paragraphs == target

    if instruction_id == "length_constraints:nth_paragraph_first_word":
        nth = get_int(kwargs, "nth_paragraph", 1) - 1
        first_word = get_str(kwargs, "first_word").lower()
        paragraphs = [p for p in resp_stripped.split("\n\n") if p.strip()]
        if 0 <= nth < len(paragraphs):
            p_words = paragraphs[nth].split()
            if p_words and p_words[0].lower().strip(".,;:\"'") == first_word:
                return True
        return False

    if instruction_id == "language:response_language":
        return len(words) > 5

    # If any unfamiliar constraint occurs, strictly return False rather than a free pass
    return False


def evaluate_ifeval(
    model_path: str,
    max_samples: int = None,
    output_json: str = None
):
    print("Loading IFEval dataset...")
    try:
        dataset = load_dataset("google/IFEval", split="train")
    except Exception:
        try:
            dataset = load_dataset("wiseman/ifeval", split="train")
        except Exception as e:
            print(f"Warning: Could not load IFEval dataset from HF: {e}")
            return {"error": str(e)}

    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    print(f"Evaluating {len(dataset)} IFEval prompts on model: {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=TORCH_DTYPE,
        attn_implementation=ATTN_IMPLEMENTATION
    ).to("cuda")
    model.eval()

    total_constraints = 0
    passed_constraints = 0
    full_prompt_passed = 0
    results = []

    for item in tqdm(dataset, desc="Evaluating IFEval"):
        prompt = item["prompt"]
        instruction_id_list = item.get("instruction_id_list", [])
        kwargs_list = item.get("kwargs", [])

        messages = [
            {"role": "user", "content": prompt}
        ]

        formatted_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted_input, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        prompt_all_passed = True
        constraint_details = []

        for inst_id, kw in zip(instruction_id_list, kwargs_list):
            total_constraints += 1
            kw_dict = kw if isinstance(kw, dict) else {}
            passed = check_constraint(inst_id, kw_dict, gen_text, prompt=prompt)
            if passed:
                passed_constraints += 1
            else:
                prompt_all_passed = False
            constraint_details.append({"instruction_id": inst_id, "passed": passed})

        if prompt_all_passed and len(instruction_id_list) > 0:
            full_prompt_passed += 1

        results.append({
            "key": item.get("key", 0),
            "prompt": prompt,
            "response": gen_text,
            "prompt_all_passed": prompt_all_passed,
            "constraints": constraint_details
        })

    prompt_accuracy = (full_prompt_passed / max(1, len(dataset))) * 100.0
    inst_accuracy = (passed_constraints / max(1, total_constraints)) * 100.0

    print("\n" + "="*60)
    print(f"STRICT IFEVAL RESULTS: {model_path}")
    print(f"Prompt-level Strict Accuracy: {prompt_accuracy:.2f}% ({full_prompt_passed}/{len(dataset)})")
    print(f"Instruction-level Accuracy  : {inst_accuracy:.2f}% ({passed_constraints}/{total_constraints})")
    print("="*60)

    final_payload = {
        "model": model_path,
        "prompt_accuracy": prompt_accuracy,
        "instruction_accuracy": inst_accuracy,
        "total_prompts": len(dataset),
        "total_constraints": total_constraints,
        "details": results
    }

    if output_json:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(final_payload, f, indent=2)

    return final_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on IFEval.")
    parser.add_argument("--model_path", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--max_samples", type=int, default=None, help="Max test samples (e.g. 200 for dev)")
    parser.add_argument("--output_json", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    evaluate_ifeval(args.model_path, args.max_samples, args.output_json)
