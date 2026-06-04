import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def extract_json(text):
    # Prefer JSON inside a ```json fenced block
    marker = "```json"
    if marker in text:
        start = text.rfind(marker)
        block = text[start + len(marker):]
        end = block.find("```")
        if end != -1:
            block = block[:end].strip()
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                pass

    # Fallback: try the last JSON-looking object
    start = text.rfind("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-batches", type=int, default=500)
    args = parser.parse_args()

    Path("outputs").mkdir(exist_ok=True)

    print(f"Loading model: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
            if len(records) >= args.max_batches:
                break

    valid_json = 0
    failed = 0
    total_input_tokens = 0
    total_output_tokens = 0

    start_time = time.time()

    with open(args.output, "w", encoding="utf-8") as out:
        for idx, record in enumerate(records):
            comments = "\n".join(f"- {t}" for t in record["texts"])

            prompt = f"""
You are extracting semantic audience-demand signals from pre-release YouTube trailer comment sections.

Return ONLY valid JSON. Do not use markdown. Do not wrap your answer in ```json.

Schema:
{{
  "hype_score": 0,
  "skepticism_score": 0,
  "purchase_intent_score": 0,
  "controversy_score": 0,
  "sequel_fatigue_score": 0,
  "summary": "brief summary"
}}

Movie: {record["title"]}

YouTube Trailer Comments:
{comments}
"""

            messages = [{"role": "user", "content": prompt}]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            total_input_tokens += inputs["input_ids"].shape[1]

            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=180,
                        do_sample=False
                    )

                output_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
                total_output_tokens += output_tokens

                decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
                parsed = extract_json(decoded)

                if parsed is not None:
                    valid_json += 1
                else:
                    failed += 1

                out.write(json.dumps({
                    "batch_id": record["batch_id"],
                    "movie_id": record["movie_id"],
                    "title": record["title"],
                    "video_id": record.get("video_id", ""),
                    "channel_id": record.get("channel_id", ""),
                    "parsed_json": parsed,
                    "raw_output": decoded[-1000:]
                }, ensure_ascii=False) + "\n")

            except Exception as e:
                failed += 1
                out.write(json.dumps({
                    "batch_id": record["batch_id"],
                    "error": str(e)
                }) + "\n")

            if (idx + 1) % 25 == 0:
                print(f"Processed {idx + 1}/{len(records)} batches")

    total_seconds = time.time() - start_time
    seconds_per_batch = total_seconds / len(records)

    peak_memory_gb = 0
    if torch.cuda.is_available():
        peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    print("\n===== TIMING RESULTS =====")
    print(f"Batches processed: {len(records)}")
    print(f"Total runtime seconds: {total_seconds:.2f}")
    print(f"Total runtime minutes: {total_seconds / 60:.2f}")
    print(f"Seconds per batch: {seconds_per_batch:.3f}")
    print(f"Valid JSON outputs: {valid_json}")
    print(f"Failed/invalid outputs: {failed}")
    print(f"JSON validity rate: {valid_json / len(records):.2%}")
    print(f"Average input tokens: {total_input_tokens / len(records):.1f}")
    print(f"Average output tokens: {total_output_tokens / len(records):.1f}")
    print(f"Peak GPU memory GB: {peak_memory_gb:.2f}")

    print("\nEstimated full-run examples:")
    for full_batches in [5000, 10000, 20000]:
        h200_hours = (seconds_per_batch * full_batches) / 3600
        print(f"{full_batches} batches: {h200_hours:.2f} H200-hours")


if __name__ == "__main__":
    main()
