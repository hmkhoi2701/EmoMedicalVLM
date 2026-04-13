from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import argparse
from io import BytesIO
from copy import deepcopy
import torch
import json
import os
from tqdm import tqdm
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_utils import get_dataset
from emotion_prompts import SYSTEM_PROMPT, USER_PROMPTS


def setup_logger(log_file: Path):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def load_image(sample: Dict[str, Any], dataset_name: str) -> Image.Image:
    if "slake" in dataset_name:
        return Image.open(sample["image"]).convert("RGB")

    if "vqa-rad" in dataset_name or "vqa_rad" in dataset_name or "vqarad" in dataset_name:
        image_field = sample["image"]

        if isinstance(image_field, Image.Image):
            return image_field.convert("RGB")

        if isinstance(image_field, dict):
            if "bytes" in image_field and image_field["bytes"] is not None:
                return Image.open(BytesIO(image_field["bytes"])).convert("RGB")
            if "path" in image_field and image_field["path"] is not None:
                return Image.open(image_field["path"]).convert("RGB")
            raise ValueError(f"Unsupported VQA-RAD image dict keys: {list(image_field.keys())}")

        if isinstance(image_field, str):
            return Image.open(image_field).convert("RGB")

        raise ValueError(f"Unsupported VQA-RAD image type: {type(image_field)}")

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def normalize_yes_no_answer(text: str) -> str:
    cleaned = text.strip().replace("\n", " ")
    lowered = cleaned.lower()

    if lowered.startswith("yes"):
        return "Yes"
    if lowered.startswith("no"):
        return "No"
    return cleaned


def build_messages(conv_mode: str, emotion: str, yes_no: bool):
    if conv_mode == "single":
        return [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": None},
                    {
                        "type": "text",
                        "text": USER_PROMPTS[emotion]
                        + " Question: {question}"
                        + (" Please answer with 'Yes' or 'No'." if yes_no else "")
                    },
                ],
            },
        ]

    return [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": USER_PROMPTS[emotion]},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I'm here to help."},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": None},
                {
                    "type": "text",
                    "text": " Question: {question}"
                    + (" Please answer with 'Yes' or 'No'." if yes_no else "")
                },
            ],
        },
    ]


def make_sample_uid(sample: Dict[str, Any], sample_idx: int) -> str:
    image_part = ""
    if "image" in sample:
        image_field = sample["image"]
        if isinstance(image_field, str):
            image_part = image_field
        elif isinstance(image_field, dict):
            if "path" in image_field and image_field["path"] is not None:
                image_part = str(image_field["path"])
            elif "bytes" in image_field and image_field["bytes"] is not None:
                image_part = f"bytes_{len(image_field['bytes'])}"
            else:
                image_part = "image_dict"
        else:
            image_part = str(type(image_field))

    question = str(sample.get("question", ""))
    answer = str(sample.get("answer", ""))
    return f"{sample_idx}|||{image_part}|||{question}|||{answer}"


def load_finished_uids(output_path: Path, logger) -> set:
    finished = set()
    if not output_path.exists():
        return finished

    with open(output_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                uid = obj.get("_sample_uid")
                if uid is not None:
                    finished.add(uid)
            except Exception:
                logger.warning(f"Skipping malformed jsonl line {line_num} in {output_path}")

    return finished


def build_write_dict(
    sample: Dict[str, Any],
    decoded: str,
    model_id: str,
    args,
    sample_uid: str,
    sample_idx: int,
    save_scores: bool = False,
    max_prob: float = None,
    yes_prob: float = None,
    no_prob: float = None,
) -> Dict[str, Any]:
    write_dict = {
        "_sample_uid": sample_uid,
        "_sample_idx": sample_idx,
        "question": sample["question"],
        "answer": sample["answer"],
        "model_answer": decoded,
        "dataset": args.dataset,
        "conv_mode": args.conv_mode,
        "model_name": model_id,
        "emotion": args.emotion,
        "yes_no": args.yes_no,
    }

    for key in ["location", "modality", "answer_type", "content_type"]:
        if key in sample:
            write_dict[key] = sample[key]

    if "image" in sample and isinstance(sample["image"], str):
        write_dict["image"] = sample["image"]
    elif "image" in sample and isinstance(sample["image"], dict):
        image_field = sample["image"]
        if "path" in image_field and image_field["path"] is not None:
            write_dict["image"] = image_field["path"]

    if save_scores:
        write_dict["max_prob"] = max_prob
        write_dict["yes_prob"] = yes_prob
        write_dict["no_prob"] = no_prob

    return write_dict


def prepare_batch_inputs(
    batch_samples: List[Dict[str, Any]],
    messages_template,
    processor,
    model,
    dtype,
) -> Tuple[Dict[str, torch.Tensor], List[int]]:
    batch_messages = []
    input_lens = []

    for item in batch_samples:
        sample = item["sample"]
        image = item["image"]
        question = sample["question"]

        sample_messages = deepcopy(messages_template)
        sample_messages[-1]["content"][0]["image"] = image
        sample_messages[-1]["content"][1]["text"] = sample_messages[-1]["content"][1]["text"].format(
            question=question
        )

        batch_messages.append(sample_messages)

        temp_inputs = processor.apply_chat_template(
            sample_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_lens.append(temp_inputs["input_ids"].shape[-1])

    inputs = processor.apply_chat_template(
        batch_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to(model.device, dtype=dtype)

    return inputs, input_lens


def batched(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="BoKelvin/SLAKE", help="The dataset to run the model on.")
    parser.add_argument("--split", type=str, default="test", help="The split of the dataset to run the model on.")
    parser.add_argument("--output_file", type=str, default="output/medgemma.jsonl", help="The file to save the model's predictions to.")
    parser.add_argument("--emotion", type=str, default="default", help="The emotion category for the user prompt.")
    parser.add_argument("--yes_no", action="store_true", help="Whether to filter yes/no questions and include them in the prompt.")
    parser.add_argument("--conv_mode", type=str, default="single", choices=["single", "multi"], help="Whether to use single-turn or multi-turn conversation format.")
    parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite the output file if it already exists.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference.")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Override max_new_tokens.")
    args = parser.parse_args()

    if args.emotion == "default" and args.conv_mode == "multi":
        print("Skipping invalid combination: default emotion with multi-turn mode.")
        return

    output_path = Path(args.output_file)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path("logs/medgemma")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{output_path.stem}.log"

    logger = setup_logger(log_file)

    logger.info("Starting run_medgemma.py")
    logger.info(f"Args: {vars(args)}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Log file: {log_file}")

    if args.overwrite and output_path.exists():
        output_path.unlink()
        logger.info(f"Removed existing output file: {output_path}")

    model_id = "google/medgemma-1.5-4b-it"
    dtype = torch.bfloat16

    logger.info(f"Loading model: {model_id}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)

    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id

    logger.info(f"Loading dataset: dataset={args.dataset}, split={args.split}, yes_no={args.yes_no}")
    samples = get_dataset(args.dataset, args.split, args.yes_no)
    logger.info(f"Loaded {len(samples)} samples")

    save_scores = args.yes_no and args.conv_mode == "single" and args.batch_size == 1
    if save_scores:
        yes_id = processor.tokenizer.convert_tokens_to_ids("Yes")
        no_id = processor.tokenizer.convert_tokens_to_ids("No")
        logger.info(f"save_scores=True | yes_id={yes_id}, no_id={no_id}")
    else:
        yes_id = None
        no_id = None
        logger.info("save_scores=False")

    finished_uids = load_finished_uids(output_path, logger)
    logger.info(f"Loaded {len(finished_uids)} finished sample ids from existing output")

    messages = build_messages(args.conv_mode, args.emotion, args.yes_no)
    dataset_name = args.dataset.lower()

    all_items = []
    for idx, sample in enumerate(samples):
        uid = make_sample_uid(sample, idx)
        if uid in finished_uids:
            continue
        all_items.append({
            "sample_idx": idx,
            "sample_uid": uid,
            "sample": sample,
        })

    logger.info(f"Remaining samples to run: {len(all_items)} / {len(samples)}")

    if len(all_items) == 0:
        logger.info("Nothing to do. All samples already finished.")
        return

    if args.max_new_tokens is not None:
        max_new_tokens = args.max_new_tokens
    else:
        if args.yes_no:
            max_new_tokens = 8
        else:
            max_new_tokens = 128

    logger.info(f"Using batch_size={args.batch_size}")
    logger.info(f"Using max_new_tokens={max_new_tokens}")

    num_written = 0
    num_failed = 0
    processed_now = 0

    progress_bar = tqdm(total=len(all_items), desc="Processing samples")

    for batch_items in batched(all_items, args.batch_size):
        loaded_batch = []
        failed_in_loading = []

        for item in batch_items:
            try:
                image = load_image(item["sample"], dataset_name)
                loaded_batch.append({
                    **item,
                    "image": image,
                })
            except Exception as e:
                num_failed += 1
                processed_now += 1
                failed_in_loading.append(item)
                logger.exception(f"Error loading sample {item['sample_idx']}: {e}")
                progress_bar.update(1)

        if not loaded_batch:
            continue

        try:
            inputs, input_lens = prepare_batch_inputs(
                loaded_batch,
                messages,
                processor,
                model,
                dtype,
            )

            with torch.inference_mode():
                if save_scores and len(loaded_batch) == 1:
                    generation = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )
                else:
                    generation = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )

            results_to_write = []

            if save_scores and len(loaded_batch) == 1:
                probs = generation["scores"][0][0].softmax(dim=-1)
                max_prob, _ = probs.max(dim=-1)
                max_prob = round(max_prob.item(), 4)
                yes_prob = round(probs[yes_id].item(), 4)
                no_prob = round(probs[no_id].item(), 4)

                seq = generation["sequences"][0]
                generated = seq[input_lens[0]:]
                decoded = processor.decode(generated, skip_special_tokens=True).strip()
                decoded = decoded.replace("\n", " ").strip()
                if args.yes_no:
                    decoded = normalize_yes_no_answer(decoded)

                item = loaded_batch[0]
                write_dict = build_write_dict(
                    sample=item["sample"],
                    decoded=decoded,
                    model_id=model_id,
                    args=args,
                    sample_uid=item["sample_uid"],
                    sample_idx=item["sample_idx"],
                    save_scores=True,
                    max_prob=max_prob,
                    yes_prob=yes_prob,
                    no_prob=no_prob,
                )
                results_to_write.append(write_dict)

            else:
                for i, item in enumerate(loaded_batch):
                    seq = generation[i]
                    generated = seq[input_lens[i]:]
                    decoded = processor.decode(generated, skip_special_tokens=True).strip()
                    decoded = decoded.replace("\n", " ").strip()
                    if args.yes_no:
                        decoded = normalize_yes_no_answer(decoded)

                    write_dict = build_write_dict(
                        sample=item["sample"],
                        decoded=decoded,
                        model_id=model_id,
                        args=args,
                        sample_uid=item["sample_uid"],
                        sample_idx=item["sample_idx"],
                        save_scores=False,
                    )
                    results_to_write.append(write_dict)

            with open(output_path, "a", encoding="utf-8") as f:
                for obj in results_to_write:
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")

            batch_written = len(results_to_write)
            num_written += batch_written
            processed_now += len(loaded_batch)
            progress_bar.update(len(loaded_batch))

            if processed_now % 50 == 0 or processed_now == len(all_items):
                logger.info(
                    f"Processed current_run={processed_now}/{len(all_items)} | "
                    f"written_now={num_written} | failed_now={num_failed}"
                )

        except torch.cuda.OutOfMemoryError as e:
            logger.exception(f"CUDA OOM on batch starting at sample_idx={loaded_batch[0]['sample_idx']}: {e}")
            logger.info("Falling back to single-sample inference for this batch.")

            torch.cuda.empty_cache()

            for item in loaded_batch:
                try:
                    single_inputs, single_input_lens = prepare_batch_inputs(
                        [item],
                        messages,
                        processor,
                        model,
                        dtype,
                    )

                    with torch.inference_mode():
                        if save_scores:
                            generation = model.generate(
                                **single_inputs,
                                max_new_tokens=max_new_tokens,
                                do_sample=False,
                                output_scores=True,
                                return_dict_in_generate=True,
                            )
                            probs = generation["scores"][0][0].softmax(dim=-1)
                            max_prob, _ = probs.max(dim=-1)
                            max_prob = round(max_prob.item(), 4)
                            yes_prob = round(probs[yes_id].item(), 4)
                            no_prob = round(probs[no_id].item(), 4)
                            seq = generation["sequences"][0]
                            generated = seq[single_input_lens[0]:]
                            decoded = processor.decode(generated, skip_special_tokens=True).strip()
                        else:
                            generation = model.generate(
                                **single_inputs,
                                max_new_tokens=max_new_tokens,
                                do_sample=False,
                            )
                            seq = generation[0]
                            generated = seq[single_input_lens[0]:]
                            decoded = processor.decode(generated, skip_special_tokens=True).strip()
                            max_prob = None
                            yes_prob = None
                            no_prob = None

                    decoded = decoded.replace("\n", " ").strip()
                    if args.yes_no:
                        decoded = normalize_yes_no_answer(decoded)

                    write_dict = build_write_dict(
                        sample=item["sample"],
                        decoded=decoded,
                        model_id=model_id,
                        args=args,
                        sample_uid=item["sample_uid"],
                        sample_idx=item["sample_idx"],
                        save_scores=save_scores,
                        max_prob=max_prob,
                        yes_prob=yes_prob,
                        no_prob=no_prob,
                    )

                    with open(output_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(write_dict, ensure_ascii=False) + "\n")

                    num_written += 1

                except Exception as inner_e:
                    num_failed += 1
                    logger.exception(f"Error processing sample {item['sample_idx']} after OOM fallback: {inner_e}")

                finally:
                    processed_now += 1
                    progress_bar.update(1)

        except Exception as e:
            logger.exception(f"Batch error starting at sample_idx={loaded_batch[0]['sample_idx']}: {e}")

            for item in loaded_batch:
                num_failed += 1
                processed_now += 1
                progress_bar.update(1)

    progress_bar.close()
    logger.info(
        f"Finished inference | total_dataset={len(samples)} | remaining_this_run={len(all_items)} | "
        f"written_now={num_written} | failed_now={num_failed}"
    )


if __name__ == "__main__":
    main()