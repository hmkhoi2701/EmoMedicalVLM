from io import BytesIO
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Tuple

def load_image(sample: Dict[str, Any], dataset_name: str) -> Image.Image:
    if "slake" in dataset_name or "vqa-med-2019" in dataset_name:
        return Image.open(sample["image"]).convert("RGB")

    if "vqa-rad" in dataset_name:
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


def get_image_path(sample: Dict[str, Any], dataset_name: str, cache_dir: Path) -> str:
    if "slake" in dataset_name or "vqa-med-2019" in dataset_name:
        return sample["image"]

    if "vqa-rad" in dataset_name:
        image_field = sample["image"]

        if isinstance(image_field, str) and Path(image_field).is_file():
            return image_field

        if isinstance(image_field, dict) and image_field.get("path") is not None:
            image_path = Path(image_field["path"])
            if image_path.is_file():
                return str(image_path)

        cache_dir.mkdir(parents=True, exist_ok=True)
        image_id = sample.get("image_id", len(list(cache_dir.glob("*.png"))))
        image_path = cache_dir / f"{image_id}.png"
        if not image_path.exists():
            load_image(sample, dataset_name).save(image_path)
        return str(image_path)

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_output_dict(
    sample: Dict[str, Any],
    question: str,
    model_answer: str,
    emotion: str,
    dataset_name: str,
) -> Dict[str, Any]:
    dataset_name = dataset_name.lower()

    if "slake" in dataset_name:
        return {
            "image": sample["image"],
            "question": question,
            "answer": sample["answer"],
            "model_answer": model_answer,
            "emotion": emotion,
            "location": sample["location"],
            "modality": sample["modality"],
            "answer_type": sample["answer_type"],
            "content_type": sample["content_type"],
        }

    if "vqa-rad" in dataset_name:
        return {
            "image": sample["image_id"] if "image_id" in sample else sample.get("image", ""),
            "question": question,
            "answer": sample.get("answer", ""),
            "model_answer": model_answer,
            "emotion": emotion,
        }

    return {
        "image": sample.get("image_id", sample.get("image", "")),
        "question": question,
        "answer": sample.get("answer", ""),
        "model_answer": model_answer,
        "emotion": emotion,
    }
