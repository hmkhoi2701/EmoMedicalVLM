from datasets import load_dataset
import pandas as pd
import os

def get_dataset(dataset_name, split, yes_no=False):
    if dataset_name == 'BoKelvin/SLAKE' or dataset_name == 'SLAKE':
        assert split in ['train', 'validation', 'test'], "Split must be one of 'train', 'validation', or 'test'."
        splits = {'train': 'train.json', 'validation': 'validation.json', 'test': 'test.json'}
        df = pd.read_json("SLAKE/" + splits[split])
        df = df[df['q_lang'] == 'en']
        if yes_no:
            df = df[df['answer'].isin(['Yes', 'No'])]
        samples = []
        for _, row in df.iterrows():
            samples.append({
                "image": os.path.join("SLAKE/imgs", row["img_name"]),
                "question": row["question"],
                "answer": row["answer"],
                "location": row["location"],
                "modality": row["modality"],
                "answer_type": row["answer_type"],
                "content_type": row["content_type"],
            })
    elif dataset_name == 'vqa-rad':
        assert split in ['train', 'test'], "Split must be one of 'train', or 'test'."
        splits = {'train':'vqa-rad/data/train-00000-of-00001-eb8844602202be60.parquet',
                  'test':'vqa-rad/data/test-00000-of-00001-e5bc3d208bb4deeb.parquet'}
        df = pd.read_parquet(splits[split])
        if yes_no:
            df = df[df['answer'].isin(['yes', 'no'])]
        samples = []
        for _, row in df.iterrows():
            samples.append({
                "image": row["image"],
                "question": row["question"],
                "answer": row["answer"],
                })
        return samples