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

        return samples