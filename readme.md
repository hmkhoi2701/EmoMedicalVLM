## Download and extract the SLAKE dataset

```bash
apt install -y git-lfs
git clone https://huggingface.co/datasets/BoKelvin/SLAKE
cd SLAKE
git lfs pull
unzip imgs.zip
```

## Download and extract the VQA-Rad dataset

```bash
git clone https://huggingface.co/datasets/flaviagiammarino/vqa-rad
```

## Test runs with MedGemma

For a single run with default prompt, run 

```python models/run_medgemma.py``` 

or completed run with all emotions with ```bash test_all_emotions.sh```