## Download and extract the SLAKE dataset

```bash
apt install -y git-lfs
git clone https://huggingface.co/datasets/BoKelvin/SLAKE
cd SLAKE
git lfs pull
unzip imgs.zip
```

## Download the VQA dataset

```bash
apt install -y git-lfs
git clone https://huggingface.co/datasets/flaviagiammarino/vqa-rad
cd vqa-rad
git lfs pull
```

## Test runs with MedGemma

For a single run with default prompt, run 

```python models/run_medgemma.py``` 

or completed run with all emotions with ```bash test_all_emotions.sh```