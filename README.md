## Environment setting
```bash
pip install -r requirements.txt
```


## Preprocess
Download VocalSet directory from google drive
Put the directory to 'data' directory


## Extract Features 
```bash
python . extract -c ./configs/mert/MERT-v1-95M/VocalSetT.yaml
```


## Fine-tuning & Test
```bash
python . probe -c ./configs/mert/MERT-v1-95M/VocalSetT.yaml
```



