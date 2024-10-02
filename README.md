## Environment setting
```bash
pip install -r requirements.txt
```
```
mkdir data
```


## Download dataset
You can download dataset from [google drive](https://drive.google.com/drive/folders/1tAVbPfEgRVE4_f7dyRPW3n67UyUkFbut?usp=drive_link).

Put the `VocalSet` directory to `data` directory



## Extract Features 
```bash
python . extract -c ./configs/mert/MERT-v1-95M/VocalSetT.yaml
```


## Fine-tuning & Test
```bash
python . probe -c ./configs/mert/MERT-v1-95M/VocalSetT.yaml
```

### Contact
Please contact me by email if you have any questions.

`js_choi@korea.ac.kr`


