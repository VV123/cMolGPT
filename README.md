# De novo conditional molecular design

Enforcing target embeddings as queries and keys.

## Data

### [Mol_target_dataloader](https://github.com/alfredyewang/Mol_target_dataloader)


## How to run

### Train
```
  python3 main.py --batch_size 512 --mode [train, finetune] \
                  --path model_chem.h5 --loadmodel
```
*In the case of fine-tuning, the base model will be replaced in place.

*You can change the number of targets in [model_auto.py](https://github.com/VV123/cMolGPT/blob/f0eba15dbf53b47a35afc305674c997354472590/model_auto.py#L58C66-L58C107).

### Infer/Generate
```
  python3 main.py --mode infer --target 2 --path model_finetune.h5
```
