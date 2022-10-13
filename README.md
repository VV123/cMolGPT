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
### Infer
```
  python3 main.py --mode infer --target 2 --path model_finetune.h5
```
