# 💊 cMolGPT: A Conditional Generative Pre-Trained Transformer for Target-Specific De Novo Molecular Generation
![Github repo stars](https://img.shields.io/github/stars/VV123/cMolGPT)
![GitHub last commit](https://img.shields.io/github/last-commit/VV123/cMolGPT)

📝 [Medium post](https://medium.com/@wenlu.wang.1/cmolgpt-a-conditional-generative-pre-trained-transformer-for-de-novo-molecular-design-92ad2532da50)

Implementation of ["cMolGPT: A Conditional Generative Pre-Trained Transformer for Target-Specific De Novo Molecular Generation"](https://pubmed.ncbi.nlm.nih.gov/37298906/).
Enforcing target embeddings as queries and keys.

Please feel free to open an issue or email wenlu.wang.1@gmail.com and ye.wang@biogen.com if you have any questions. We will respond as soon as we can.

## Dependencies

environment_v100.yml tested on NVIDIA V100

environment_a6000.yml tested on RTX A6000

[Create env from yml file](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)

## Data

### [Mol_target_dataloader](https://github.com/alfredyewang/Mol_target_dataloader)
Please download this repo and put the folder in the root directory.
If you would like to finetune with your own target data, please replace 'target.smi'.

## How to run

*unzip train.sim.zip

### Train
```
  python3 main.py --batch_size 512 --mode train \
                  --path model_base.h5 
```
### Fine-tune
```
  python3 main.py --batch_size 512 --mode finetune \
                  --path model_base.h5 --loadmodel
```
*In the case of fine-tuning, the base model will be overwritten in place.

*You can change the number of targets in [model_auto.py](https://github.com/VV123/cMolGPT/blob/f0eba15dbf53b47a35afc305674c997354472590/model_auto.py#L58C66-L58C107).

### Infer/Generate
```
  python3 main.py --mode infer --target [0/1/2/3] --path model_finetune.h5
```

No target
```
  python3 main.py --mode infer --target 0 --path model_finetune.h5
```

Target 2
```
  python3 main.py --mode infer --target 2 --path model_finetune.h5
```

