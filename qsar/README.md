# QSAR model

### Process QSAR data 
```
  mkidr sdf npy smiles
  python3 process_training_data.py
```
### Process generated compounds 
```
  python3 process_generated_data.py
```

### Train QSAR model and infer activity
```
  mkidr npy/y
  python3 qsar.py
```

### Draw figure with top 1000/2000/5000 most activity compounds
```
  python3 draw_fig.py
```
<center>

<div align=center><img width="800" height="500" src="https://raw.githubusercontent.com/VV123/denovo/main/qsar/Target1.png"/></div>
</center>  