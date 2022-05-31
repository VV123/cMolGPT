import vocabulary as mv
import dataset as md
import torch.utils.data as tud

from utils import read_delimited_file


mol_list = list(read_delimited_file('./chembl.filtered.smi'))
vocabulary = mv.create_vocabulary(smiles_list=mol_list, tokenizer=mv.SMILESTokenizer())
Dataset = md.Dataset(mol_list, vocabulary, mv.SMILESTokenizer())
coldata = tud.DataLoader(Dataset, 2, collate_fn=Dataset.collate_fn)

for batch in coldata:
    print(batch)
    # exit()
