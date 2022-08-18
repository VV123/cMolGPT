import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim import SGD, Adam
from torch.nn import MSELoss, L1Loss
from torch.nn.init import xavier_uniform_
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from transformers import BertTokenizer, BertModel
import numpy as np
import sys
from model_auto import Seq2SeqTransformer, PositionalEncoding, generate_square_subsequent_mask, create_mask
from utils import top_k_top_p_filtering, open_file, read_csv_file, load_sets
import vocabulary as mv
import dataset as md
import torch.utils.data as tud
from utils import read_delimited_file
import os.path
import glob
import math
import torch
import torch.nn as nn
from collections import Counter
from torch import Tensor
import io
import time
from topk import topk_filter

torch.manual_seed(0)

def evaluate(model, valid_iter):
    model.eval()
    losses = 0
    for idx, _tgt in (enumerate(valid_iter)):
        _target = None
        if type(_tgt) is tuple:
            _tgt, _target = _tgt
            _target = torch.LongTensor(_target).to(device)
        tgt = _tgt.transpose(0, 1).to(device)
        tgt_input = tgt[:-1, :]

        tgt_mask, tgt_padding_mask = create_mask(tgt_input)

        if _target is None:
            target = torch.zeros((tgt_input.size()[-1]), dtype=torch.int32).to(device)
        else:
            target = _target
        #target = torch.zeros((tgt_input.size()[-1]), dtype=torch.int32).to(device)
        logits = model(tgt_input, tgt_mask, tgt_padding_mask, target)
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
    return losses / len(valid_iter)


def train_epoch(model, train_iter, optimizer):
    model.train()
    losses = 0
    for idx, _tgt in enumerate(train_iter):
        _target = None
        if type(_tgt) is tuple:
            _tgt, _target = _tgt
            _target = torch.LongTensor(_target).to(device)
        #print(type(_tgt) is tuple)
        tgt = _tgt.transpose(0, 1).to(device)
        # remove encoder
        tgt_input = tgt[:-1, :]

        tgt_mask, tgt_padding_mask = create_mask(tgt_input)
        if _target is None:
            target = torch.zeros((tgt_input.size()[-1]), dtype=torch.int32).to(device)
        else:
            target = _target
        
        logits = model(tgt_input, tgt_mask, tgt_padding_mask, target)
      
        optimizer.zero_grad()
      
        tgt_out = tgt[1:,:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        loss.backward()

        optimizer.step()
        if idx % 100 == 0:
            print('Train Epoch: {}\t Loss: {:.6f}'.format(epoch, loss.item()))     
        losses += loss.item()

    print('====> Epoch: {0} total loss: {1:.4f}.'.format(epoch, losses))
    return losses / len(train_iter)

def greedy_decode(model, max_len, start_symbol, target):
    #memory = torch.zeros(40, 512, 512).to('cuda')
    #memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len-1):
        #s, b = ys.size()
        # batch_size = 1
        b = 1
        s = max_len
        FFD = 512
        if target == 0:
            _target = torch.zeros((b), dtype=torch.int32).to(device)
        else:
            _target = (torch.ones((b), dtype=torch.int32)*target).to(device)
        #memory = torch.zeros(s, b, FFD).to('cuda')
        #memory = memory.to(device)
        #memory_mask = torch.zeros(ys.shape[0], memory.shape[0]).to(device).type(torch.bool)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                                    .type(torch.bool)).to(device)
        out = model.decode(ys, tgt_mask, _target)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1]) #[b, vocab_size]
        pred_proba_t = topk_filter(prob, top_k=30) #[b, vocab_size]
        probs = pred_proba_t.softmax(dim=1) #[b, vocab_size]
        next_word = torch.multinomial(probs, 1)
        #_, next_word = torch.max(prob, dim = 1)
        next_word = next_word.item()
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(ys.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
          break
    return ys

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--mode', choices=['train', 'infer', 'baseline', 'finetune'],\
        default='train',help='Run mode')
    arg_parser.add_argument('--device', choices=['cuda', 'cpu'],\
        default='cuda',help='Device')
    arg_parser.add_argument('--epoch', default='100', type=int)
    arg_parser.add_argument('--batch_size', default='512', type=int)
    arg_parser.add_argument('--layer', default=3, type=int)
    arg_parser.add_argument('--path', default='model_chem.h5', type=str)
    arg_parser.add_argument('--datamode', default=1, type=int)
    arg_parser.add_argument('--target', default=1, type=int)
    arg_parser.add_argument('--d_model', default=512, type=int)
    arg_parser.add_argument('--nhead', default=8, type=int)
    arg_parser.add_argument('--embedding_size', default=200, type=int)
    arg_parser.add_argument('--loadmodel', default=False, action="store_true")
    arg_parser.add_argument("--loaddata", default=False, action="store_true")
    args = arg_parser.parse_args()

    print('==========  Transformer x->x ==============')

    #scaffold_list, decoration_list = zip(*read_csv_file('zinc/zinc.smi', num_fields=2))
    #vocabulary = mv.DecoratorVocabulary.from_lists(scaffold_list, decoration_list)
    #training_sets = load_sets('zinc/zinc.smi')
    #dataset = md.DecoratorDataset(training_sets, vocabulary=vocabulary)

    mol_list0_train = list(read_delimited_file('train.smi'))
    mol_list0_test = list(read_delimited_file('test.smi'))
    
    mol_list1, target_list = zip(*read_csv_file('Mol_target_dataloader/target.smi', num_fields=2))
    mol_list = mol_list0_train
    mol_list.extend(mol_list0_test) 
    mol_list.extend(mol_list1)
    vocabulary = mv.create_vocabulary(smiles_list=mol_list, tokenizer=mv.SMILESTokenizer())
    
    train_data = md.Dataset(mol_list0_train, vocabulary, mv.SMILESTokenizer())
    test_data = md.Dataset(mol_list0_test, vocabulary, mv.SMILESTokenizer())

    BATCH_SIZE = args.batch_size
    SRC_VOCAB_SIZE = len(vocabulary)
    TGT_VOCAB_SIZE = len(vocabulary)

    EMB_SIZE = args.d_model
    NHEAD = args.nhead
    FFN_HID_DIM = 512

    NUM_ENCODER_LAYERS = args.layer
    NUM_DECODER_LAYERS = args.layer
    NUM_EPOCHS = args.epoch
    PAD_IDX = 0
    BOS_IDX = 1
    EOS_IDX = 2
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = args.device

    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, 
                                 EMB_SIZE, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
                                 FFN_HID_DIM, args=args)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    
    #num_train= int(len(dataset)*0.8)
    #num_test= len(dataset) -num_train
    #train_data, test_data = torch.utils.data.random_split(dataset, [num_train, num_test])
    train_iter = tud.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_data.collate_fn, drop_last=True)
    test_iter = tud.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=test_data.collate_fn, drop_last=True)
    valid_iter = test_iter

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    optimizer = torch.optim.Adam(
        transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95) 
    if args.mode == 'train':
        transformer = transformer.to(DEVICE)

        if args.loadmodel:
            transformer.load_state_dict(torch.load(args.path))


        min_loss, val_loss = float('inf'), float('inf')
        for epoch in range(1, NUM_EPOCHS+1):
            start_time = time.time()
            train_loss = train_epoch(transformer, train_iter, optimizer)
            scheduler.step()
            end_time = time.time()
            if (epoch+1)%10==0:
                torch.save(transformer.state_dict(), args.path+'_'+str(epoch+1))
                print('Model saved every 10 epoches.') 
            
            if (epoch+1)%1==0:
                val_loss = evaluate(transformer, valid_iter)
                if val_loss < min_loss:
                    min_loss = val_loss
                    torch.save(transformer.state_dict(), args.path)
                    print('Model saved!') 

            print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "
                f"Epoch time = {(end_time - start_time):.3f}s"))
    
    elif args.mode == 'finetune':
        from Mol_target_dataloader.utils import read_csv_file
        import Mol_target_dataloader.dataset as md

        mol_list1, target_list = zip(*read_csv_file('Mol_target_dataloader/target.smi', num_fields=2))
        #vocabulary = mv.create_vocabulary(smiles_list=mol_list, tokenizer=mv.SMILESTokenizer())
        finetune_dataset = md.Dataset(mol_list1, target_list, vocabulary, mv.SMILESTokenizer())
        num_train= int(len(finetune_dataset)*0.8)
        num_test= len(finetune_dataset) -num_train
        train_data, val_data = torch.utils.data.random_split(finetune_dataset, [num_train, num_test])

        train_iter = tud.DataLoader(train_data, args.batch_size, collate_fn=finetune_dataset.collate_fn, shuffle=True)
        val_iter = tud.DataLoader(val_data, args.batch_size, collate_fn=finetune_dataset.collate_fn, shuffle=True)
        transformer = transformer.to(DEVICE)
        transformer.load_state_dict(torch.load(args.path))

        min_loss, val_loss = float('inf'), float('inf')
        for epoch in range(1, NUM_EPOCHS+1):
            start_time = time.time()
            train_loss = train_epoch(transformer, train_iter, optimizer)
            scheduler.step()
            end_time = time.time()
            if (epoch+1)%1==0:
                val_loss = evaluate(transformer, val_iter)
                if val_loss < min_loss:
                    min_loss = val_loss
                    torch.save(transformer.state_dict(), args.path)
                    print('Model saved!')

            print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "
                f"Epoch time = {(end_time - start_time):.3f}s"))

        
    elif args.mode == 'infer':
        if args.device == 'cpu':
            transformer.load_state_dict(torch.load(args.path,  map_location=torch.device('cpu')))
        else:
            transformer.load_state_dict(torch.load(args.path))
        device = args.device
        transformer.to(device)
        transformer.eval()
        _target = args.target
        print('Target: {0}'.format(_target))
        for i in range(3):
            ybar = greedy_decode(transformer, max_len=100, start_symbol=BOS_IDX, target=_target).flatten()
            #print(ybar)
            ybar = mv.SMILESTokenizer().untokenize(vocabulary.decode(ybar.to('cpu').data.numpy()))
            #print('prediction')
            print(ybar)
       
