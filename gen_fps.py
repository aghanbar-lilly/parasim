from transformers import BertModel, BertTokenizer
import re
import numpy as np
import pandas as pd
from sys import argv
from tqdm import tqdm,trange

if len(argv)<1:
    print('not enough args')
    exit(2)

tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
model = BertModel.from_pretrained("Rostlab/prot_bert")


def embedderat(seq,idcs):
    embs = []

    s = ' '.join(seq)
    encoded_input = tokenizer(s, return_tensors='pt')
    output = model(**encoded_input)
    e = output.to_tuple()[0].detach().numpy()[0]
    idx = np.array(idcs,dtype=int)+1 #adding one because of the start token
    e = e[idcs].mean(axis=0) #average the features of the residues from the query indices
    embs.append(e)
    return embs

def weighted_embedderat(seq,idcs,weights):
    embs = []

    s = ' '.join(seq)
    encoded_input = tokenizer(s, return_tensors='pt')
    output = model(**encoded_input)
    e = output.to_tuple()[0].detach().numpy()[0]
    idx = np.array(idcs,dtype=int)+1 #adding one because of the start token
    e =np.sum(e[idcs]*np.array(weights).reshape(-1,1),axis=0)/np.sum(weights)
    #e = e[idcs].mean(axis=0) #average the features of the residues from the query indices
    embs.append(e)
    return embs
    
def get_indices(seq,cdrs,par_masks):
  indices = []
  
  for j,cdr in enumerate(cdrs):
    start = seq.find(cdr)
    if start == -1:
      print(start)
      return None
    par_mask = list(map(float,par_masks[j].split()))
    for i,par in enumerate(par_mask):
      if par >= 0.5:
        indices.append(start+i)
  return indices

def get_weighted_indices(seq,cdrs,par_masks):
  indices = []
  weights = []
  for j,cdr in enumerate(cdrs):
    start = seq.find(cdr)
    if start == -1:
      print(start)
      return None
    par_mask = list(map(float,par_masks[j].split()))
    for i,par in enumerate(par_mask):
      if par >= -1.:
        indices.append(start+i)
        weights.append(par)
        #weights.append(1)
  return indices,weights
  
  
def gen_emb(df):
    embs = {}

    pbar = tqdm(total=len(df))
    for k,row in df.iterrows():
        seqh = row['heavy']
        cdrh = row[['cdrh1','cdrh2','cdrh3']].tolist()
        ph = row[['ph1','ph2','ph3']].tolist()
        #ind_h = get_indices(seqh,cdrh,ph)
        ind_h,w_h = get_weighted_indices(seqh,cdrh,ph)
        #print(k,ind_h)
        if ind_h is None:
            continue
        #emb_h = embedderat(seqh,ind_h)
        emb_h = weighted_embedderat(seqh,ind_h,w_h)
        seql = row['light']
        cdrl = row[['cdrl1','cdrl2','cdrl3']].tolist()
        pl = row[['pl1','pl2','pl3']].tolist()
        #ind_l = get_indices(seql,cdrl,pl)
        ind_l,w_l = get_weighted_indices(seql,cdrl,pl)
        if ind_l is None:
            continue
        #emb_l = embedderat(seql,ind_l)
        emb_l = weighted_embedderat(seql,ind_l,w_l)
        emb = np.concatenate([emb_h,emb_l],axis=-1)[0]
        embs[row['pdb']] = emb
        pbar.update(1)
    pbar.close()
        

    return embs



df = pd.read_csv(argv[1])
df =df.drop_duplicates(subset=['heavy','light'])
df.dropna(inplace=True)

embs = gen_emb(df)



np.save('fps.npy',embs, allow_pickle=True)