import numpy as np
import argparse
import torch
import pandas as pd
label_list = ['0','1']
def get_tokens(token_ids):
        #print ("ids:", len(ids))
        tokens = "CSTPAGNDEQHRKMILVFYWX"
        ids = [23, 10, 15, 16, 6, 7, 17, 14, 9, 18, 22, 13, 12, 21, 11, 5, 8, 19, 20, 24, 25]
        
        token_id_map = {}
        for i in range(len(tokens)):
            token_id_map[ids[i]] = tokens[i]
        token_id_map[2] = 'X'
        token_id_map[3] = 'X'
        token_id_map[0] = 'X'
        returned = []
        for token_id in token_ids:
            temp = []
            for id in token_id:
                temp.append(token_id_map[id])
            returned.append(temp)
        return returned

def aa_features():
        # Meiler's features
        prop1 = [[1.77, 0.13, 2.43,  1.54,  6.35, 0.17, 0.41],
             [1.31, 0.06, 1.60, -0.04,  5.70, 0.20, 0.28],
             [3.03, 0.11, 2.60,  0.26,  5.60, 0.21, 0.36],
             [2.67, 0.00, 2.72,  0.72,  6.80, 0.13, 0.34],
             [1.28, 0.05, 1.00,  0.31,  6.11, 0.42, 0.23],
             [0.00, 0.00, 0.00,  0.00,  6.07, 0.13, 0.15],
             [1.60, 0.13, 2.95, -0.60,  6.52, 0.21, 0.22],
             [1.60, 0.11, 2.78, -0.77,  2.95, 0.25, 0.20],
             [1.56, 0.15, 3.78, -0.64,  3.09, 0.42, 0.21],
             [1.56, 0.18, 3.95, -0.22,  5.65, 0.36, 0.25],
             [2.99, 0.23, 4.66,  0.13,  7.69, 0.27, 0.30],
             [2.34, 0.29, 6.13, -1.01, 10.74, 0.36, 0.25],
             [1.89, 0.22, 4.77, -0.99,  9.99, 0.32, 0.27],
             [2.35, 0.22, 4.43,  1.23,  5.71, 0.38, 0.32],
             [4.19, 0.19, 4.00,  1.80,  6.04, 0.30, 0.45],
             [2.59, 0.19, 4.00,  1.70,  6.04, 0.39, 0.31],
             [3.67, 0.14, 3.00,  1.22,  6.02, 0.27, 0.49],
             [2.94, 0.29, 5.89,  1.79,  5.67, 0.30, 0.38],
             [2.94, 0.30, 6.47,  0.96,  5.66, 0.25, 0.41],
             [3.21, 0.41, 8.08,  2.25,  5.94, 0.32, 0.42],
             [0.00, 0.00, 0.00,  0.00,  0.00, 0.00, 0.00]]
        return np.array(prop1)

def one_to_number(res_str):
        aa_s = "CSTPAGNDEQHRKMILVFYWX"
        return [aa_s.index(r) for r in res_str]

def gen_miller_features (res_seq_one):	
    ints = one_to_number(res_seq_one)
    feats = aa_features()[ints]
    return feats

def get_fscore(true_dev, pred_labels_dev): 
    total_dev = 0 
    cor_dev = 0 
    predict_1 = 0 
    true_1 = 0 
    cor_1 = 0 
    #print ("TRUE:", true_dev) 
    #print ("PRED:", pred_labels_dev) 
    for x in range(len(true_dev)): 
        if true_dev[x] == pred_labels_dev[x]: 
            cor_dev += 1 
 
        total_dev += 1 
        if true_dev[x] == '1': 
            true_1 += 1 
            if true_dev[x] == pred_labels_dev[x]: 
                cor_1 += 1 
        if pred_labels_dev[x] == '1': 
            predict_1 += 1 
    prec = 0.0 
    rec = 0.0 
    f1 = 0.0 
    acc = float(cor_dev)/total_dev 
    if predict_1 != 0: 
        prec = float(cor_1)/predict_1 
    if true_1 != 0: 
        rec = float(cor_1)/true_1 
    if prec != 0.0 and rec != 0.0: 
        f1 =  2*prec*rec/(prec+rec)
    return acc, prec, rec, f1
def add_space(st):
  return ' '.join(list(st))
  #return 'yah '+st

def gen_label(chain,cdrs,cdrlbls):
  labels = np.ones((len(chain)),dtype=int)*-100
  for cd,lb in zip(cdrs,cdrlbls):
    #print ("cd:", cd)
    start = chain.find(cd) ; end = start+len(cd)
    if start ==-1:
      return None
    labels[start:end] = [int(l) for l in lb.split()]
  return labels

def prepare_data(file_name):
    df2 = pd.read_csv(file_name)
    df2.dropna(inplace=True)

    seq_heavy = df2['heavy'].tolist()
    seq_light = df2['light'].tolist()
    cdrh1 =  df2['cdrh1'].tolist()
    cdrh2 = df2['cdrh2'].tolist()
    cdrh3 = df2['cdrh3'].tolist()
    cdrl1 = df2['cdrl1'].tolist()
    cdrl2 = df2['cdrl2'].tolist()
    cdrl3 = df2['cdrl3'].tolist()

    ph1 = df2['ph1'].tolist()
    ph2 = df2['ph2'].tolist()
    ph3 = df2['ph3'].tolist()

    pl1 = df2['pl1'].tolist()
    pl2 = df2['pl2'].tolist()
    pl3 = df2['pl3'].tolist()

    labels_heavy = []
    sequences = []
    for i,seq in enumerate(seq_heavy):
        if type(seq)!= type('s'):
            continue
        lab = gen_label(seq,[cdrh1[i],cdrh2[i],cdrh3[i]],[ph1[i],ph2[i],ph3[i]])
        if lab is None:
            continue
        sequences.append(seq)
        labels_heavy.append(lab)
    labels_light = []
    for i,seq in enumerate(seq_light):
        if type(seq)!= type('s'):
            continue
        lab = gen_label(seq,[cdrl1[i],cdrl2[i],cdrl3[i]],[pl1[i],pl2[i],pl3[i]])
        if lab is None:
            continue
        sequences.append(seq)
        labels_light.append(lab)

    sequences = [add_space(i) for i in sequences]
    labels = labels_heavy+labels_light
    return sequences, labels

def add_labels(encodings,unproc_labels):
  labels = []
  for i in range(len(encodings['input_ids'])):
    l = [-100]+unproc_labels[i].tolist()+[-100]*(len(encodings['input_ids'][i])-1-len(unproc_labels[i]))
    labels.append(l)
  return labels

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def extend_list(list1):
    #print ("list1:", list1)
    result = []
    for el in list1:
        result.extend(el)
    return result


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    #results = metric.compute(predictions=true_predictions, references=true_labels)
    acc, prec, rec, f1 = get_fscore(extend_list(true_labels), extend_list(true_predictions))
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "accuracy": acc,
    }

parser = argparse.ArgumentParser(description='Paratope prediction model')
def addArgument():
    parser.add_argument('--data', default="parapred_dataset.csv", type=str,
                    help='dataset to train and evaluate')
    parser.add_argument('--lstm',  default=1, type=int,
                    help='if add LSTM layer on the top(0 for no, 1 for yes')
    parser.add_argument('--restrictedLoss',  default=1, type=int,
                    help='if restrict the loss calculation on CDR section(1 for yes, 0 for no)')
    parser.add_argument('--lstm_dimension',  default=1024, type=int,
                    help='the dimension of lstm layer')
    parser.add_argument('--num_epochs', default=15, type=int,
                    help='number of epochs')
    parser.add_argument('--learning_rate', default=2e-6, type=float,
                    help='learning rate of the model')
    parser.add_argument('--train_predict', default=0, type=int,
                    help='train with large data set and predict')
    parser.add_argument('--predict_data', default='parapred_dataset.csv', type=str,
                    help='dataset to predict')
    parser.add_argument('--predict_only', default=0, type=int,
                    help='load model and predict')
    parser.add_argument("--predict_model", type=str,
                    help='combine use with predict_only, specify the location of the model')
def getArgs():
    args = parser.parse_args()
    print ("ARGS:", args)
    return args
