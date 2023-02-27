import numpy as np
import pandas as pd
import os
from sys import argv

if len(argv)<2:
    print('not enough args')
    exit(2)
    
def run_anarci(seq):
    os.system('ANARCI --scheme chothia -i {} > anarc.out'.format(seq))


def parse_cdr_txt(out,par,chaintype):

    with open(out) as fin:
        lines = fin.readlines()
    
    if chaintype == 'H':
        s1 = 26-2; e1 = 33+2
        s2 = 52-2; e2 = 57+2
        s3 = 95-2; e3 = 103+2
    elif chaintype == 'L':
        s1 = 24-2; e1 = 35+2
        s2 = 50-2; e2 = 57+2
        s3 = 89-2; e3 = 98+2
    else:
        return None
    cdrn1 = [i for i in range(s1,e1)]
    cdrn2 = [i for i in range(s2,e2)]
    cdrn3 = [i for i in range(s3,e3)]
    cdr1 = []
    cdr2 = []
    cdr3 = []
    p1 = []
    p2 = []
    p3 = []
    numbering = []
    for line in lines:
        if r'//' in line:
            break
        if '#' not in line:
            sp = line.split()
 
            n = int(sp[1])
            p = '0' #always set par to 0
                    
            if len(sp) not in [3,4]:
                print(len(sp))
                
            numbering.append([n,sp[-1],p])
            
            
    for n in numbering:
        if n[0] in cdrn1:
            cdr1.append(n[1])
            p1.append(n[2])
        elif n[0] in cdrn2:
            cdr2.append(n[1])
            p2.append(n[2])
        elif n[0] in cdrn3:
            cdr3.append(n[1])
            p3.append(n[2])
    
    def remove_dash(cdr,par):
        return [par[i] for i in range(len(par)) if cdr[i] != '-']
        
    p1 = remove_dash(cdr1,p1)
    p2 = remove_dash(cdr2,p2)
    p3 = remove_dash(cdr3,p3)
    cdr1 = ''.join(cdr1)
    cdr2 = ''.join(cdr2)
    cdr3 = ''.join(cdr3)
    cdr1 = cdr1.replace('-','')
    cdr2 = cdr2.replace('-','')
    cdr3 = cdr3.replace('-','')
    
    p1 = ' '.join(p1)
    p2 = ' '.join(p2)
    p3 = ' '.join(p3)
            
    return cdr1,cdr2,cdr3,p1,p2,p3




df = pd.read_csv(argv[1])
df = df.dropna()
#df.to_csv('qings_seqs.csv')
dic = {'pdb':[],'heavy':[],'light':[],'cdrh1':[],'cdrh2':[],'cdrh3':[],'cdrl1':[],'cdrl2':[],'cdrl3':[],'ph1':[],'ph2':[],'ph3':[],'pl1':[],'pl2':[],'pl3':[]}

for i,row in df.iterrows():
    pdb = row['pdb']
    
    heavy = row['heavy']
    run_anarci(heavy)
    cdrh1,cdrh2,cdrh3,ph1,ph2,ph3 = parse_cdr_txt('anarc.out','',"H")
    
    light = row['light']
    run_anarci(light)
    cdrl1,cdrl2,cdrl3,pl1,pl2,pl3 = parse_cdr_txt('anarc.out','',"L")
    
    dic['pdb'] += [pdb] ; dic['heavy'] += [heavy] ; dic['light'] += [light]
    dic['cdrh1']+= [cdrh1]
    dic['cdrh2']+= [cdrh2]
    dic['cdrh3']+= [cdrh3]
    dic['cdrl1']+= [cdrl1]
    dic['cdrl2']+= [cdrl2]
    dic['cdrl3']+= [cdrl3]
    
    

    
    dic['ph1'] += [ph1]
    dic['ph2'] += [ph2]
    dic['ph3'] += [ph3]
    
    dic['pl1'] += [pl1]
    dic['pl2'] += [pl2]
    dic['pl3'] += [pl3]
    
newdf = pd.DataFrame.from_dict(dic)
newdf.to_csv(argv[2])