import pandas as pd
from sys import argv



if len(argv)<2:
    print('not enough args')
    exit(2)
    
    
df1 = pd.read_csv(argv[1])
df2 = pd.read_csv(argv[2])

seqs = df1['heavy'].tolist()


for i,seq in enumerate(seqs):
    try:
        pred = df2[df2['seq'] == ' '.join(list(seq))]['predictions'].tolist()[0]
    except:
        print(i)
        print(df2[df2['seq'] == ' '.join(list(seq))]['predictions'].tolist())
        print(seq)
        continue
    code = df1[df1['heavy'] == seq]['pdb'].tolist()[0]
    cdrh1 = df1[df1['heavy'] == seq]['cdrh1'].tolist()[0]
    cdrh2 = df1[df1['heavy'] == seq]['cdrh2'].tolist()[0]
    cdrh3 = df1[df1['heavy'] == seq]['cdrh3'].tolist()[0]


    
    ph1 = ' '.join(pred.split()[0:len(cdrh1)])
    ph2 = ' '.join(pred.split()[len(cdrh1):len(cdrh1)+len(cdrh2)])  
    ph3 = ' '.join(pred.split()[len(cdrh1)+len(cdrh2):len(cdrh1)+len(cdrh2)+len(cdrh3)])
    
    df1.at[i,'ph1']=ph1
    df1.at[i,'ph2']=ph2
    df1.at[i,'ph3']=ph3


    #print(pred)
    
    
seqs = df1['light'].tolist()

for i,seq in enumerate(seqs):
    try:
        pred = df2[df2['seq'] == ' '.join(list(seq))]['predictions'].tolist()[0]
    except:
        print(i)
        print(df2[df2['seq'] == ' '.join(list(seq))]['predictions'].tolist())
        print(seq)
        continue
    code = df1[df1['light'] == seq]['pdb'].tolist()[0]
    cdrh1 = df1[df1['light'] == seq]['cdrl1'].tolist()[0]
    cdrh2 = df1[df1['light'] == seq]['cdrl2'].tolist()[0]
    cdrh3 = df1[df1['light'] == seq]['cdrl3'].tolist()[0]

    
    pl1 = ' '.join(pred.split()[0:len(cdrh1)])   
    pl2 = ' '.join(pred.split()[len(cdrh1):len(cdrh1)+len(cdrh2)])  
    pl3 = ' '.join(pred.split()[len(cdrh1)+len(cdrh2):len(cdrh1)+len(cdrh2)+len(cdrh3)])
    
    df1.at[i,'pl1']=pl1
    df1.at[i,'pl2']=pl2
    df1.at[i,'pl3']=pl3

df1.to_csv(argv[1]+'predicted_pars.csv')