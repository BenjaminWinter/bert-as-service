# -*- coding: utf-8 -*-
# encoding=utf8
import sys
import time
import json as json
from joblib import Parallel, delayed
from service.client import BertClient
from tqdm import tqdm
import gzip
import pickle
from bert.tokenization import FullTokenizer
import numpy as np
import uuid

def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]
        
def encode(examples):
    tokenizer = FullTokenizer('../pytorch-pretrained-BERT/uncased/vocab.txt')
    bc = BertClient(ip='localhost', port=int(sys.argv[1]))
    batch = []
    batch_keys = []
    batch_token_lens = []
    encoded = {}
    for i in examples:
        for (k, v) in i.items():
            batch.append(v)
            batch_keys.append(k)
            batch_token_lens.append(len(tokenizer.tokenize(v)))
    be = bc.encode(batch).astype('float32')
    #print(be[0])
    be = be.tolist()
    res = []
    for j , code in enumerate(be):
        if batch_token_lens[j] < len(be[j]):
            clipped = be[j][1:batch_token_lens[j] + 1]
        else:
            clipped = be[j]
        #print('({},{})'.format(len(clipped), len(clipped[0])))
        #encoded[batch_keys[j]] = clipped
        uid = uuid.uuid4().hex
        res.append({batch_keys[j]: uid})
        np.save((sys.argv[2] + '/' + uid + ".npy").encode('utf-8').decode('utf-8'), np.array(clipped).astype('float32'))
    return res
     
encoded = {}
data = {}

dtrain = json.load(open("../hotpot/hotpot_train_v1.json", encoding="utf-8"))
ddev = json.load(open("../hotpot/hotpot_dev_distractor_v1.json", encoding="utf-8"))
if sys.argv[2] == 'questions':
    for ds in [dtrain, ddev]:
        for dp in ds:
            data[dp['_id']] = dp['question']
else:
    for ds in [dtrain, ddev]:
        for dp in ds:
            for p in dp['context']:
             data[p[0]] = "<t> " + p[0] + " <t> " + ' '.join(p[1])

dlist = [{k:data[k]} for k in sorted(data)]
#dlist = dlist[:20000]
#dlist = list(chunks(dlist, len(dlist)//10))[int(sys.argv[4])]
bc = BertClient(ip='localhost', port=int(sys.argv[1]))

batch = []
batch_keys = []
batch_token_lens = []
lendata = len(data.items())

encoded = Parallel(n_jobs=20)(delayed(encode)(chunk) for chunk in tqdm(list(chunks(dlist, 160))))
encoded = [item for sublist in encoded for item in sublist]
print("Saved items: " + str(len(encoded)))
encoded = { k: v for d in encoded for k, v in d.items() }
json.dump(encoded, open(sys.argv[2] + "toid.json", encoding="utf-8", mode="w"))
    