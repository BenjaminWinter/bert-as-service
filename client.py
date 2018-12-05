import sys
import time
import ujson as json
from joblib import Parallel, delayed
from service.client import BertClient
from tqdm import tqdm
import gzip
import pickle
from bert.tokenization import FullTokenizer


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
    be = bc.encode(batch).astype('float16')
    #print(be[0])
    be = be.tolist()
    for j , code in enumerate(be):
        if batch_token_lens[j] < len(be[j]):
            clipped = be[j][:batch_token_lens[j]]
        else:
            clipped = be[j]
        #print('({},{})'.format(len(clipped), len(clipped[0])))
        encoded[batch_keys[j]] = clipped
    return encoded
     
encoded = {}
data = {}
dtrain = json.load(open("../hotpot/hotpot_train_v1.json"))
ddev = json.load(open("../hotpot/hotpot_dev_distractor_v1.json"))
if sys.argv[2] == 'questions':
    for ds in [dtrain, ddev]:
        for dp in ds:
            data[dp['_id']] = dp['question']
            data
else:
    for ds in [dtrain, ddev]:
        for dp in ds:
            for p in dp['context']:
             data[p[0]] = "<t> " + p[0] + " <t> " + ' '.join(p[1])

dlist = [{k:data[k]} for k in sorted(data)]
dlist = list(chunks(dlist, len(dlist)//10))[int(sys.argv[4])]
bc = BertClient(ip='localhost', port=int(sys.argv[1]))

batch = []
batch_keys = []
batch_token_lens = []
lendata = len(data.items())

encoded = Parallel(n_jobs=20)(delayed(encode)(chunk) for chunk in tqdm(list(chunks(dlist, 160))))
encoded = { k: v for d in encoded for k, v in d.items() }
#tokenizer = FullTokenizer('../pytorch-pretrained-BERT/uncased/vocab.txt')

#for i, (k, v) in enumerate(tqdm(data.items())):
#    batch.append(v)
#    batch_keys.append(k)
#    batch_token_lens.append(len(tokenizer.tokenize(v)))
#    if (i+1) % 64 == 0 or i == lendata - 1:
#        be = bc.encode(batch).astype('float16')
#        be = be.tolist()
#        for j , code in enumerate(be):
#            if batch_token_lens[j] < len(be[j]):
#                clipped = be[j][:batch_token_lens[j]]
#            #print('({},{})'.format(len(clipped), len(clipped[0])))
#            encoded[batch_keys[j]] = clipped
#        batch = []
#        batch_keys = []
        #print(len(encoded.keys()))
        

#with gzip.GzipFile(sys.argv[2] + ".gz", 'w') as f:
with open(sys.argv[3], 'w') as f:
    f.write(json.dumps(encoded, double_precision=5))
