# Commands

## Server
CUDA_VISIBLE_DEVICES=... python app.py -num_worker=4 -max_seq_len=512 -model_dir=../pytorch-pretrained-BERT/uncased/ -pooling_strategy=NONE -port 5555

## Client
python distclient.py 5555 questions

and

python distclient.py 5555 paragraphs

