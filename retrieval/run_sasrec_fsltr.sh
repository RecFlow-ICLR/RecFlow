set -x
set -e
set -o pipefail

tag="sasrec-fsltr-1st"

bs=4096
lr=1e-1
neg_num=200

flow_negs=realshow,coarse_neg,prerank_neg
flow_neg_nums=1,1,1

python -B -u run_sasrec_fsltr.py \
--epochs=1 \
--batch_size=${bs} \
--infer_batch_size=900 \
--emb_dim=8 \
--lr=${lr} \
--seq_len=50 \
--cuda="0" \
--print_freq=100 \
--neg_num=${neg_num} \
--flow_negs=${flow_negs} \
--flow_neg_nums=${flow_neg_nums} \
--tag=${tag} > "./logs/bs-${bs}_lr-${lr}_${neg_num}_${flow_negs}_${flow_neg_nums}_${tag}.log" 2>&1

python -B -u eval_sasrec_fsltr.py \
--epochs=1 \
--batch_size=${bs} \
--infer_recall_batch_size=900 \
--emb_dim=8 \
--lr=${lr} \
--seq_len=50 \
--cuda="0" \
--print_freq=100 \
--neg_num=${neg_num} \
--flow_negs=${flow_negs} \
--flow_neg_nums=${flow_neg_nums} \
--tag=${tag} >> "./logs/bs-${bs}_lr-${lr}_${neg_num}_${flow_negs}_${flow_neg_nums}_${tag}.log" 2>&1