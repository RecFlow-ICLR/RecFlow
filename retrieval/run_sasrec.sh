set -x
set -e
set -o pipefail

tag="sasrec-1st"

bs=4096
lr=1e-1
neg_num=200

python -B -u run_sasrec.py \
--epochs=1 \
--batch_size=${bs} \
--infer_batch_size=900 \
--emb_dim=8 \
--lr=${lr} \
--seq_len=50 \
--cuda="0" \
--print_freq=100 \
--neg_num=${neg_num} \
--tag=${tag} > "./logs/bs-${bs}_lr-${lr}_${neg_num}_${tag}.log" 2>&1

python -B -u eval_sasrec.py \
--epochs=1 \
--batch_size=${bs} \
--infer_recall_batch_size=900 \
--emb_dim=8 \
--lr=${lr} \
--seq_len=50 \
--cuda="0" \
--print_freq=100 \
--neg_num=${neg_num} \
--tag=${tag} >> "./logs/bs-${bs}_lr-${lr}_${neg_num}_${tag}.log" 2>&1