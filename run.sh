export CUDA_VISIBLE_DEVICES=0,1,2,3
data=PATH_TO_DATA
modelfile=PATH_TO_SAVE_MODEL
expert_lagging=SET_EXPERT_LAGGING #1,3,5,7,9,11,13,15

# Fisrt-stage: Pertrain an equal-weight MoE Wait-k
python train.py  --ddp-backend=no_c10d ${data} --arch transformer --share-all-embeddings \
 --optimizer adam \
 --adam-betas '(0.9, 0.98)' \
 --clip-norm 0.0 \
 --lr 5e-4 \
 --lr-scheduler inverse_sqrt \
 --warmup-init-lr 1e-07 \
 --warmup-updates 4000 \
 --dropout 0.3 \
 --criterion label_smoothed_cross_entropy \
 --reset-dataloader --reset-lr-scheduler --reset-optimizer\
 --label-smoothing 0.1 \
 --encoder-attention-heads 8 \
 --decoder-attention-heads 8 \
 --left-pad-source False \
 --fp16 \
 --equal-weight \
 --expert-lagging ${expert_lagging} \
 --save-dir ${modelfile} \
 --max-tokens 4096 --update-freq 2

# Sencond-stage: Finetune MoE Wait-k with various expert weights
python train.py  --ddp-backend=no_c10d ${data} --arch transformer --share-all-embeddings \
 --optimizer adam \
 --adam-betas '(0.9, 0.98)' \
 --clip-norm 0.0 \
 --lr 5e-4 \
 --lr-scheduler inverse_sqrt \
 --warmup-init-lr 1e-07 \
 --warmup-updates 4000 \
 --dropout 0.3 \
 --criterion label_smoothed_cross_entropy \
 --reset-dataloader --reset-lr-scheduler --reset-optimizer\
 --label-smoothing 0.1 \
 --encoder-attention-heads 8 \
 --decoder-attention-heads 8 \
 --left-pad-source False \
 --fp16 \
 --expert-lagging ${expert_lagging} \
 --save-dir ${modelfile} \
 --max-tokens 4096 --update-freq 2