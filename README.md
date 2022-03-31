# Universal Simultaneous Machine Translation with Mixture-of-Experts Wait-k Policy

Source code for our EMNLP 2021 paper "Universal Simultaneous Machine Translation with Mixture-of-Experts Wait-k Policy" [[PDF](https://aclanthology.org/2021.emnlp-main.581.pdf)]

Our method is implemented based on the open-source toolkit [Fairseq](https://github.com/pytorch/fairseq).

## Requirements and Installation

- Python version = 3.6

- [PyTorch](http://pytorch.org/) version = 1.7

- Install fairseq:

  ```bash
  git clone https://github.com/ictnlp/MoE-Waitk.git
  cd MoE-Waitk
  pip install --editable ./
  ```


## Quick Start

### Data Pre-processing

We use the data of IWSLT15 English-Vietnamese (download [here](https://nlp.stanford.edu/projects/nmt/)) WMT16 English-Romanian (download [here](https://www.statmt.org/wmt16/)) and WMT15 German-English (download [here](https://www.statmt.org/wmt15/)).

For WMT16 English-Romanian and WMT15 German-English, we tokenize the corpus via [mosesdecoder/scripts/tokenizer/normalize-punctuation.perl](https://github.com/moses-smt/mosesdecoder) and apply BPE with 32K merge operations via [subword_nmt/apply_bpe.py](https://github.com/rsennrich/subword-nmt).

Then, we process the data into the fairseq format, adding `--joined-dictionary` for WMT15 German-English:

```bash
src=SOURCE_LANGUAGE
tgt=TARGET_LANGUAGE
train_data=PATH_TO_TRAIN_DATA
vaild_data=PATH_TO_VALID_DATA
test_data=PATH_TO_TEST_DATA
data=PATH_TO_DATA

# add --joined-dictionary for WMT16 English-Romanian and WMT15 German-English
fairseq-preprocess --source-lang ${src} --target-lang ${tgt} \
    --trainpref ${train_data} --validpref ${vaild_data} \
    --testpref ${test_data}\
    --destdir ${data} \
    --workers 20
```

### Training

Train MoE Wait-k Policy in two stage, according to the following command:

- For Transformer-Small with 4 attention heads: we set ***expert lagging*** = 1,6,11,16
- For Transformer-Base with 8 attention heads: we set ***expert lagging*** = 1,3,5,7,9,11,13,15
- For Transformer-Big with 16 attention heads: we set ***expert lagging*** = 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16

1. **First-stage**: fix the expert weights equal, and pre-train expert parameters.

```bash
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
```

2. **Second-stage**: jointly ﬁne-tune the parameters of experts and their weights.

```bash
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
```

### Inference

Evaluate the model with the following command:

```bash
export CUDA_VISIBLE_DEVICES=0
data=PATH_TO_DATA
modelfile = PATH_TO_SAVE_MODEL
ref_dir=PATH_TO_REFERENCE
testk=TEST_WAIT_K

# average last 5 checkpoints
python scripts/average_checkpoints.py --inputs ${modelfile} --num-update-checkpoints 5 --output ${modelfile}/average-model.pt 

# generate translation
python generate.py ${data} --path $modelfile/average-model.pt --batch-size 250 --beam 1 --left-pad-source False --fp16  --remove-bpe --test-wait-k ${testk} > pred.out

grep ^H pred.out | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > pred.translation
multi-bleu.perl -lc ${ref_dir} < pred.translation
```

## Citation

In this repository is useful for you, please cite as:

```Bibtex
@inproceedings{zhang-feng-2021-universal,
    title = "Universal Simultaneous Machine Translation with Mixture-of-Experts Wait-k Policy",
    author = "Zhang, Shaolei  and
      Feng, Yang",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.581",
    doi = "10.18653/v1/2021.emnlp-main.581",
    pages = "7306--7317",
}
```

