# Procedure for reproducing experiments

## PTB/CTB data conversion

We split PTB (WSJ section of Treebank-3) and CTB (version 5.1) following [previous researches](https://aclanthology.org/D08-1059.pdf).
Dependency trees are converted from constituency trees, using the tool of [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/) (version 4.5.8).

### Preparation of PTB data

CoNLL-U format dependency trees of PTB are generated as below.
```
java -mx1g edu.stanford.nlp.trees.ud.UniversalDependenciesConverter -treeFile treebank > treebank.conllu
```

### Preparation of CTB data

The original data of CTB is encoded as GB2312. We first convert it to UTF-8.
```
iconv -c -f GB2313 -t UTF-8 < bracketed_tree > bracketed_tree.utf8
```

CoNLL-U format dependency trees of CTB are generated as below.
```
java -mx1g edu.stanford.nlp.trees.international.pennchinese.ChineseGrammaticalStructure -basic -keepPunct -conllx -treeFile bracketed_tree.utf8 > bracketed_tree.utf8.conllu
```

### Error recovery for CTB trees

Sentence 16046 of Section 1117 of CTB cannot be converted correctly. This is caused by spaces within a token.

Below is a part of the parse tree.
```
(NP (NP-APP (NP-PN (NR 育空))
	  (ADJP (JJ 知名))
	  (NP (NN 环保)
	      (NN 摄影家)))
  (NP-PN (NR Ken Madsen)))
```
There is a space in the token `Ken Madsen`, a person's name. Although the name is a single leaf node, the Stanford Converter recognizes it as two tokens separated by a space.
This causes an index out of boundary error.

We fix the original node
```
(NP-PN (NR Ken Madsen))
```
to
```
(NP-PN (NR Ken) (NR Madsen))
```
making it possible to generate a valid dependency tree in CoNLL-U format.

The relevant files are located under `/ctb_error_recovery`.

## Preparing and executing reproducing experiments

```bash
git clone https://github.com/hiroshi-matsuda-rit/parsing-as-tagging.git
cd parsing-as-tagging
python3 -m venv venv  # use python 3.10 (requires python < 3.11)
source venv/bin/activate
pip install -r requirements.txt

# preparing java library
sudo apt install -y 
mkdir ../malt
cd ../malt
curl -o maltparser-1.9.2.tar.gz http://maltparser.org/dist/maltparser-1.9.2.tar.gz
tar zxf maltparser-1.9.2.tar.gz
rm maltparser-1.9.2.tar.gz
cd -

ln -sf /home/sagyou/llmpp/treebanks data/
ln -sf /home/sagyou/llmpp/ud data/

python data/dep2bht.py

# ignore logs of bitsandbytes bug report info
for lang in English Chinese en ja zh ko ar fr de sl ; do python run.py vocab --lang $lang --tagger hexa ; done

CUDA_VISIBLE_DEVICES=0 python run.py train --lang English --max-depth 6 --tagger hexa --model bert --epochs 50  --batch-size 32 --lr 2e-5 --model-path xlnet-large-cased --output-path ./checkpoints.xlnet/ --use-tensorboard False &> log.ptb-xlnet &
CUDA_VISIBLE_DEVICES=1 python run.py train --lang English --max-depth 6 --tagger hexa --model bert --epochs 50  --batch-size 32 --lr 2e-5 --model-path bert-base-multilingual-cased --output-path ./checkpoints/ --use-tensorboard False &> log.ptb-bert &
CUDA_VISIBLE_DEVICES=2 python run.py train --lang Chinese --max-depth 6 --tagger hexa --model bert --epochs 50  --batch-size 32 --lr 2e-5 --model-path hfl/chinese-xlnet-mid --output-path ./checkpoints.xlnet/ --use-tensorboard False &> log.ctb-xlnet &
CUDA_VISIBLE_DEVICES=3 python run.py train --lang Chinese --max-depth 6 --tagger hexa --model bert --epochs 50  --batch-size 32 --lr 2e-5 --model-path bert-base-multilingual-cased --output-path ./checkpoints/ --use-tensorboard False &> log.ctb-bert &
wait
CUDA_VISIBLE_DEVICES=0 python run.py evaluate --lang English --max-depth 10 --tagger hexa --bert-model-path xlnet-large-cased --model-name English-hexa-bert-2e-05-50 --batch-size 64 --model-path ./checkpoints.xlnet/ &>> log.ptb-xlnet &
CUDA_VISIBLE_DEVICES=1 python run.py evaluate --lang English --max-depth 10 --tagger hexa --bert-model-path bert-base-multilingual-cased --model-name English-hexa-bert-2e-05-50 --batch-size 64 --model-path ./checkpoints/ &>> log.ptb-bert &
CUDA_VISIBLE_DEVICES=2 python run.py evaluate --lang Chinese --max-depth 10 --tagger hexa --bert-model-path hfl/chinese-xlnet-mid --model-name Chinese-hexa-bert-2e-05-50 --batch-size 64 --model-path ./checkpoints.xlnet/ &>> log.ctb-xlnet &
CUDA_VISIBLE_DEVICES=3 python run.py evaluate --lang Chinese --max-depth 10 --tagger hexa --bert-model-path bert-base-multilingual-cased --model-name Chinese-hexa-bert-2e-05-50 --batch-size 64 --model-path ./checkpoints/ &>> log.ctb-bert &


OUTPUT_PATH=./checkpoints/
MODEL_PATH=bert-base-multilingual-cased
CUDA_VISIBLE_DEVICES=0 python run.py train --lang en --max-depth 6 --tagger hexa --model bert --epochs 50  --batch-size 32 --lr 2e-5 --model-path $MODEL_PATH --output-path $OUTPUT_PATH --use-tensorboard False &> log.en-bert &
CUDA_VISIBLE_DEVICES=1 python run.py train --lang ja --max-depth 6 --tagger hexa --model bert --epochs 50  --batch-size 32 --lr 2e-5 --model-path $MODEL_PATH --output-path $OUTPUT_PATH --use-tensorboard False &> log.ja-bert &
CUDA_VISIBLE_DEVICES=2 python run.py train --lang zh --max-depth 6 --tagger hexa --model bert --epochs 50  --batch-size 32 --lr 2e-5 --model-path $MODEL_PATH --output-path $OUTPUT_PATH --use-tensorboard False &> log.zh-bert &
CUDA_VISIBLE_DEVICES=3 python run.py train --lang ko --max-depth 6 --tagger hexa --model bert --epochs 50  --batch-size 32 --lr 2e-5 --model-path $MODEL_PATH --output-path $OUTPUT_PATH --use-tensorboard False &> log.ko-bert &
CUDA_VISIBLE_DEVICES=4 python run.py train --lang ar --max-depth 6 --tagger hexa --model bert --epochs 50  --batch-size 32 --lr 2e-5 --model-path $MODEL_PATH --output-path $OUTPUT_PATH --use-tensorboard False &> log.ar-bert &
CUDA_VISIBLE_DEVICES=5 python run.py train --lang fr --max-depth 6 --tagger hexa --model bert --epochs 50  --batch-size 32 --lr 2e-5 --model-path $MODEL_PATH --output-path $OUTPUT_PATH --use-tensorboard False &> log.fr-bert &
CUDA_VISIBLE_DEVICES=6 python run.py train --lang de --max-depth 6 --tagger hexa --model bert --epochs 50  --batch-size 32 --lr 2e-5 --model-path $MODEL_PATH --output-path $OUTPUT_PATH --use-tensorboard False &> log.de-bert &
CUDA_VISIBLE_DEVICES=7 python run.py train --lang sl --max-depth 6 --tagger hexa --model bert --epochs 50  --batch-size 32 --lr 2e-5 --model-path $MODEL_PATH --output-path $OUTPUT_PATH --use-tensorboard False &> log.sl-bert &
wait
CUDA_VISIBLE_DEVICES=0 python run.py evaluate --lang en --max-depth 10 --tagger hexa --bert-model-path $MODEL_PATH --model-name en-hexa-bert-2e-05-50 --batch-size 64 --model-path $OUTPUT_PATH &>> log.en-bert &
CUDA_VISIBLE_DEVICES=1 python run.py evaluate --lang ja --max-depth 10 --tagger hexa --bert-model-path $MODEL_PATH --model-name ja-hexa-bert-2e-05-50 --batch-size 64 --model-path $OUTPUT_PATH &>> log.ja-bert &
CUDA_VISIBLE_DEVICES=2 python run.py evaluate --lang zh --max-depth 10 --tagger hexa --bert-model-path $MODEL_PATH --model-name zh-hexa-bert-2e-05-50 --batch-size 64 --model-path $OUTPUT_PATH &>> log.zh-bert &
CUDA_VISIBLE_DEVICES=3 python run.py evaluate --lang ko --max-depth 10 --tagger hexa --bert-model-path $MODEL_PATH --model-name ko-hexa-bert-2e-05-50 --batch-size 64 --model-path $OUTPUT_PATH &>> log.ko-bert &
CUDA_VISIBLE_DEVICES=4 python run.py evaluate --lang ar --max-depth 10 --tagger hexa --bert-model-path $MODEL_PATH --model-name ar-hexa-bert-2e-05-50 --batch-size 64 --model-path $OUTPUT_PATH &>> log.ar-bert &
CUDA_VISIBLE_DEVICES=5 python run.py evaluate --lang fr --max-depth 10 --tagger hexa --bert-model-path $MODEL_PATH --model-name fr-hexa-bert-2e-05-50 --batch-size 64 --model-path $OUTPUT_PATH &>> log.fr-bert &
CUDA_VISIBLE_DEVICES=6 python run.py evaluate --lang de --max-depth 10 --tagger hexa --bert-model-path $MODEL_PATH --model-name de-hexa-bert-2e-05-50 --batch-size 64 --model-path $OUTPUT_PATH &>> log.de-bert &
CUDA_VISIBLE_DEVICES=7 python run.py evaluate --lang sl --max-depth 10 --tagger hexa --bert-model-path $MODEL_PATH --model-name sl-hexa-bert-2e-05-50 --batch-size 64 --model-path $OUTPUT_PATH &>> log.sl-bert &
```

=== The rest part is the original README.md from the fork. ===

# Parsing as Tagging
<p align="center">
  <img src="https://github.com/rycolab/parsing-as-tagging/blob/main/header.jpg" width=400>
  <img src="https://github.com/rycolab/parsing-as-tagging/blob/main/header-hexa.png" width=400>
</p>
This repository contains code for training and evaluation of two papers:

- On Parsing as Tagging 
- Hexatagging: Projective Dependency Parsing as Tagging

## Setting Up The Environment
Set up a virtual environment and install the dependencies:
```bash
pip install -r requirements.txt
```

## Getting The Data
### Constituency Parsing
Follow the instructions in this [repo](https://github.com/nikitakit/self-attentive-parser/tree/master/data) to do the initial preprocessing on English WSJ and SPMRL datasets. The default data path is `data/spmrl` folder, where each file titled in `[LANGUAGE].[train/dev/test]` format.
### Dependency Parsing with Hexatagger
1. Convert CoNLL to Binary Headed Trees:
```bash
python data/dep2bht.py
```
This will generate the phrase-structured BHT trees in the `data/bht` directory. 
We placed the processed files already under the `data/bht` directory.

## Building The Tagging Vocab
In order to use taggers, we need to build the vocabulary of tags for in-order, pre-order and post-order linearizations. You can cache these vocabularies using:
```bash
python run.py vocab --lang [LANGUAGE] --tagger [TAGGER]
```
Tagger can be `td-sr` for top-down (pre-order) shift--reduce linearization, `bu-sr` for bottom-up (post-order) shift--reduce linearization,`tetra` for in-order, and `hexa` for hexatagging linearization.

## Training
Train the model and store the best checkpoint.
```bash
python run.py train --batch-size [BATCH_SIZE]  --tagger [TAGGER] --lang [LANGUAGE] --model [MODEL] --epochs [EPOCHS] --lr [LR] --model-path [MODEL_PATH] --output-path [PATH] --max-depth [DEPTH] --keep-per-depth [KPD] [--use-tensorboard]
```
- batch size: use 32 to reproduce the results
- tagger: `td-sr` or `bu-sr` or `tetra`
- lang: language, one of the nine languages reported in the paper
- model: `bert`, `bert+crf`, `bert+lstm`
- model path: path that pretrained model is saved
- output path: path to save the best trained model
- max depth: maximum depth to keep in the decoding lattice
- keep per depth: number of elements to keep track of in the decoding step
- use-tensorboard: whether to store the logs in tensorboard or not (true or false)

## Evaluation
Calculate evaluation metrics: fscore, precision, recall, loss.
```bash
python run.py evaluate --lang [LANGUAGE] --model-name [MODEL]  --model-path [MODEL_PATH] --bert-model-path [BERT_PATH] --max-depth [DEPTH] --keep-per-depth [KPD]  [--is-greedy]
```
- lang: language, one of the nine languages reported in the paper
- model name: name of the checkpoint
- model path: path of the checkpoint
- bert model path: path to the pretrained model
- max depth: maximum depth to keep in the decoding lattice
- keep per depth: number of elements to keep track of in the decoding step
- is greedy: whether or not use the greedy decoding, default is false

# Exact Commands for Hexatagging
The above commands can be used together with different taggers, models, and on different languages. To reproduce our Hexatagging results, here we put the exact commands used for training and evaluation of Hexatagger. 
## Train
### PTB (English)
```bash
CUDA_VISIBLE_DEVICES=0 python run.py train --lang English --max-depth 6 --tagger hexa --model bert --epochs 50 --batch-size 32 --lr 2e-5 --model-path xlnet-large-cased --output-path ./checkpoints/ --use-tensorboard True
# model saved at ./checkpoints/English-hexa-bert-3e-05-50
```
### CTB (Chinese)
```bash
CUDA_VISIBLE_DEVICES=0 python run.py train --lang Chinese --max-depth 6 --tagger hexa --model bert --epochs 50 --batch-size 32 --lr 2e-5 --model-path hfl/chinese-xlnet-mid --output-path ./checkpoints/ --use-tensorboard True
# model saved at ./checkpoints/Chinese-hexa-bert-2e-05-50
```

### UD
```bash
CUDA_VISIBLE_DEVICES=0 python run.py train --lang bg --max-depth 6 --tagger hexa --model bert --epochs 50  --batch-size 32 --lr 2e-5 --model-path bert-base-multilingual-cased --output-path ./checkpoints/ --use-tensorboard True
```

## Evaluate
### PTB
```bash
python run.py evaluate --lang English --max-depth 10 --tagger hexa --bert-model-path xlnet-large-cased --model-name English-hexa-bert-3e-05-50 --batch-size 64 --model-path ./checkpoints/
```

### CTB
```bash
python run.py evaluate --lang Chinese --max-depth 10 --tagger hexa --bert-model-path bert-base-chinese --model-name Chinese-hexa-bert-3e-05-50 --batch-size 64 --model-path ./checkpoints/
```
### UD
```bash
python run.py evaluate --lang bg --max-depth 10 --tagger hexa --bert-model-path bert-base-multilingual-cased --model-name bg-hexa-bert-1e-05-50 --batch-size 64 --model-path ./checkpoints/
```


## Predict
### PTB
```bash
python run.py predict --lang English --max-depth 10 --tagger hexa --bert-model-path xlnet-large-cased --model-name English-hexa-bert-3e-05-50 --batch-size 64 --model-path ./checkpoints/
```

# Citation
If you find this repository useful, please cite our papers:
```bibtex
@inproceedings{amini-cotterell-2022-parsing,
    title = "On Parsing as Tagging",
    author = "Amini, Afra  and
      Cotterell, Ryan",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.607",
    pages = "8884--8900",
}
```

```bibtex
@inproceedings{amini-etal-2023-hexatagging,
    title = "Hexatagging: Projective Dependency Parsing as Tagging",
    author = "Amini, Afra  and
      Liu, Tianyu  and
      Cotterell, Ryan",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-short.124",
    pages = "1453--1464",
}
```

