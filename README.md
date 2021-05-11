# 🤖 Danish Transformers 

Transformers constitute the current paradigm within Natural Language Processing (NLP) for a variety of downstream tasks. 
The number of transformers trained on danish corpora are limited, which is why the ambition of this repository is to provide the danish NLP community with alternatives to already established models. 
The pretrained models in this repository are trained using [🤗Transformers](https://github.com/huggingface/transformers), 
and checkpoints are made available at the HuggingFace model hub [here](https://huggingface.co/sarnikowski) for both [PyTorch](https://pytorch.org/) and [TensorFlow](https://www.tensorflow.org/).

## Model Weights

Details on how to use the models, can be found by clicking the architecture headers.

### [ConvBERT](convbert/README.md)

* [**`convbert-small-da-cased`**][danish-small-convbert-cased]: 12-layer, 256-hidden, 4-heads**
* [**`convbert-medium-small-da-cased`**][danish-medium-small-convbert-cased]: 12-layer, 384-hidden, 6-heads**

### [ELECTRA](electra/README.md)

* [**`electra-small-discriminator-da-256-cased`**][danish-small-electra-discriminator]: 12-layer, 256-hidden, 4-heads
* [**`electra-small-generator-da-256-cased`**][danish-small-electra-generator]: 12-layer, 64-hidden, 1-heads

**Pretrained using [ELECTRA][electra-paper] pretraining approach.

## Benchmarks

All downstream task benchmarks are evaluated on **finetuned** versions of the transformer models.
The dataset used for benchmarking both NER and POS tagging, is the Danish Dependency Treebank [UD-DDT](https://github.com/UniversalDependencies/UD_Danish-DDT).
All models were trained for 3 epochs on the train set.
All scores reported, are averages calculated from (N=5) random seed runs for each model, where `σ` refers to the standard deviation.

#### Named Entity Recognition

The table below shows the `F1-scores` on the test+dev set on the entities `LOC`, `ORG`, `PER` and `MISC` over (N=5) runs.

| **Model**                                                                         | **Params** | **LOC** |  **ORG** |  **PER** |  **MISC** |     **Micro AVG**      |
|-----------------------------------------------------------------------------------|------------|---------|----------|----------|-----------|------------------------|
| [**bert-base-multilingual-cased**][multilingual-base-bert]                        |   ~177M    |  87.02  |  75.24   |  91.28   |  75.94    |     83.18 (σ=0.81)     |
| [**danish-bert-uncased-v2**][danish-base-bert]                                    |   ~110M    |  87.40  |  75.43   |  93.92   |  76.21    |     84.19 (σ=0.75)     |
| +++++++++++++++++++++++++++                                                       |   +++++    |  ++++   |  ++++    |  ++++    |  ++++     |     +++++++++++        |
| [**convbert-medium-small-da-cased**][danish-medium-small-convbert-cased]          |   ~24.3M   |  88.61  |  75.97   |  90.15   |  77.07    |     83.54 (σ=0.55)     |
| [**convbert-small-da-cased**][danish-small-convbert-cased]                        |   ~12.9M   |  85.86  |  71.21   |  89.07   |  73.50    |     80.76 (σ=0.40)     |
| [**electra-small-da-cased**][danish-small-electra-discriminator]                  |   ~13.3M   |  86.30  |  70.05   |  88.34   |  71.31    |     79.63 (σ=0.22)     |

#### Part-of-speech Tagging

The table below shows the `F1-scores` on the test+dev set over (N=5) runs.

| **Model**                                                                         | **Params** |   **Micro AVG**   |
|-----------------------------------------------------------------------------------|------------|-------------------|
| [**bert-base-multilingual-cased**][multilingual-base-bert]                        |   ~177M    |   97.42 (σ=0.09)  |
| [**danish-bert-uncased-v2**][danish-base-bert]                                    |   ~110M    |   98.08 (σ=0.05)  |
| +++++++++++++++++++++++++++                                                       |   +++++    |   +++++++++++     |
| [**convbert-medium-small-da-cased**][danish-medium-small-convbert-cased]          |   ~24.3M   |   97.92 (σ=0.03)  |
| [**convbert-small-da-cased**][danish-small-convbert-cased]                        |   ~12.9M   |   97.32 (σ=0.03)  |
| [**electra-small-da-cased**][danish-small-electra-discriminator]                  |   ~13.3M   |   97.42 (σ=0.05)  |


## Data

The custom danish corpora used for pretraining, was created from the following sources:

* [Oscar](https://oscar-corpus.com/) ~9.5gb
* [Leipzig danish corpora](https://wortschatz.uni-leipzig.de/en/download) ~1.5gb
* [Wikipedia Monolingual Corpora](https://linguatools.org/tools/corpora/wikipedia-monolingual-corpora/) ~1.5gb
* [OPUS](http://opus.nlpl.eu/) ~3gb
* [DaNewsroom](https://github.com/danielvarab/da-newsroom) ~2gb

All characters in the corpus were transliterated to ASCII with the exception of `æøåÆØÅ§`.
Sources containing web crawled data, were cleaned of overrepresented NSFW ads and commercials.
The final dataset consists of `14,483,456` precomputed tensors of length 256.

## References

* Zihang Jiang, Weihao Yu, Daquan Zhou, Yunpeng Chen, Jiashi Feng, Shuicheng Yan 2020. [ConvBERT: Improving BERT with Span-based Dynamic Convolution](https://arxiv.org/abs/2008.02496)
* Kevin Clark, Minh-Thang Luong, Quoc V. Le, Christopher D. Manning. 2020. [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators][electra-paper]
* Pedro Javier Ortiz Suárez, Laurent Romary, Benoît Sagot. 2020. [A Monolingual Approach to Contextualized Word Embeddings for Mid-Resource Languages](https://arxiv.org/abs/2006.06202)
* Daniel Varab, Natalie Schluter. 2020. [DaNewsroom: A Large-scale Danish Summarisation Dataset](https://www.aclweb.org/anthology/2020.lrec-1.831/)
* Rasmus Hvingelby, Amalie B. Pauli, Maria Barrett, Christina Rosted, Lasse M. Lidegaard and Anders Søgaard. 2020. [DaNE: A Named Entity Resource for Danish](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.565.pdf)

## Cite this work

to cite this work please use
```
@inproceedings{danish-transformers,
  title = {Danish Transformers},
  author = {Tamimi-Sarnikowski, Philip},
  year = {2020},
  publisher = {{GitHub}},
  url = {https://github.com/sarnikowski}
}
```

## License

[![CC BY 4.0][cc-by-image]][cc-by]

This work is licensed under a [Creative Commons Attribution 4.0 International License][cc-by].

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png

[multilingual-base-bert]: https://huggingface.co/bert-base-multilingual-cased
[danish-base-bert]: https://github.com/botxo/nordic_bert
[danish-small-convbert-cased]: https://huggingface.co/sarnikowski/convbert-small-da-cased
[danish-medium-small-convbert-cased]: https://huggingface.co/sarnikowski/convbert-medium-small-da-cased
[danish-small-electra-discriminator]: https://huggingface.co/sarnikowski/electra-small-discriminator-da-256-cased
[danish-small-electra-generator]: https://huggingface.co/sarnikowski/electra-small-generator-da-256-cased

[electra-paper]: https://arxiv.org/abs/2003.10555
