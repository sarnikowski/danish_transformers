# ELECTRA

The ELECTRA model was proposed in the paper [ELECTRA: Pre-training Text Encoders as Discriminators Rather than Generators](https://openreview.net/pdf?id=r1xMH1BtvB).
ELECTRA introduces a new pretraining approach, in which two transformers models are trained: the generator and the discriminator.
The generator is trained as a masked language model (MLM), where a number of tokens in an input sequence are corrupted,
and the generator is tasked with reconstructing the original sequence.
The discriminator on the other hand, is tasked with identifying which tokens were replaced by the generator in the sequence.
This approach has been shown to be highly effective in terms of compute, since the discriminator has to identify which tokens are
replaced by the generator over the entire sequence, rather than a small subset of tokens as in traditional MLM.

## Weights

* [**`electra-small-discriminator-da-256-cased`**][danish-small-electra]: 12-layer, 256-hidden, 4-heads

## Usage

Here is an example on how to load the model in [PyTorch](https://pytorch.org/) using the [🤗Transformers](https://github.com/huggingface/transformers) library:

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("sarnikowski/electra-small-discriminator-da-256-cased")
model = AutoModel.from_pretrained("sarnikowski/electra-small-discriminator-da-256-cased")
```

## Benchmarks

All downstream task benchmarks are evaluated on **finetuned** versions of the transformer models.
The dataset used for benchmarking both NER and POS tagging, is the Danish Dependency Treebank [UD-DDT](https://github.com/UniversalDependencies/UD_Danish-DDT).
All models were trained for 3 epochs on the train set with the same learning rate.
All scores reported, are averages calculated from (N=5) random seed runs for each model, where `σ` refers to the standard deviation.

#### Named entity recognition

The table below shows the average F1 scores on the test+dev set on the entities `LOC`, `ORG`, `PER` and `MISC`. Notice that `AVG` refers to the weighted average.

| **Model**                                                           | **Params** | **LOC** |  **ORG** |  **PER** |  **MISC** |     **AVG**      |
|---------------------------------------------------------------------|------------|---------|----------|----------|-----------|------------------|
| [**bert-base-multilingual-cased**][multilingual-base-bert]          |   ~177M    |  88.03  |  76.92   |  92.24   |  78.03    |  84.49 (σ=0.40)  |
| [**danish-bert-uncased-v2**][danish-base-bert]                      |   ~110M    |  88.25  |  75.42   |  94.05   |  75.73    |  84.20 (σ=0.32)  |
| [**electra-small-discriminator-da-256-cased**][danish-small-electra]|   ~13.3M   |  85.93  |  70.06   |  88.76   |  72.79    |  80.12 (σ=0.29)  |

#### Part-of-speech tagging

The table below shows the average F1 scores on the test+dev set. Notice that `F1-score` refers to the weighted average.

| **Model**                                                           | **Params** |      **F1-score**     |
|---------------------------------------------------------------------|------------|-----------------------|
| [**bert-base-multilingual-cased**][multilingual-base-bert]          |   ~177M    |     97.41 (σ=0.08)    |
| [**danish-bert-uncased-v2**][danish-base-bert]                      |   ~110M    |     98.05 (σ=0.06)    |
| [**electra-small-discriminator-da-256-cased**][danish-small-electra]|   ~13.3M   |     97.42 (σ=0.06)    |

## Data

The danish corpora used for pretraining was created from the following sources:

* [Oscar](https://oscar-corpus.com/) ~9.5gb
* [Leipzig danish corpora](https://wortschatz.uni-leipzig.de/en/download) ~1.5gb
* [Wikipedia Monolingual Corpora](https://linguatools.org/tools/corpora/wikipedia-monolingual-corpora/) ~1.5gb
* [OPUS](http://opus.nlpl.eu/) ~3gb
* [DaNewsroom](https://github.com/danielvarab/da-newsroom) ~2gb

All characters in the corpus were transliterated to ASCII with the exception of `æøåÆØÅ§`.
Sources containing web crawled data, were cleaned of overrepresented "dirty" ads and commercials.
The final dataset consists of `14,483,456` precomputed tensors of length 256.

## References

* Kevin Clark, Minh-Thang Luong, Quoc V. Le, Christopher D. Manning. 2020. [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555)
* Pedro Javier Ortiz Suárez, Laurent Romary, Benoît Sagot. 2020. [A Monolingual Approach to Contextualized Word Embeddings for Mid-Resource Languages](https://arxiv.org/abs/2006.06202)
* Daniel Varab, Natalie Schluter. 2020. [DaNewsroom: A Large-scale Danish Summarisation Dataset](https://www.aclweb.org/anthology/2020.lrec-1.831/)
* Rasmus Hvingelby, Amalie B. Pauli, Maria Barrett, Christina Rosted, Lasse M. Lidegaard and Anders Søgaard. 2020. [DaNE: A Named Entity Resource for Danish](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.565.pdf)

[multilingual-base-bert]: https://huggingface.co/bert-base-multilingual-cased
[danish-base-bert]: https://github.com/botxo/nordic_bert
[danish-small-electra]: https://huggingface.co/sarnikowski/electra-small-discriminator-da-256-cased
