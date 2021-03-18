# ConvBERT

The ConvBERT model was proposed in the paper [ConvBERT: Improving BERT with Span-based Dynamic Convolution](https://arxiv.org/abs/2008.02496).

**Abstract:** *Pre-trained language models like BERT and its variants have recently achieved impressive performance in various natural language understanding tasks.
However, BERT heavily relies on the global self-attention block and thus suffers large memory footprint and computation cost. Although all its attention heads
query on the whole input sequence for generating the attention map from a global perspective, we observe some heads only need to learn local dependencies,
which means the existence of computation redundancy. We therefore propose a novel span-based dynamic convolution to replace these self-attention heads to
directly model local dependencies. The novel convolution heads, together with the rest self-attention heads, form a new mixed attention block that is more
efficient at both global and local context learning. We equip BERT with this mixed attention design and build a ConvBERT model.
Experiments have shown that ConvBERT significantly outperforms BERT and its variants in various downstream tasks, with lower training cost and fewer model parameters.
Remarkably, ConvBERTbase model achieves 86.4 GLUE score, 0.7 higher than ELECTRAbase, while using less than 1/4 training cost. Code and pre-trained models will be released.*

The models presented here are trained using [ELECTRA][electra-paper] pretraining approach, and are of the discriminator, unless stated otherwise.

## Weights

* [**`convbert-small-da-cased`**][danish-small-convbert-cased]: 12-layer, 256-hidden, 4-heads
* [**`convbert-medium-small-da-cased`**][danish-medium-small-convbert-cased]: 12-layer, 384-hidden, 6-heads

## Usage

Here is an example on how to load the model in [PyTorch](https://pytorch.org/) using the [🤗Transformers](https://github.com/huggingface/transformers) library:

```python
from transformers import ConvBertTokenizer, ConvBertModel

tokenizer = ConvBertTokenizer.from_pretrained("sarnikowski/convbert-medium-small-da-cased")
model = ConvBertModel.from_pretrained("sarnikowski/convbert-medium-small-da-cased")
```

[danish-small-convbert-cased]: https://huggingface.co/sarnikowski/convbert-small-da-cased
[danish-medium-small-convbert-cased]: https://huggingface.co/sarnikowski/convbert-medium-small-da-cased
[electra-paper]: https://arxiv.org/abs/2003.10555
