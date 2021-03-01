# ELECTRA

The ELECTRA model was proposed in the paper [ELECTRA: Pre-training Text Encoders as Discriminators Rather than Generators][electra-paper].
ELECTRA introduces a new pretraining approach, in which two transformers models are trained: the generator and the discriminator.
The generator is trained as a masked language model (MLM), where a number of tokens in an input sequence are corrupted,
and the generator is tasked with reconstructing the original sequence.
The discriminator on the other hand, is tasked with identifying which tokens were replaced by the generator in the sequence.
This approach has been shown to be highly effective in terms of compute, since the discriminator has to identify which tokens are
replaced by the generator over the entire sequence, rather than a small subset of tokens as in traditional MLM.

## Weights

* [**`electra-small-discriminator-da-256-cased`**][danish-small-electra-discriminator]: 12-layer, 256-hidden, 4-heads
* [**`electra-small-generator-da-256-cased`**][danish-small-electra-generator]: 12-layer, 64-hidden, 1-heads

## Usage

Here is an example on how to load the model in [PyTorch](https://pytorch.org/) using the [🤗Transformers](https://github.com/huggingface/transformers) library:

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("sarnikowski/electra-small-discriminator-da-256-cased")
model = AutoModel.from_pretrained("sarnikowski/electra-small-discriminator-da-256-cased")
```

[danish-small-electra-discriminator]: https://huggingface.co/sarnikowski/electra-small-discriminator-da-256-cased
[danish-small-electra-generator]: https://huggingface.co/sarnikowski/electra-small-generator-da-256-cased
[electra-paper]: https://arxiv.org/abs/2003.10555
