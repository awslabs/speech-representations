# Speech Representations

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Models and code for deep learning representations developed by the AWS AI Speech team:

- [DeCoAR (self-supervised contextual representations for speech recognition)](https://arxiv.org/abs/1912.01679)
- **Coming soon:** BERTphone (phonetically-aware acoustic BERT for speaker and language recognition)

We also support other pre-trained models, namely [wav2vec](https://github.com/pytorch/fairseq/tree/master/examples/wav2vec).


## Installation

We provide a library and CLI to featurize speech utterances. We hope to release training/fine-tuning code in the future.

[Kaldi](https://github.com/kaldi-asr/kaldi) should be installed to `kaldi/`, or `$KALDI_ROOT` should be set.

We expect Python 3.6+. Our models are defined in MXNet; we may support export to PyTorch in the future. Clone this repository, then:
```sh
pip install -e .
pip install mxnet-mkl~=1.6.0   # ...or mxnet-cu102mkl for GPU w/ CUDA 10.2, etc.
pip install torch fairseq      # optional; for featurizing with wav2vec
```


## Pre-trained models

First, download the model weights:
```sh
mkdir artifacts
# For DeCoAR trained on LibriSpeech (257M)
wget -qO- https://apache-mxnet.s3-us-west-2.amazonaws.com/gluon/models/decoar-encoder-29b8e2ac.zip | zcat > artifacts/decoar-encoder-29b8e2ac.params
# For wav2vec trained on LibriSpeech (311M)
wget -P artifacts/ https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_large.pt
```
We support featurizing individual files with the CLI:
```sh
speech-reps featurize --model {decoar,wav2vec} --in-wav <input_file>.wav --out-npy <output_file>.npy
# --params <file>: load custom weights (otherwise use `artifacts/`)
# --gpu <int>:     use GPU (otherwise use CPU)
```
or in code:
```sh
from speech_reps.featurize import DeCoARFeaturizer
# Load the model on GPU 0
featurizer = DeCoARFeaturizer('artifacts/decoar-encoder-29b8e2ac.params', gpu=0)
# Returns a (time, feature) NumPy array
data = featurizer.file_to_feats('my_wav_file.wav')
```

 We plan to support Kaldi `.scp` and `.ark` files soon. For now, batches can be processed with the underlying `featurizer._model`.


## References

If you found our package or pre-trained models useful, please cite the relevant work:

**[DeCoAR](https://arxiv.org/abs/1912.01679)**
```
@inproceedings{decoar,
  author    = {Shaoshi Ling and Yuzong Liu and Julian Salazar and Katrin Kirchhoff},
  title     = {Deep Contextualized Acoustic Representations For Semi-Supervised Speech Recognition},
  booktitle = {{ICASSP}},
  pages     = {6429--6433},
  publisher = {{IEEE}},
  year      = {2020}
}
```
**BERTphone**
```
@inproceedings{bertphone,
  author    = {Shaoshi Ling and Julian Salazar and Yuzong Liu and Katrin Kirchhoff},
  title     = {BERTphone: Phonetically-aware Encoder Representations for Speaker and Language Recognition},
  booktitle = {{Speaker Odyssey}},
  publisher = {{ISCA}},
  year      = {2020}
}
```
