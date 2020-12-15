# Speech Representations

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Models and code for deep learning representations developed by the AWS AI Speech team:

- [DeCoAR (self-supervised contextual representations for speech recognition)](https://arxiv.org/abs/1912.01679)
- [BERTphone (phonetically-aware acoustic BERT for speaker and language recognition)](https://www.isca-speech.org/archive/Odyssey_2020/abstracts/93.html)
- [DeCoAR 2.0 (deep contextualized acoustic representation with vector quantization)](https://arxiv.org/abs/2012.06659)

We also support other pre-trained models, namely [wav2vec and wav2vec 2.0](https://github.com/pytorch/fairseq/tree/master/examples/wav2vec).


## Installation

We provide a library and CLI to featurize speech utterances. We hope to release training/fine-tuning code in the future.

[Kaldi](https://github.com/kaldi-asr/kaldi) should be installed to `kaldi/`, or `$KALDI_ROOT` should be set.

We expect Python 3.6+. Our models are defined in MXNet; we may support export to PyTorch in the future. Clone this repository, then:
```sh
pip install -e .
pip install mxnet-mkl~=1.6.0   # ...or mxnet-cu102mkl for GPU w/ CUDA 10.2, etc.
pip install gluonnlp # optional; for featurizing with bertphone
pip install torch fairseq      # optional; for featurizing with wav2vec, decaor 2.0
```


## Pre-trained models

First, download the model weights:
```sh
mkdir artifacts
# For DeCoAR trained on LibriSpeech (257M)
wget -qO- https://apache-mxnet.s3-us-west-2.amazonaws.com/gluon/models/decoar-encoder-29b8e2ac.zip | zcat > artifacts/decoar-encoder-29b8e2ac.params
# For wav2vec trained on LibriSpeech (311M)
wget -P artifacts/ https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_large.pt
# For wav2vec_2.0 trained on LibriSpeech (please download it from https://github.com/pytorch/fairseq/blob/master/examples/wav2vec/README.md)
# For BertPhone_8KHz(λ=0.2) trained on Fisher
wget -qO- https://apache-mxnet.s3-us-west-2.amazonaws.com/gluon/models/bertphone_fisher_02-87159543.zip | zcat > artifacts/bertphone_fisher_02-87159543.params
# For Decoar 2.0 to be released
```
We support featurizing individual files with the CLI:
```sh
speech-reps featurize --model {decoar,bertphone,wav2vec etc.} --in-wav <input_file>.wav --out-npy <output_file>.npy
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
**[BERTphone](https://www.isca-speech.org/archive/Odyssey_2020/abstracts/93.html)**
```
@inproceedings{bertphone,
  author    = {Shaoshi Ling and Julian Salazar and Yuzong Liu and Katrin Kirchhoff},
  title     = {BERTphone: Phonetically-aware Encoder Representations for Speaker and Language Recognition},
  booktitle = {{Speaker Odyssey}},
  publisher = {{ISCA}},
  year      = {2020}
}
```
**[DeCoAR 2.0](https://arxiv.org/abs/2012.06659)**
```
@misc{ling2020decoar,
      title={DeCoAR 2.0: Deep Contextualized Acoustic Representations with Vector Quantization}, 
      author={Shaoshi Ling and Yuzong Liu},
      year={2020},
      eprint={2012.06659},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```
