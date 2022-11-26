# Speech Representations

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Models and code for deep learning representations developed by the AWS AI Speech team:

- [DeCoAR (self-supervised contextual representations for speech recognition)](https://arxiv.org/abs/1912.01679)
- [BERTphone (phonetically-aware acoustic BERT for speaker and language recognition)](https://www.isca-speech.org/archive/Odyssey_2020/abstracts/93.html)
- [DeCoAR 2.0 (deep contextualized acoustic representation with vector quantization)](https://arxiv.org/abs/2012.06659)

**NOTE: This repo is not actively maintained. For future experiments with DeCoAR and DeCoAR 2.0, we suggest using [the S3PRL speech toolkit](https://github.com/s3prl/s3prl), which has active and standardized featurizer/upstream/downstream wrappers for these models.**

## Installation

We provide a library and CLI to featurize speech utterances. We hope to release training/fine-tuning code in the future.

[Kaldi](https://github.com/kaldi-asr/kaldi) should be installed to `kaldi/`, or `$KALDI_ROOT` should be set.

We expect Python 3.6+. The BERTphone model are defined in MXNet and our DeCoAR models are defined in Pytorch. Clone this repository, then:
```sh
pip install -e .
# For DeCoAR
pip install torch fairseq
# For BERTphone
pip install mxnet-mkl~=1.6.0   # ...or mxnet-cu102mkl for GPU w/ CUDA 10.2, etc.
pip install gluonnlp # optional; for featurizing with bertphone
```


## Pre-trained models

First, download the model weights:
```sh
mkdir artifacts
cd artifacts
# For DeCoAR trained on LibriSpeech (257M)
wget https://github.com/awslabs/speech-representations/releases/download/decoar/checkpoint_decoar.pt
# For BERTphone 8KHz (λ=0.2) trained on Fisher
wget https://github.com/awslabs/speech-representations/releases/download/bertphone/bertphone_fisher_02-87159543.params
# For Decoar 2.0:
wget https://github.com/awslabs/speech-representations/releases/download/decoar2/checkpoint_decoar2.pt

```
We support featurizing individual files with the CLI:
```sh
speech-reps featurize --model {decoar,bertphone,decoar2} --in-wav <input_file>.wav --out-npy <output_file>.npy
# --params <file>: load custom weights (otherwise use `artifacts/`)
# --gpu <int>:     use GPU (otherwise use CPU)
```
or in code:
```sh
from speech_reps.featurize import DeCoARFeaturizer
# Load the model on GPU 0
featurizer = DeCoARFeaturizer('artifacts/checkpoint_decoar.pt', gpu=0)
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
