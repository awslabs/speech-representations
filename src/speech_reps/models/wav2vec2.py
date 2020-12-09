# This wrapper code is based on
#     https://github.com/pytorch/fairseq/blob/master/examples/wav2vec/wav2vec_featurize.py
# and is subject to the MIT license:
# 
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model
from torch import nn


class PretrainedWav2Vec2Model(nn.Module):
    def __init__(self, fname):
        super().__init__()

        checkpoint = torch.load(fname, map_location=torch.device('cpu'))
        self.args = checkpoint["args"]
        model = Wav2Vec2Model.build_model(self.args, None)
        model.load_state_dict(checkpoint["model"])
        model.eval()

        self.model = model

    def forward(self, x):
        with torch.no_grad():
            x, padding = self.model.extract_features(x, None)

        return x


class Wav2Vec2():
    """ Lightweight wrapper around a fairspeech embedding model """

    def __init__(self, fname, gpu):
        self.gpu = gpu
        self.model = PretrainedWav2Vec2Model(fname)
        if self.gpu is not None:
            self.model = self.model.cuda(gpu)

    def __call__(self, x):
        x = torch.from_numpy(x).float()
        if self.gpu is not None:
            x = x.cuda(self.gpu)
        with torch.no_grad():
            z = self.model(x.unsqueeze(0))

        return z.squeeze(0).cpu().numpy()
