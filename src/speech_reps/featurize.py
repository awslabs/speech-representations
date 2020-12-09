import argparse
import logging
import math
from pathlib import Path
from subprocess import check_call, CalledProcessError
from tempfile import TemporaryDirectory

import kaldi_io
import mxnet as mx
import numpy as np
import soundfile as sf

from .models.decoar import DeCoAR
from .models.wav2vec import Wav2Vec
from .models.wav2vec2 import Wav2Vec2
from .models.decoar2 import DeCoAR2
from .models.bertphone import BertPhone

class Featurizer:


    @classmethod
    def populate_parser(cls, parser):
        parser.add_argument('--model', type=str, choices=['decoar', 'decoar 2.0', 'bertphone', 'wav2vec'],
            help="Which model to featurize with")
        parser.add_argument('--params', type=str,
            help="Model parameter file")
        parser.add_argument('--gpu', type=int,
            help="GPU to use")
        # Valid pair: .wav --> .npy
        parser.add_argument('--in-wav', type=str,
            help="Input .wav file")
        parser.add_argument('--out-npy', type=str,
            help="Output .npy file")
        parser.set_defaults(func=cls.factory)


    @classmethod
    def factory(cls, args):

        params_file = None
        if args.params:
            params_file = Path(args.params)

        if args.model == 'decoar':
            featurizer = DeCoARFeaturizer(params_file, args.gpu)
        elif args.model == 'wav2vec':
            featurizer = Wav2VecFeaturizer(params_file, args.gpu)
        elif args.model == 'wav2vec_2.0':
            featurizer = Wav2Vec2Featurizer(params_file, args.gpu)
        elif args.model == 'decoar_2.0':
            featurizer = DeCoAR2Featurizer(params_file, args.gpu)
        elif args.model.startswith('bertphone'):
            featurizer = BertPhoneFeaturizer(params_file, args.gpu)


        if args.in_wav and args.out_npy:
            in_wav = Path(args.in_wav)
            out_npy = Path(args.out_npy)
            featurizer.wav_to_npy(in_wav, out_npy)
        else:
            raise ValueError("--in-wav, --out-npy must both be in use")


    def file_to_feats(self, file):
        """Takes an audio file and returns a np.array of (time, feature)"""
        return self._file_to_feats(Path(file))


    def wav_to_npy(self, in_wav, out_npy):
        feats = self.file_to_feats(in_wav)
        np.save(out_npy, feats)


    def _file_to_feats(self, file):
        # Expects a Path object
        raise NotImplementedError

class Wav2VecFeaturizer(Featurizer):


    def __init__(self, params_file, gpu=None):
        super().__init__()
        if params_file is None:
            params_file = Path('artifacts/wav2vec_large.pt')
        self._model = Wav2Vec(params_file, gpu)


    def _file_to_feats(self, file):
        wav, sr = sf.read(file)
        assert sr == 16e3
        __, feats = self._model(wav)
        return feats.transpose(1,0)

class Wav2Vec2Featurizer(Featurizer):


    def __init__(self, params_file, gpu=None):
        super().__init__()
        self._model = Wav2Vec2(params_file, gpu)


    def _file_to_feats(self, file):
        wav, sr = sf.read(file)
        assert sr == 16e3
        __, feats = self._model(wav)
        return feats.transpose(1,0)

class DeCoARFeaturizer(Featurizer):


    def __init__(self, params_file, gpu=None):
        super().__init__() 

        if params_file is None:
            params_file = Path('artifacts/decoar-encoder-29b8e2ac.params ')
        # Load the model
        self._model = DeCoAR(40, 1024, num_hidden=1024, num_layers=4, dropout=0)
        self._ctx = mx.gpu(gpu) if gpu is not None else mx.cpu()
        self._model.load_parameters(str(params_file), ignore_extra=True, ctx=self._ctx)
        self._model.hybridize(static_alloc=True)
        logging.info(self._model)


    def _file_to_feats(self, file):

        assert file.suffix == '.wav'
        # To support CMVN files in the future
        cmvn_spec = None

        def _run_cmd(cmd):
            logging.warn("Running {}".format(cmd))
            try:
                check_call(cmd, shell=True, universal_newlines=True)
            except CalledProcessError as e:
                logging.error("Failed with code {}:".format(e.returncode))
                logging.error(e.output)
                raise e

        with TemporaryDirectory() as temp_dir:

            temp_dir = Path(temp_dir)

            # Create config placeholder
            conf_file = temp_dir / 'fbank.conf'
            conf_file.write_text('--num-mel-bins=40\n')

            # Create SCP placeholder
            input_scp = temp_dir / 'input.scp'
            input_scp.write_text('file-0 {}\n'.format(file))

            # Compute speech features
            feat_ark = temp_dir / "feat.ark"
            feat_scp = temp_dir / "feat.scp"
            cmd = f"compute-fbank-feats --config={conf_file} scp:{input_scp} ark,scp:{feat_ark},{feat_scp}"
            _run_cmd(cmd)

            cmvn_scp = temp_dir / "cmvn.scp"
            if cmvn_spec is not None:
                # If CMVN specifier is provided, we create a dummy scp
                cmvn_scp.write_text("file-0 {cmvn_spec}\n")
            else:
                # Compute CMVN stats
                cmvn_ark = temp_dir / "cmvn.ark"
                cmd = f"compute-cmvn-stats scp:{feat_scp} ark,scp:{cmvn_ark},{cmvn_scp}"
                _run_cmd(cmd)

            # Apply CMVN
            final_ark = temp_dir / "final.ark"
            final_scp = temp_dir / "final.scp"
            cmd = f"apply-cmvn --norm-vars=true scp:{cmvn_scp} scp:{feat_scp} ark,scp:{final_ark},{final_scp}"
            _run_cmd(cmd)

            with final_scp.open('rb') as fp:
                feats = [features for _, features in kaldi_io.read_mat_scp(fp)][0]

        # Process data
        feats_new = feats
        # Turn the audio into a one-entry batch (TC --> TNC)
        data = mx.nd.expand_dims(mx.nd.array(feats_new, ctx=self._ctx), axis=1)
        data_len = mx.nd.array([data.shape[0]], ctx=self._ctx)

        vecs = self._model(data, data_len).flatten()
        reps = vecs.asnumpy()

        return reps

class DeCoAR2Featurizer(Featurizer):


    def __init__(self, params_file, gpu=None):
        super().__init__()

        self._model = DeCoAR2(params_file, gpu)


    def _file_to_feats(self, file):

        assert file.suffix == '.wav'
        # To support CMVN files in the future
        cmvn_spec = None

        def _run_cmd(cmd):
            logging.warn("Running {}".format(cmd))
            try:
                check_call(cmd, shell=True, universal_newlines=True)
            except CalledProcessError as e:
                logging.error("Failed with code {}:".format(e.returncode))
                logging.error(e.output)
                raise e

        with TemporaryDirectory() as temp_dir:

            temp_dir = Path(temp_dir)

            # Create config placeholder
            conf_file = temp_dir / 'fbank.conf'
            conf_file.write_text('--num-mel-bins=80\n')

            # Create SCP placeholder
            input_scp = temp_dir / 'input.scp'
            input_scp.write_text('file-0 {}\n'.format(file))

            # Compute speech features
            feat_ark = temp_dir / "feat.ark"
            feat_scp = temp_dir / "feat.scp"
            cmd = f"compute-fbank-feats --config={conf_file} scp:{input_scp} ark,scp:{feat_ark},{feat_scp}"
            _run_cmd(cmd)

            cmvn_scp = temp_dir / "cmvn.scp"
            if cmvn_spec is not None:
                # If CMVN specifier is provided, we create a dummy scp
                cmvn_scp.write_text("file-0 {cmvn_spec}\n")
            else:
                # Compute CMVN stats
                cmvn_ark = temp_dir / "cmvn.ark"
                cmd = f"compute-cmvn-stats scp:{feat_scp} ark,scp:{cmvn_ark},{cmvn_scp}"
                _run_cmd(cmd)

            # Apply CMVN
            final_ark = temp_dir / "final.ark"
            final_scp = temp_dir / "final.scp"
            cmd = f"apply-cmvn --norm-vars=true scp:{cmvn_scp} scp:{feat_scp} ark,scp:{final_ark},{final_scp}"
            _run_cmd(cmd)

            with final_scp.open('rb') as fp:
                feats = [features for _, features in kaldi_io.read_mat_scp(fp)][0]

        # Process data
        feats_new = feats

        reps = self._model(np.array(feats_new))

        return reps

class BertPhoneFeaturizer(Featurizer):

    def __init__(self, params_file, gpu=None):
        super().__init__()

        # Load the model
        self._model = BertPhone('bert_12_768_12', prefix='OpenSpeechModel_')
        self._ctx = mx.gpu(gpu) if gpu is not None else mx.cpu()
        self._model.collect_params().load(str(params_file),  ignore_extra=True, ctx=self._ctx)
        self._model.hybridize(static_alloc=True)
        logging.info(self._model)

    def _file_to_feats(self, file):

        assert file.suffix == '.wav'
        # To support CMVN files in the future
        cmvn_spec = None

        def transform(feat):
            '''
            Transform into a sequnce with every 3 frames stacked
            '''
            shape = feat.shape
            out = np.zeros((int(math.ceil(float(shape[0]) / 3)), 3 * shape[1]))
            if feat.shape[0] % 3 == 1:
                feat = np.pad(feat, ((1, 1), (0, 0)), 'edge')
            elif feat.shape[0] % 3 == 2:
                feat = np.pad(feat, ((1, 0), (0, 0)), 'edge')
            # middle one
            out[:, shape[1]: 2 * shape[1]] = feat[1:feat.shape[0] - 1: 3]
            # left context
            out[:, :shape[1]] = feat[0:feat.shape[0] - 2:3, :]
            # right context
            out[:, shape[1] * 2:shape[1] * 3] = feat[2:feat.shape[0]:3, :]
            return out

        def _run_cmd(cmd):
            logging.warn("Running {}".format(cmd))
            try:
                check_call(cmd, shell=True, universal_newlines=True)
            except CalledProcessError as e:
                logging.error("Failed with code {}:".format(e.returncode))
                logging.error(e.output)
                raise e

        with TemporaryDirectory() as temp_dir:

            temp_dir = Path(temp_dir)

            # Create config placeholder
            conf_file = temp_dir / 'mfcc.conf'
            conf_file.write_text('--use-energy=false\n')
            conf_file.write_text('--sample-frequency=8000\n')
            conf_file.write_text('--num-mel-bins=40\n')
            conf_file.write_text('--num-ceps=40\n')
            conf_file.write_text('--low-freq=40\n')
            conf_file.write_text('--high-freq=-200\n')

            # Create SCP placeholder
            input_scp = temp_dir / 'input.scp'
            input_scp.write_text('file-0 {}\n'.format(file))

            # Compute speech features
            feat_ark = temp_dir / "feat.ark"
            feat_scp = temp_dir / "feat.scp"
            cmd = f"compute-mfcc-feats --config={conf_file} scp:{input_scp} ark,scp:{feat_ark},{feat_scp}"
            _run_cmd(cmd)

            cmvn_scp = temp_dir / "cmvn.scp"
            if cmvn_spec is not None:
                # If CMVN specifier is provided, we create a dummy scp
                cmvn_scp.write_text("file-0 {cmvn_spec}\n")
            else:
                # Compute CMVN stats
                cmvn_ark = temp_dir / "cmvn.ark"
                cmd = f"compute-cmvn-stats scp:{feat_scp} ark,scp:{cmvn_ark},{cmvn_scp}"
                _run_cmd(cmd)

            # Apply CMVN
            final_ark = temp_dir / "final.ark"
            final_scp = temp_dir / "final.scp"
            cmd = f"apply-cmvn --norm-vars=true scp:{cmvn_scp} scp:{feat_scp} ark,scp:{final_ark},{final_scp}"
            _run_cmd(cmd)

            with final_scp.open('rb') as fp:
                feats = [features for _, features in kaldi_io.read_mat_scp(fp)][0]

        # Process data
        feats_new = feats
        # Turn the audio into a one-entry batch (TC --> TNC)
        data = mx.nd.expand_dims(mx.nd.array(feats_new, ctx=self._ctx), axis=1)
        # Stack every three frames
        data = transform(data)

        data_len = mx.nd.array([data.shape[0]], ctx=self._ctx)

        vecs = self._model(data, data_len).flatten()
        reps = vecs.asnumpy()

        return reps

