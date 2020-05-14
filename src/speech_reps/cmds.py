import argparse
import logging
import os
from pathlib import Path

def main():
    """Defines arguments for all subcommands"""

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')

    # Discover Kaldi
    if not "KALDI_ROOT" in os.environ:
        kaldi_path = (Path(os.path.realpath(__file__)).parent.parent.parent / 'kaldi').resolve()
        os.environ["KALDI_ROOT"] = str(kaldi_path)
        logging.warn("$KALDI_ROOT not set, will set KALDI_ROOT='{}'".format(kaldi_path))
        if not kaldi_path.is_dir():
            raise NotADirectoryError("Either set $KALDI_ROOT or install to {}".format(kaldi_path))
    # This will set $PATH
    import kaldi_io

    # Then, import submodules
    from .featurize import Featurizer

    parser = argparse.ArgumentParser(description="Speech representations")
    subparsers = parser.add_subparsers(help="Run 'speech-reps {subcommand} -h' for details")

    # featurize
    parser_featurize = subparsers.add_parser('featurize', help='Converts audio into features')
    Featurizer.populate_parser(parser_featurize)

    args = parser.parse_args()

    # Run command-specific functions
    args.func(args)
