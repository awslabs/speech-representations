# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from mxnet.gluon import Block
from mxnet.gluon import nn
from gluonnlp.model.transformer import BasePositionwiseFFN, BaseTransformerEncoderCell, BaseTransformerEncoder

###############################################################################
#                              COMPONENTS                                     #
###############################################################################

bert_12_768_12_hparams = {
    'attention_cell': 'multi_head',
    'num_layers': 12,
    'units': 768,
    'hidden_size': 3072,
    'max_length': 512,
    'num_heads': 12,
    'scaled': True,
    'dropout': 0.1,
    'use_residual': True,
    'embed_size': 768,
    'embed_dropout': 0,
    'token_type_vocab_size': 2,
    'word_embed': None,
}

bert_hparams = {
    'bert_12_768_12': bert_12_768_12_hparams
}


class BERTLayerNorm(nn.LayerNorm):
    """BERT style Layer Normalization.

    Epsilon is added inside the square root and set to 1e-12 by default.

    Inputs:
        - **data**: input tensor with arbitrary shape.
        - **out**: output tensor with the same shape as `data`.
    """

    def __init__(self, epsilon=1e-12, in_channels=0, prefix=None, params=None):
        super(BERTLayerNorm, self).__init__(epsilon=epsilon, in_channels=in_channels,
                                            prefix=prefix, params=params)

    def hybrid_forward(self, F, data, gamma, beta):
        """forward computation."""
        return F.LayerNorm(data, gamma=gamma, beta=beta, axis=self._axis, eps=self._epsilon)


class BERTPositionwiseFFN(BasePositionwiseFFN):
    """Structure of the Positionwise Feed-Forward Neural Network for
    BERT.

    Different from the original positionwise feed forward network
    for transformer, `BERTPositionwiseFFN` uses `GELU` for activation
    and `BERTLayerNorm` for layer normalization.

    Parameters
    ----------
    units : int
        Number of units for the output
    hidden_size : int
        Number of units in the hidden layer of position-wise feed-forward networks
    dropout : float
        Dropout probability for the output
    use_residual : bool
        Add residual connection between the input and the output
    weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    bias_initializer : str or Initializer
        Initializer for the bias vector.
    prefix : str, default None
        Prefix for name of `Block`s (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells. Created if `None`.
    activation : str, default 'gelu'
        Activation methods in PositionwiseFFN
    layer_norm_eps : float, default None
        Epsilon for layer_norm

    Inputs:
        - **inputs** : input sequence of shape (batch_size, length, C_in).

    Outputs:
        - **outputs** : output encoding of shape (batch_size, length, C_out).
    """

    def __init__(self, units=512, hidden_size=2048, dropout=0.0, use_residual=True,
                 weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None, activation='gelu', layer_norm_eps=None):
        super(BERTPositionwiseFFN, self).__init__(units=units, hidden_size=hidden_size,
                                                  dropout=dropout, use_residual=use_residual,
                                                  weight_initializer=weight_initializer,
                                                  bias_initializer=bias_initializer,
                                                  prefix=prefix, params=params,
                                                  # extra configurations for BERT
                                                  activation=activation,
                                                  use_bert_layer_norm=True,
                                                  layer_norm_eps=layer_norm_eps)


class BERTEncoder(BaseTransformerEncoder):
    """Structure of the BERT Encoder.

    Different from the original encoder for transformer,
    `BERTEncoder` uses learnable positional embedding, `BERTPositionwiseFFN`
    and `BERTLayerNorm`.

    Parameters
    ----------
    attention_cell : AttentionCell or str, default 'multi_head'
        Arguments of the attention cell.
        Can be 'multi_head', 'scaled_luong', 'scaled_dot', 'dot', 'cosine', 'normed_mlp', 'mlp'
    num_layers : int
        Number of attention layers.
    units : int
        Number of units for the output.
    hidden_size : int
        number of units in the hidden layer of position-wise feed-forward networks
    max_length : int
        Maximum length of the input sequence
    num_heads : int
        Number of heads in multi-head attention
    scaled : bool
        Whether to scale the softmax input by the sqrt of the input dimension
        in multi-head attention
    dropout : float
        Dropout probability of the attention probabilities.
    use_residual : bool
    output_attention: bool, default False
        Whether to output the attention weights
    output_all_encodings: bool, default False
        Whether to output encodings of all encoder cells
    weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    bias_initializer : str or Initializer
        Initializer for the bias vector.
    prefix : str, default None.
        Prefix for name of `Block`s. (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells. Created if `None`.
    activation : str, default 'gelu'
        Activation methods in PositionwiseFFN
    layer_norm_eps : float, default None
        Epsilon for layer_norm

    Inputs:
        - **inputs** : input sequence of shape (batch_size, length, C_in)
        - **states** : list of tensors for initial states and masks.
        - **valid_length** : valid lengths of each sequence. Usually used when part of sequence
            has been padded. Shape is (batch_size, )

    Outputs:
        - **outputs** : the output of the encoder. Shape is (batch_size, length, C_out)
        - **additional_outputs** : list of tensors.
            Either be an empty list or contains the attention weights in this step.
            The attention weights will have shape (batch_size, num_heads, length, mem_length)
    """

    def __init__(self, attention_cell='multi_head', num_layers=2,
                 units=512, hidden_size=2048, max_length=50,
                 num_heads=4, scaled=True, dropout=0.0,
                 use_residual=True, output_attention=False, output_all_encodings=False,
                 weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None, activation='gelu', layer_norm_eps=None):
        super(BERTEncoder, self).__init__(attention_cell=attention_cell,
                                          num_layers=num_layers, units=units,
                                          hidden_size=hidden_size, max_length=max_length,
                                          num_heads=num_heads, scaled=scaled, dropout=dropout,
                                          use_residual=use_residual,
                                          output_attention=output_attention,
                                          output_all_encodings=output_all_encodings,
                                          weight_initializer=weight_initializer,
                                          bias_initializer=bias_initializer,
                                          prefix=prefix, params=params,
                                          # extra configurations for BERT
                                          positional_weight='learned',
                                          use_bert_encoder=True,
                                          use_layer_norm_before_dropout=False,
                                          scale_embed=False,
                                          activation=activation,
                                          layer_norm_eps=layer_norm_eps)


class BERTEncoderCell(BaseTransformerEncoderCell):
    """Structure of the Transformer Encoder Cell for BERT.

    Different from the original encoder cell for transformer,
    `BERTEncoderCell` adds bias terms for attention and the projection
    on attention output. It also uses `BERTPositionwiseFFN` and
    `BERTLayerNorm`.

    Parameters
    ----------
    attention_cell : AttentionCell or str, default 'multi_head'
        Arguments of the attention cell.
        Can be 'multi_head', 'scaled_luong', 'scaled_dot', 'dot', 'cosine', 'normed_mlp', 'mlp'
    units : int
        Number of units for the output
    hidden_size : int
        number of units in the hidden layer of position-wise feed-forward networks
    num_heads : int
        Number of heads in multi-head attention
    scaled : bool
        Whether to scale the softmax input by the sqrt of the input dimension
        in multi-head attention
    dropout : float
    use_residual : bool
    output_attention: bool
        Whether to output the attention weights
    weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    bias_initializer : str or Initializer
        Initializer for the bias vector.
    prefix : str, default None
        Prefix for name of `Block`s. (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells. Created if `None`.
    activation : str, default 'gelu'
        Activation methods in PositionwiseFFN
    layer_norm_eps : float, default None
        Epsilon for layer_norm

    Inputs:
        - **inputs** : input sequence. Shape (batch_size, length, C_in)
        - **mask** : mask for inputs. Shape (batch_size, length, length)

    Outputs:
        - **outputs**: output tensor of the transformer encoder cell.
            Shape (batch_size, length, C_out)
        - **additional_outputs**: the additional output of all the transformer encoder cell.
    """

    def __init__(self, attention_cell='multi_head', units=128,
                 hidden_size=512, num_heads=4, scaled=True,
                 dropout=0.0, use_residual=True, output_attention=False,
                 weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None, activation='gelu', layer_norm_eps=None):
        super(BERTEncoderCell, self).__init__(attention_cell=attention_cell,
                                              units=units, hidden_size=hidden_size,
                                              num_heads=num_heads, scaled=scaled,
                                              dropout=dropout, use_residual=use_residual,
                                              output_attention=output_attention,
                                              weight_initializer=weight_initializer,
                                              bias_initializer=bias_initializer,
                                              prefix=prefix, params=params,
                                              # extra configurations for BERT
                                              attention_use_bias=True,
                                              attention_proj_use_bias=True,
                                              use_bert_layer_norm=True,
                                              use_bert_ffn=True,
                                              activation=activation,
                                              layer_norm_eps=layer_norm_eps)


###############################################################################
#                                FULL MODEL                                   #
###############################################################################

class BertPhone(Block):
    """Generic Model for BERT (Bidirectional Encoder Representations from Transformers).

    Parameters
    ----------
    encoder : BERTEncoder
        Bidirectional encoder that encodes the input sentence.
    vocab_size : int or None, default None
        The size of the vocabulary.
    token_type_vocab_size : int or None, default None
        The vocabulary size of token types.
    units : int or None, default None
        Number of units for the final pooler layer.
    embed_size : int or None, default None
        Size of the embedding vectors. It is used to generate the word and token type
        embeddings if word_embed and token_type_embed are None.
    embed_dropout : float, default 0.0
        Dropout rate of the embedding weights. It is used to generate the source and target
        embeddings if word_embed and token_type_embed are None.
    embed_initializer : Initializer, default None
        Initializer of the embedding weights. It is used to generate the source and target
        embeddings if word_embed and token_type_embed are None.
    word_embed : Block or None, default None
        The word embedding. If set to None, word_embed will be constructed using embed_size and
        embed_dropout.
    token_type_embed : Block or None, default None
        The token type embedding. If set to None and the token_type_embed will be constructed using
        embed_size and embed_dropout.
    use_pooler : bool, default True
        Whether to include the pooler which converts the encoded sequence tensor of shape
        (batch_size, seq_length, units) to a tensor of shape (batch_size, units)
        for segment level classification task.
    use_decoder : bool, default True
        Whether to include the decoder for masked language model prediction.
    use_classifier : bool, default True
        Whether to include the classifier for next sentence classification.
    prefix : str or None
        See document of `mx.gluon.Block`.
    params : ParameterDict or None
        See document of `mx.gluon.Block`.

    Inputs:
        - **inputs**: input sequence tensor, shape (batch_size, seq_length)
        - **token_types**: input token type tensor, shape (batch_size, seq_length).
            If the inputs contain two sequences, then the token type of the first
            sequence differs from that of the second one.
        - **valid_length**: optional tensor of input sequence valid lengths, shape (batch_size,)
        - **masked_positions**: optional tensor of position of tokens for masked LM decoding,
            shape (batch_size, num_masked_positions).

    Outputs:
        - **sequence_outputs**: Encoded sequence, which can be either a tensor of the last
            layer of the Encoder, or a list of all sequence encodings of all layers.
            In both cases shape of the tensor(s) is/are (batch_size, seq_length, units).
        - **attention_outputs**: output list of all intermediate encodings per layer
            Returned only if BERTEncoder.output_attention is True.
            List of num_layers length of tensors of shape
            (num_masks, num_attention_heads, seq_length, seq_length)
        - **pooled_output**: output tensor of pooled representation of the first tokens.
            Returned only if use_pooler is True. Shape (batch_size, units)
        - **next_sentence_classifier_output**: output tensor of next sentence classification.
            Returned only if use_classifier is True. Shape (batch_size, 2)
        - **masked_lm_outputs**: output tensor of sequence decoding for masked language model
            prediction. Returned only if use_decoder True.
            Shape (batch_size, num_masked_positions, vocab_size)
    """

    def __init__(self, model_name, prefix=None, params=None, **kwargs):
        super(BertPhone, self).__init__(prefix=prefix, params=params)
        predefined_args = bert_hparams[model_name]
        mutable_args = ['use_residual', 'dropout', 'embed_dropout', 'word_embed']
        mutable_args = frozenset(mutable_args)
        assert all((k not in kwargs or k in mutable_args) for k in predefined_args), \
            'Cannot override predefined model settings.'
        predefined_args.update(kwargs)
        # encoder
        self.encoder = BERTEncoder(attention_cell=predefined_args['attention_cell'],
                              num_layers=predefined_args['num_layers'],
                              units=predefined_args['units'],
                              hidden_size=predefined_args['hidden_size'],
                              max_length=predefined_args['max_length'],
                              num_heads=predefined_args['num_heads'],
                              scaled=predefined_args['scaled'],
                              dropout=predefined_args['dropout'],
                              output_attention=False,
                              output_all_encodings=False,
                              use_residual=predefined_args['use_residual'])

        self.embed = nn.Dense(predefined_args['embed_size'], flatten=False)


    def forward(self, inputs, valid_length=None):  # pylint: disable=arguments-differ
        """Generate the representation given the inputs.

        This is used in training or fine-tuning a BERT model.
        """
        x = self.embed(inputs)
        x = x.transpose(axes=(1, 0, 2))
        x, additional_outputs = self.encoder(x, states=None, valid_length=valid_length)
        x = x.transpose(axes=(1, 0, 2))

        return x

