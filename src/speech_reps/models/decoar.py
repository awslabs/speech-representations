from mxnet import gluon
from mxnet.gluon import nn, rnn
import mxnet.ndarray as nd


class DeCoAR(gluon.Block):

    def __init__(self, feature_size, num_embed, num_hidden,
                 num_layers, dropout,  **kwargs):
        super(DeCoAR, self).__init__(**kwargs)

        with self.name_scope():
            self._label_size = feature_size
            self.encoder = nn.Dense(num_embed, flatten=False)
            self.rnn_forward = rnn.LSTM(num_hidden, num_layers, dropout=dropout,
                           input_size=num_embed, bidirectional=False)
            self.rnn_backward = rnn.LSTM(num_hidden, num_layers, dropout=dropout,
                           input_size=num_embed, bidirectional=False)

    def forward(self, input, data_len):
        hidden_forward, hidden_backward = self.begin_state(func=nd.zeros, batch_size=len(data_len), ctx=input.context)
        x = self.encoder(input)
        x_forward, hidden_forward = self.rnn_forward(x, hidden_forward)
        x_reverse = nd.SequenceReverse(x, sequence_length=data_len, use_sequence_length=True,
                                         axis=0)
        x_backward, hidden_backward = self.rnn_backward(x_reverse, hidden_backward)
        x_backward = nd.SequenceReverse(x_backward,
                          sequence_length=data_len,
                          use_sequence_length=True, axis=0)
        return nd.concat(x_forward, x_backward, dim=2)

    def begin_state(self, *args, **kwargs):
        return self.rnn_forward.begin_state(*args, **kwargs), self.rnn_backward.begin_state(*args, **kwargs)
