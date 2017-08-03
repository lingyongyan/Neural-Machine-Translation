import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import Attention


class AttentionDecoderRNN(nn.Module):
    """Recurrent neural network that makes use of gated recurrent units to translate encoded inputs using attention."""

    def __init__(self, attention_model, hidden_size, output_size, n_layers=1, dropout_p=.1):
        super(AttentionDecoderRNN, self).__init__()
        self.attention_model = attention_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size * 2, output_size)

        # Choose attention model
        if attention_model is not None:
            self.attention = Attention(attention_model, hidden_size)

    def forward(self, word_input, last_context, last_hidden, encoder_outputs):
        """Run forward propagation one step at a time.
        
        Get the embedding of the current input word (last output word) [s = 1 x batch_size x seq_len]
        then combine them with the previous context. Use this as input and run through the RNN. Next,
        calculate the attention from the current RNN state and all encoder outputs. The final output
        is the next word prediction using the RNN hidden state and context vector.
        
        Args:
            word_input: torch Variable representing the word input constituent
            last_context: torch Variable representing the previous context
            last_hidden: torch Variable representing the previous hidden state output
            encoder_outputs: torch Variable containing the encoder output values
            
        Return:
            output: torch Variable representing the predicted word constituent 
            context: torch Variable representing the context value
            hidden: torch Variable representing the hidden state of the RNN
            attention_weights: torch Variable retrieved from the attention model
        """

        # Run through RNN
        word_embedded = self.embedding(word_input).view(1, 1, -1)
        rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)
        rnn_output, hidden = self.gru(rnn_input, last_hidden)

        # Calculate attention
        attention_weights = self.attention(rnn_output.squeeze(0), encoder_outputs)
        context = attention_weights.bmm(encoder_outputs.transpose(0, 1))

        # Predict output
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)))
        return output, context, hidden, attention_weights
