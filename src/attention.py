import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Attention(nn.Module):
    """Attention nn module that is responsible for computing the alignment scores."""

    def __init__(self, method, hidden_size):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        # Define layers
        if self.method == 'general':
            self.attention = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attention = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, self.hidden_size))

    def forward(self, hidden, encoder_outputs):
        """Attend all encoder inputs conditioned on the previous hidden state of the decoder.
        
        After creating variables to store the attention energies, calculate their 
        values for each encoder output and return the normalized values.
        
        Args:
            hidden: decoder hidden output used for condition
            encoder_outputs: list of encoder outputs
            
        Returns:
             Normalized (0..1) energy values, re-sized to 1 x 1 x seq_len
        """

        seq_len = len(encoder_outputs)
        energies = Variable(torch.zeros(seq_len)).cuda()
        for i in range(seq_len):
            energies[i] = self._score(hidden, encoder_outputs[i])
        return F.softmax(energies).unsqueeze(0).unsqueeze(0)

    def _score(self, hidden, encoder_output):
        """Calculate the relevance of a particular encoder output in respect to the decoder hidden."""

        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
        elif self.method == 'general':
            energy = self.attention(encoder_output)
            energy = hidden.dot(energy)
        elif self.method == 'concat':
            energy = self.attention(torch.cat((hidden, encoder_output), 1))
            energy = self.other.dor(energy)
        return energy
