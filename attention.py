import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Attention(nn.Module):
    def __init__(self, method, hidden_size, max_length=10):
        super(Attention, self).__init__()

        # Keep parameters
        self.method = method
        self.hidden_size = hidden_size

        # Define layers
        if self.method == 'general':
            self.attention = nn.Linear(self.hidden_size, self.hidden_size)

        elif self.method == 'concat':
            self.attention = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, self.hidden_size))

    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)

        # Create variables to store the attention energies
        attention_energies = Variable(torch.zeros(seq_len))
        attention_energies = attention_energies.cuda()

        # Calculate energies for each encoder output
        for i in range(seq_len):
            attention_energies[i] = self.score(hidden, encoder_outputs[i])

        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return F.softmax(attention_energies).unsqueeze(0).unsqueeze(0)

    def score(self, hidden, encoder_output):

        if self.method == 'dot':
            energy = hidden.dot(encoder_output)

        elif self.method == 'general':
            energy = self.attention(encoder_output)
            energy = hidden.dot(energy)

        elif self.method == 'concat':
            energy = self.attention(torch.cat((hidden, encoder_output), 1))
            energy = self.other.dor(energy)

        return energy
