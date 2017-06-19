import argparse
import etl
import helpers
import os
import torch
from attention_decoder import AttentionDecoderRNN
from encoder_rnn import EncoderRNN
from language import Language
from torch.autograd import Variable


# Parse argument for input sentence
parser = argparse.ArgumentParser()
parser.add_argument('input')
args = parser.parse_args()

input_lang, output_lang, pairs = etl.prepare_data('spa')
attn_model = 'general'
hidden_size = 500
n_layers = 2
dropout_p = 0.05

# Initialize models
encoder = EncoderRNN(input_lang.n_words, hidden_size, n_layers)
decoder = AttentionDecoderRNN(attn_model, hidden_size, output_lang.n_words, n_layers, dropout_p=dropout_p)

is_missing_params = (not os.path.exists('data/attention_params')
                     or not os.path.exists('data/decoder_params')
                     or not os.path.exists('data/encoder_params'))
if is_missing_params:
    print('Model parameters were not found. Please train a new seq2seq model.')
    exit(1)
else:
    encoder.load_state_dict(torch.load('data/encoder_params'))
    decoder.load_state_dict(torch.load('data/decoder_params'))
    decoder.attention.load_state_dict(torch.load('data/attention_params'))

# Move models to GPU
encoder.cuda()
decoder.cuda()


def evaluate(sentence, max_length=10):
    input_variable = etl.variable_from_sentence(input_lang, sentence)
    input_length = input_variable.size()[0]

    # Run through encoder
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([[Language.sos_token]]))  # SOS
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    decoder_input = decoder_input.cuda()
    decoder_context = decoder_context.cuda()

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context,
                                                                                     decoder_hidden, encoder_outputs)
        decoder_attentions[di, :decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == Language.eos_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda()

    return decoded_words, decoder_attentions[:di + 1, :len(encoder_outputs)]

sentence = helpers.normalize_string(args.input)
output_words, decoder_attn = evaluate(sentence)
output_sentence = ' '.join(output_words)
print(output_sentence)
