import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] =':16:8' #This is a command to reduce non-deterministic behavior in CUDA
import warnings
warnings.simplefilter("ignore", UserWarning)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from torchtext.data.metrics import bleu_score    


from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import itertools
import spacy
import numpy as np

import random
import math
import time

import tqdm as tq
import argparse 

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
matplotlib.use("Agg")


import warnings
warnings.simplefilter("ignore", UserWarning)

import en_core_web_sm, de_core_news_sm

import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

parser = argparse.ArgumentParser()
parser.add_argument('--attn', dest='attn', metavar='a', default="sdp")
parser.add_argument('--eval', dest='eval', action='store_true', default=False)
parser.add_argument('--beam', dest='beam', type=int, default=False)   
args = parser.parse_args()

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    


def main():
    
    logging.info('Using device: {}'.format(dev))  


    logging.info("Loading tokenizers and dataset")
    spacy_de = spacy.load('de_core_news_sm')#de_core_news_sm.load() 
    spacy_en = spacy.load('en_core_web_sm')#en_core_web_sm.load()


    SRC = Field(tokenize = lambda text: [tok.text for tok in spacy_de.tokenizer(text)], 
                init_token = '<sos>', eos_token = '<eos>', lower = True)

    TRG = Field(tokenize = lambda text: [tok.text for tok in spacy_en.tokenizer(text)], 
                init_token = '<sos>', eos_token = '<eos>', lower = True)

    train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'), fields = (SRC, TRG))

    SRC.build_vocab(train_data, min_freq = 2)
    TRG.build_vocab(train_data, min_freq = 2)


    BATCH_SIZE = 128

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size = BATCH_SIZE, device = dev,
        sort_key = lambda x : len(x.src))


    # Build model
    logging.info("Building model with attention mechanism: "+args.attn)

    src_vocab_size = len(SRC.vocab)
    dest_vocab_size = len(TRG.vocab)
    word_embed_dim = 256
    hidden_dim = 512
    dropout_rate = 0.5

    if args.attn == "none":
        attn = Dummy(dev=dev)
    elif args.attn == "mean":
        attn = MeanPool()
    elif args.attn == "sdp":
        attn = SingleQueryScaledDotProductAttention(hidden_dim, hidden_dim)

    enc = BidirectionalEncoder(src_vocab_size, word_embed_dim, hidden_dim, hidden_dim, dropout_rate)
    dec = Decoder(dest_vocab_size, word_embed_dim, hidden_dim, hidden_dim, attn, dropout_rate)
    model = Seq2Seq(enc, dec, dev).to(dev)


    criterion = nn.CrossEntropyLoss(ignore_index = TRG.vocab.stoi[TRG.pad_token])
    if not args.eval:
        print("\n")
        logging.info("Training the model")

        # Set up cross-entropy loss but ignore the pad token when computing it
        
        optimizer = optim.Adam(model.parameters())

        best_valid_loss = float('inf')

        for epoch in range(10):
            
            
            train_loss = train(model, train_iterator, optimizer, criterion, epoch+1)
            valid_loss = evaluate(model, valid_iterator, criterion, epoch+1)
            
                        
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), args.attn+'-best-checkpoint.pt')
            
            logging.info(f'Epoch: {epoch+1:02}\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            logging.info(f'Epoch: {epoch+1:02}\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    model.load_state_dict(torch.load(args.attn+'-best-checkpoint.pt'))


    # Test model
    print("\n")
    logging.info("Running test evaluation:")
    test_loss = evaluate(model, test_iterator, criterion,0)
    bleu = calculate_bleu(test_data, SRC, TRG, model, dev)
    logging.info(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} | Test BLEU {bleu*100:.2f}')


    random.seed(42)
    for i in range(10):
        example_id = random.randint(0, len(test_data.examples))
        src = vars(test_data.examples[example_id])['src']
        trg = vars(test_data.examples[example_id])['trg']
        translation, attention = translate_sentence(src, SRC, TRG, model, dev)

        print("\n--------------------")
        print(f'src = {src}')
        print(f'trg = {trg}')
        print(f'prd = {translation}')
        
        save_attention_plot(src, translation, attention, example_id)

    print("\n")
    

##########################################################################################
# Task 2.1
##########################################################################################

class SingleQueryScaledDotProductAttention(nn.Module):

    # kq_dim  is the  dimension  of keys  and  values. Linear  layers  should  be used to  project  inputs to these  dimensions.
    def __init__(self, enc_hid_dim, dec_hid_dim, kq_dim=512):
        super().__init__()

        self.linearQ = nn.Linear(dec_hid_dim, kq_dim)
        self.linearK = nn.Linear(enc_hid_dim * 2, kq_dim)
        self.kq_dim = kq_dim
        self.softmax = nn.Softmax(-1)


    # hidden  is h_t^{d} from Eq. (11)  and has dim => [batch_size , dec_hid_dim]
    # encoder_outputs  is the  word  representations  from Eq. (6) and has dim => [src_len , batch_size , enc_hid_dim * 2]
    def forward(self, hidden, encoder_outputs):

        q = self.linearQ(hidden)
        k_t = self.linearK(encoder_outputs)
        v_t = encoder_outputs

        # add a extra dimension on query so that we can use bmm function
        q = q.unsqueeze(1)
        k_t = k_t.permute(1, 2, 0)

        alpha = self.softmax(torch.bmm(q, k_t / math.sqrt(self.kq_dim)))
        attended_val = torch.bmm(alpha, v_t.permute(1, 0, 2))

        alpha = alpha.squeeze(1)
        attended_val = attended_val.squeeze(1)

        assert attended_val.shape == (hidden.shape[0], encoder_outputs.shape[2])
        assert alpha.shape == (hidden.shape[0], encoder_outputs.shape[0])

        return attended_val, alpha

##########################################################################################
# Model Definitions
##########################################################################################

class Dummy(nn.Module):    
    
    def __init__(self, dev):
        super().__init__()
        self.dev = dev

    def forward(self, hidden, encoder_outputs):
        zout = torch.zeros( (hidden.shape[0], encoder_outputs.shape[2]) ).to(self.dev)
        zatt = torch.zeros( (hidden.shape[0], encoder_outputs.shape[0]) ).to(self.dev)
        return zout, zatt

class MeanPool(nn.Module):    
    
    def __init__(self):
        super().__init__()
        
    def forward(self, hidden, encoder_outputs):
        
        output = torch.mean(encoder_outputs, dim=0, keepdim=True).squeeze(0)
        alpha = F.softmax(torch.ones(hidden.shape[0], encoder_outputs.shape[0]), dim=0)

        return output, alpha

class BidirectionalEncoder(nn.Module):
    def __init__(self, src_vocab, emb_dim, enc_hid_dim, dec_hid_dim, dropout=0.5):
        super().__init__()
        
        self.enc_hidden_dim = enc_hid_dim
        self.emb = nn.Embedding(src_vocab, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):

        # embed source tokens
        embedded = self.dropout(self.emb(src))

        # process with bidirectional GRU model
        enc_hidden_states, _ = self.rnn(embedded)

        # compute a global sentence representation to feed as the initial hidden state of the decoder
        # concatenate the forward GRU's representation after the last word and
        # the backward GRU's representation after the first word

        last_forward = enc_hidden_states[-1, :, :self.enc_hidden_dim]
        first_backward = enc_hidden_states[0, :, self.enc_hidden_dim:]

        # transform to the size of the decoder hidden state with a fully-connected layer
        sent = F.relu(self.fc(torch.cat((last_forward, first_backward), dim = 1)))

        return enc_hidden_states, sent


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, attention, dropout=0.5,):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.GRU(emb_dim, dec_hid_dim)
        
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        #Embed input
        input = input.unsqueeze(0)        
        embedded = self.dropout(self.embedding(input))
        
        #Step decoder model forward
        output, hidden = self.rnn(embedded, hidden.unsqueeze(0))
        
        #Perform attention operation
        attended_feature, a = self.attention(hidden.squeeze(0), encoder_outputs) 
        
        #Make prediction
        prediction = self.fc_out(torch.cat((output, attended_feature.unsqueeze(0)), dim = 2))
        
        #Output prediction (scores for each word), the updated hidden state, and the attention map (for visualization)        
        return prediction, hidden.squeeze(0), a

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg):
        
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)
                
        
        for t in range(1, trg_len):
           
            # Step decoder model forward, getting output prediction, updated hidden, and attention distribution 
            output, hidden, a = self.decoder(trg[t-1], hidden, encoder_outputs)
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output

        
        return outputs


##########################################################################################
# Train / Eval Functions
##########################################################################################

def train(model, iterator, optimizer, criterion, epoch):
    
    model.train()
    
    epoch_loss = 0
    pbar = tq.tqdm(desc="Epoch {}".format(epoch), total=len(iterator), unit="batch")
        
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
         
        optimizer.zero_grad()
        
        output = model(src, trg)
                
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        
        loss = criterion(output, trg)
        
        loss.backward() 
        
        optimizer.step()
        
        epoch_loss += loss.item()
        pbar.update(1)

    pbar.close()        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, epoch):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg) 

            output_dim = output.shape[-1]
            
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


##########################################################################################
# Utility Functions
##########################################################################################
def beamsearch(model, trg_field, encoder_outputs, hidden, trg_indexes, attentions, beam, max_length, device, src_len):
    logsoftmax = nn.LogSoftmax(-1)
    trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
    
    output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs)
    # copy the attention to the first word
    attentions[0] = attention.squeeze(0)
    
    output = logsoftmax(output.squeeze())
    
    top_value, top_idx = torch.topk(output, beam)

    new_h_t = torch.cat(beam*[hidden], 0)
    new_encoder_outputs = torch.cat(beam*[encoder_outputs], 1)
    
    last_top_idx = top_idx
    last_top_value = top_value
      
    decodedList = []
    for i in last_top_idx.view(beam, 1):
        decodedList.append([i])

    for i in range (1, src_len):

        new_pred, new_h_t, attention = model.decoder(last_top_idx, new_h_t, new_encoder_outputs)
   
        # tranform the output
        new_pred = logsoftmax(new_pred.squeeze())
        
        # select topK based on cumulative loglikelihood 
        cumulative_log = new_pred + torch.reshape(last_top_value, (beam, 1))
        
        # create a index table for finding topk
        new_top_value, new_top_idx = torch.topk(cumulative_log.flatten(), beam)
        idx_table = np.array(np.unravel_index(new_top_idx.cpu().numpy(), cumulative_log.shape)).T
        
        # derive the value of new topK index 
        derived_new_top_value = []
        derived_new_top_idx = []
        
        # for updating
        temp_decodedList = []
        
        for idx in idx_table:
            derived_new_top_value.append(torch.unsqueeze(cumulative_log[idx[0]][idx[1]], 0))
            derived_new_top_idx.append(torch.unsqueeze(torch.tensor(idx[1]), 0))
            # update attention
            attentions[i][idx[0]] = attention[idx[0]]
            
            temp_decodedList.append(decodedList[idx[0]] + [(torch.unsqueeze(torch.tensor(idx[1]), 0)).to(dev)])
        
        decodedList = temp_decodedList
        derived_new_top_value = torch.cat(derived_new_top_value)
        derived_new_top_idx = torch.cat(derived_new_top_idx)
        
        temp_h_t = []
        for idx in idx_table:
            temp_h_t.append(new_h_t[idx[0]])
        
        temp_h_t = torch.stack(temp_h_t)
        
        last_top_idx = derived_new_top_idx.to(dev)
        last_top_value = derived_new_top_value.to(dev)
        new_h_t = temp_h_t.to(dev)
        
    #final_result = torch.cat(decodedList[derived_new_top_value.argmax()])
    final_result = decodedList[derived_new_top_value.argmax()]

    return final_result, derived_new_top_value.argmax()

def translate_sentence(sentence, src_field, trg_field, model, device, max_len = 50):
    model.eval()
    
    if isinstance(sentence, str):
        nlp = spacy.load('de')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]
        
    
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    
    src_len = torch.LongTensor([len(src_indexes)])
    
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)

        
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    
    if args.beam == 5:
        beam = 5
    elif args.beam == 10:
        beam = 10
    elif args.beam == 20:
        beam = 20
    elif args.beam == 50:
        beam = 50
        
    if args.beam:
        attentions = torch.zeros(max_len, beam, len(src_indexes)).to(device)
        
        outputs, max_idx = beamsearch(model, trg_field, encoder_outputs, hidden, trg_indexes,
                                    attentions, beam, max_len, device, src_len)
    
        # get rid of the part after eos
        attentions[:, -1, :] = attentions[:, max_idx, :]
        
        #final_output = list(itertools.takewhile(lambda x: x != trg_field.vocab.stoi[trg_field.eos_token], output))
        #final_output.append(trg_field.vocab.stoi[trg_field.eos_token])
        
        for trg_i in outputs:
            trg_indexes.append(trg_i)
            if trg_i == trg_field.vocab.stoi[trg_field.eos_token]:
                break
            
    else:
        attentions = torch.zeros(max_len, 1, len(src_indexes)).to(device)
        
        for i in range(max_len):
            trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
            
            with torch.no_grad():
                output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs)
            
            attentions[i] = attention.squeeze()
            
            pred_token = output.squeeze().argmax().item()
            
            trg_indexes.append(pred_token)
            
            if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
                break
        
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    
    return trg_tokens[1:], attentions[:len(trg_tokens)-1]
 
def save_attention_plot(sentence, translation, attention, index):
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    
    attention = attention[:, -1, :].squeeze(1).cpu().detach().numpy()
    
    cax = ax.matshow(attention, cmap='Greys_r', vmin=0, vmax=1)
    fig.colorbar(cax)
   
    ax.tick_params(labelsize=15)
    
    x_ticks = [''] + ['<sos>'] + [t.lower() for t in sentence] + ['<eos>']
    y_ticks = [''] + translation
     
    ax.set_xticklabels(x_ticks, rotation=45)
    ax.set_yticklabels(y_ticks)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig("examples/"+str(index)+'_translation.png')
    


def calculate_bleu(data, src_field, trg_field, model, device, max_len = 50):
        
        trgs = []
        pred_trgs = []
        
        for datum in data:
            
            src = vars(datum)['src']
            trg = vars(datum)['trg']
            
            pred_trg, _ = translate_sentence(src, src_field, trg_field, model, device, max_len)
            
            #cut off <eos> token
            pred_trg = pred_trg[:-1]
            
            pred_trgs.append(pred_trg)
            trgs.append([trg])
            
        return bleu_score(pred_trgs, trgs)



if __name__ == "__main__":

    main()
