'''
This script lists down all the helper functions which are required for processing raw data
'''
from attention import AttentionLayer
from tensorflow.keras.models import Model
from pickle import load
from tensorflow import keras
from numpy import argmax
from pickle import dump
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Input, Dense,Embedding, Concatenate, TimeDistributed
from numpy import array
from unicodedata import normalize
import string
import numpy as np

# Function to Save data to pickle form
def save_clean_data(data,filename):
    dump(data,open(filename,'wb'))
    print('Saved: %s' % filename)

# Function to load pickle data from disk
def load_files(filename):
    return load(open(filename,'rb'))

model = keras.models.load_model("/Users/ASUS/Desktop/German2English/MTApp/factoryModel/output/model.h5",custom_objects={'AttentionLayer':AttentionLayer})

max_length_german = load_files("/Users/ASUS/Desktop/German2English/MTApp/factoryModel/output/max_length_german.pkl")
max_length_english = load_files("/Users/ASUS/Desktop/German2English/MTApp/factoryModel/output/max_length_english.pkl")
Eword2index = load_files("/Users/ASUS/Desktop/German2English/MTApp/factoryModel/output/Eword2index.pkl")
Gword2index = load_files("/Users/ASUS/Desktop/German2English/MTApp/factoryModel/output/Gword2index.pkl")
Eindex2word = load_files("/Users/ASUS/Desktop/German2English/MTApp/factoryModel/output/Eindex2word.pkl")
Gindex2word = load_files("/Users/ASUS/Desktop/German2English/MTApp/factoryModel/output/Gindex2word.pkl")



# Function to clean the input data
def cleanInput(lines):
    cleanSent = []
    cleanDocs = list()
    for docs in lines[0].split():
        line = normalize('NFD', docs).encode('ascii', 'ignore')
        line = line.decode('UTF-8')
        line = [line.translate(str.maketrans('', '', string.punctuation))]
        line = line[0].lower()
        cleanDocs.append(line)
    cleanSent.append(' '.join(cleanDocs))
    return array(cleanSent)

# Function to convert sentences to sequences of integers
def encode_sequences(tokenizer,length,lines):
    # Sequences as integers
    X = tokenizer.texts_to_sequences(lines)
    # Padding the sentences with 0
    X = pad_sequences(X,maxlen=length,padding='post')
    return X


latent_dim=500
def encode_inference(model1,max_length_source):
    encoder_inputs = model1.input[0]  #loading encoder_inputs
    encoder_outputs, state_h, state_c = model1.layers[6].output #loading encoder_outputs
    encoder_states= [encoder_outputs,state_h,state_c]
    #print(encoder_outputs.shape)
    encoder_model = Model(inputs=encoder_inputs,outputs=encoder_states)
    # decoder inference
    # Below tensors will hold the states of the previous time step
    decoder_state_input_h = Input(shape=(500,),name='input_3')
    decoder_state_input_c = Input(shape=(500,),name='input_4')
    decoder_hidden_state_input = Input(shape=(max_length_source,500))
    # Get the embeddings of the decoder sequence
    decoder_inputs = model1.layers[3].output
    #print(decoder_inputs.shape)
    dec_emb_layer = model1.layers[5]
    dec_emb2= dec_emb_layer(decoder_inputs)
    # To predict the next word in the sequence, set the initial states to the states from the previous time step
    decoder_lstm = model1.layers[7]
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])
    #attention inference
    attn_layer = model1.layers[8]
    attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
    concate = model.layers[9]
    decoder_inf_concat = concate([decoder_outputs2, attn_out_inf])
    # A dense softmax layer to generate prob dist. over the target vocabulary
    decoder_dense = model1.layers[10]
    decoder_outputs2 = decoder_dense(decoder_inf_concat)
    # Final decoder model
    decoder_model = Model(
    [decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
    [decoder_outputs2] + [state_h2, state_c2])
    return encoder_model, decoder_model



def decode_sequence(input_seq,enc_model,dec_model,max_length,word2index,index2word):
    # Encode the input as state vectors.
    e_out, e_h, e_c = enc_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))

    # Chose the 'start' word as the first word of the target sequence
    target_seq[0, 0] = word2index['start']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = dec_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        if sampled_token_index == 0:
            break
        else:
            sampled_token = index2word[sampled_token_index]

            if(sampled_token!='end'):
                decoded_sentence += ' '+sampled_token

              # Exit condition: either hit max length or find stop word.
                if (sampled_token == 'end' or len(decoded_sentence.split()) >= (max_length-1)):
                    stop_condition = True

          # Update the target sequence (of length 1).
            target_seq = np.zeros((1,1))
            target_seq[0, 0] = sampled_token_index

          # Update internal states
            e_h, e_c = h, c
            
         

    return decoded_sentence



def generatePredictions(data,max_length_source):
    AllPreds=[]
    encoder_model,decoder_model = encode_inference(model,max_length_german)
    for i in range(10):
        target = decode_sequence(data[i].reshape(1,max_length_source),encoder_model,decoder_model,max_length_english,Eword2index,Eindex2word)
        AllPreds.append(target)
        return AllPreds