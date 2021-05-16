'''
This is the script and template for different models.
'''
 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from attention import AttentionLayer
from keras import backend as K 


K.clear_session() 
latent_dim=500
 
class ModelBuilding:
    @staticmethod
    def defineModel(latent_dim,max_length_source,src_vocab,trg_vocab):  
        # Encoder 
        encoder_inputs = Input(shape=(max_length_source,)) 
        enc_emb = Embedding(src_vocab, latent_dim,trainable=True)(encoder_inputs)
        #LSTM 1 
        encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True) 
        encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)
        #LSTM 2 
        encoder_lstm2 = LSTM(latent_dim,return_sequences=True,return_state=True) 
        encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)
        #LSTM 3 
        encoder_lstm3=LSTM(latent_dim, return_state=True, return_sequences=True) 
        encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2)
        # Set up the decoder. 
        decoder_inputs = Input(shape=(None,)) 
        dec_emb_layer = Embedding(trg_vocab, latent_dim,trainable=True) 
        dec_emb = dec_emb_layer(decoder_inputs)
        #LSTM using encoder_states as initial state
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True) 
        decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c])
        #Attention Layer
        attn_layer = AttentionLayer(name='attention_layer') 
        attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])
        # Concat attention output and decoder LSTM output 
        decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])
        #Dense layer
        decoder_dense = TimeDistributed(Dense(trg_vocab, activation='softmax')) 
        decoder_outputs = decoder_dense(decoder_concat_input)
        # Define the model
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs) 
        return model

