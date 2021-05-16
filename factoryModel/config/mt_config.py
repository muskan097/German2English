'''
This is the configuration file for storing all the application parameters
'''
 
import os
from os import path
 
 
#This is the base path to the Machine Translation folder
BASE_PATH = '/Users/ASUS/Desktop/German2English/MTApp'
# Define the path where data is stored
DATA_PATH = path.sep.join([BASE_PATH,'Data/deu.txt'])


# Define the path where the model is saved
MODEL_PATH = path.sep.join([BASE_PATH,'factoryModel/output/model.h5'])
# Define the path to the tokenizer
ENG_TOK_PATH = path.sep.join([BASE_PATH,'factoryModel/output/englishTokenizer.pkl'])
GER_TOK_PATH = path.sep.join([BASE_PATH,'factoryModel/output/germanTokenizer.pkl'])
# Path to Maximum lengths of German and English sentences
GER_MAXLEN = path.sep.join([BASE_PATH,'factoryModel/output/max_length_german.pkl'])
ENG_MAXLEN = path.sep.join([BASE_PATH,'factoryModel/output/max_length_english.pkl'])
# Path to the test sets
TEST_X = path.sep.join([BASE_PATH,'factoryModel/output/X_test.pkl'])
TEST_Y = path.sep.join([BASE_PATH,'factoryModel/output/y_test.pkl'])

######## German Sentence for Translation ###############

GER_SENTENCE = 'heute ist ein guter Tag'