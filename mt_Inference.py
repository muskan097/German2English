'''
This is the driver file for the inference process
'''

from tensorflow.keras.models import load_model
from factoryModel.config import mt_config as confFile
from factoryModel.utils.helperFunctions import *
 
## Define the file path to the model
modelPath = confFile.MODEL_PATH
 
# Load the model from the file path
model = load_model(modelPath)


# Get the paths for all the files and variables stored as pickle files
Eng_tokPath = confFile.ENG_TOK_PATH
Ger_tokPath = confFile.GER_TOK_PATH
testxPath = confFile.TEST_X
testyPath = confFile.TEST_Y
Ger_length = confFile.GER_MAXLEN
# Load the tokenizer from the pickle file
englishTokenizer = load_files(Eng_tokPath)
germanTokenizer = load_files(Ger_tokPath)
# Load the standard lengths
max_length_german = load_files(Ger_length)
# Load the test sets
X_test = load_files(testxPath)
y_test = load_files(testyPath)


# Generate predictions
predSent = generatePredictions(testX[20:30,:])



# Get the input sentence from the config file
inputSentence = [confFile.GER_SENTENCE]

# Clean the input sentence
cleanText = cleanInput(inputSentence)

# Encode the inputsentence as sequence of integers
seq1 = encode_sequences(germanTokenizer,int(max_length_german),cleanText)




print("[INFO] .... Predicting on own sentences...")

# Generate the prediction
predSent = generatePredictions(CleanText)
print("Original sentence : {} :: Prediction : {}".format([cleanText[0]],predSent))