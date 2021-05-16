'''
This is the script for flask application
'''
from attention import AttentionLayer
from tensorflow.keras.models import load_model
from factoryModel.config import mt_config as confFile
from factoryModel.utils.helperFunctions import *
from flask import Flask,request,render_template
 
# Initializing the flask application
app = Flask(__name__)
 
## Define the file path to the model
modelPath = confFile.MODEL_PATH
 
# Load the model from the file path
model = load_model(modelPath,custom_objects={'AttentionLayer':AttentionLayer})
# Get the paths for all the files and variables stored as pickle files
Eng_tokPath = confFile.ENG_TOK_PATH
Ger_tokPath = confFile.GER_TOK_PATH
testxPath = confFile.TEST_X
testyPath = confFile.TEST_Y
Ger_length = confFile.GER_MAXLEN
Eng_length = confFile.ENG_MAXLEN
# Load the tokenizer from the pickle file
englishTokenizer = load_files(Eng_tokPath)
germanTokenizer = load_files(Ger_tokPath)
# Load the standard lengths
max_length_german = load_files(Ger_length)
# Load the standard lengths
max_length_english = load_files(Eng_length)
# Load the test sets
X_test = load_files(testxPath)
y_test = load_files(testyPath)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/translate', methods=['POST', 'GET'])
def get_translation():
    if request.method == 'POST':
 
        result = request.form
        # Get the German sentence from the Input site
        gerSentence = str(result['input_text'])
        # Converting the text into the required format for prediction
        # Step 1 : Converting to an array
        gerAr = [gerSentence]
        # Clean the input sentence
        cleanText = cleanInput(gerAr)
        # Step 2 : Converting to sequences and padding them
        # Encode the inputsentence as sequence of integers
        seq1 = encode_sequences(germanTokenizer, int(max_length_german), cleanText)
        # Step 3 : Get the translation
        translation = generatePredictions(seq1,max_length_german)
        # prediction = model.predict(seq1,verbose=0)[0]
 
        return render_template('result.html', trans=translation)


if __name__ == '__main__':
    app.debug = True
    app.run()