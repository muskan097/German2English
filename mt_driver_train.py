'''
This is the driver file which controls the complete training process
'''
 
from factoryModel.config import mt_config as confFile
from factoryModel.preprocessing import SentenceSplit,cleanData,TrainMaker
from factoryModel.dataLoader import textLoader
from factoryModel.models import ModelBuilding
from tensorflow.keras.callbacks import ModelCheckpoint
from factoryModel.utils.helperFunctions import *
 
## Define the file path to input data set
filePath = confFile.DATA_PATH
 
print('[INFO] Starting the preprocessing phase')
 
## Load the raw file and process the data
ss = SentenceSplit(50000)
cd = cleanData()
tm = TrainMaker()

# Initializing the data set loader class and then executing the processing methods
tL = textLoader(preprocessors = [ss,cd,tm])
# Load the raw data, preprocess it and create the train and test sets
max_length_english,max_length_ german,englishTokenizer,germanTokenizer,vocab_size_source,vocab_size_target,X_train,y_train,X_test,y_test = tL.loadDoc(filePath)

print('Training shape',X_train.shape)
print('Testing shape',X_test.shape)
print('Training Y shape',y_train.shape)

print('[INFO] Starting the modelling phase')

### Initiating the training phase #########
# Initialise the model
model = ModelBuilding.EncDecbuild(500,int(max_length_german),int(vocab_size_source),int(vocab_size_target ))
# Define the checkpoints
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Fit the model on the training data set
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
model.fit([X_train, y_train[:,:-1]], y_train.reshape(y_train.shape[0], y_train.shape[1],1)[:,1:], 
                    epochs=13, 
                    callbacks=[es],
                    batch_size=200,
                    validation_data = ([X_test, y_test[:,:-1]], y_test.reshape(y_test.shape[0], y_test.shape[1], 1)[:,1:]))


### Saving the tokenizers and other variables as pickle files
save_clean_data(englishTokenizer, 'factoryModel/output/englishTokenizer.pkl')
save_clean_data(germanTokenizer, 'factoryModel/output/germanTokenizer.pkl')
save_clean_data(max_length_english, 'factoryModel/output/max_length_ english.pkl')
save_clean_data(max_length_ german, 'factoryModel/output/max_length_ german.pkl')
save_clean_data(vocab_size_source, 'factoryModel/output/vocab_size_source.pkl')
save_clean_data(vocab_size_target, 'factoryModel/output/vocab_size_target.pkl')
save_clean_data(X_train, 'factoryModel/output/X_train.pkl')
save_clean_datay_train,'factoryModel/output/y_train.pkl')
save_clean_data(X_test, 'factoryModel/output/X_test.pkl')
save_clean_data(y_test, 'factoryModel/output/y_test.pkl')

