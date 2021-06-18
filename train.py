''' 

**NOTE: This project is under construction!**

AI THAT MAKE VERSES
Just for fun!
https://github.com/QuangTranUTE/AI-that-write-verses
quangtn@hcmute.edu.vn

INSTRUCTIONS:
    ...

'''


# In[1]: PART 1. IMPORT AND FUNCTIONS
#region
import sys
from tensorflow import keras
assert sys.version_info >= (3, 5) # Python ≥3.5 is required
import tensorflow as tf
assert tf.__version__ >= "2.0" # TensorFlow ≥2.0 is required
import numpy as np
import joblib
from mosestokenizer import MosesTokenizer, MosesDetokenizer
import gdown
import zipfile
import string
import unicodedata  
    
# Setup for GPU usage:
physical_devices = tf.config.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

# Declarations and functions for data preprocessing 
oov_id = 0 # out-of-vocab word id
def word_to_id(word, vocab_list):
    if word in vocab_list:
        return vocab_list.index(word) 
    else:
        return oov_id 
def id_to_word(id, vocab_list):
    return vocab_list[id]

#endregion


# In[2]: PART 2. LOAD AND PREPROCESS DATA
# Hyperparameters:
min_occurrences = 2 # NOTE: HYPERPARAM. Each word appear many times in the dataset. We only keep the words that occur > min_occurrences in the dataset. Amitabha 
n_steps = 37 # NOTE: HYPERPARAM. Length of each training sequence. NOTE: the model can only learn pattern shorter than n_steps
#region
# LOAD DATA:
# Brief description of the training data:
#   The training text is from Truyện Kiều (The Tale of Kiều) -- a poem of 3,254 verses written in six–eigh meter. "The Tale of Kiều is an epic poem in Vietnamese written by Nguyễn Du (1765–1820), considered the most famous poem and a classic in Vietnamese literature." (Wikipedia)
#   The text can be downloaded from `datasets` on https://github.com/QuangTranUTE/AI-that-write-verse 
data_file_path = r"datasets/TruyenKieu_NguyenDu_vnthuquan.txt"
new_download = True
if new_download:
    url_data = 'https://drive.google.com/u/0/uc?id=1evteVh9wMQnC_ca2lV1L6OhzylNX2a4E'  
    download_output = data_file_path
    gdown.download(url_data, download_output, quiet=False)
f = open(data_file_path, "r", encoding="utf-8")
text_data = f.read()
f.close()
print(f'\nText length: {len(text_data)} characters.')
print('\nSome first lines:\n',text_data[:334])

# PREPROCESS DATA:
def preprocess(text_data, for_training=False):
    '''
    Preprocess data.
    Input: text_data: a string 
           for_training: bool. If True: generate vocab and stuff
    Ouput: X_processed: token ids ofpreprocessed text.
           vocab_X_size: size of vocab (only returned when for_training=True)
    '''

    # Delete all \n:
    # INFO: Mosses tokenizer (used below) can NOT deal with \n
    text_data = text_data.replace('\n',' ')

    # Convert to lowercase:
    text_data = text_data.lower() 

    # Replace punctuation by spaces (to remove them):
    marks_to_del = string.punctuation
    table = str.maketrans(marks_to_del, ' '*len(marks_to_del))
    text_data = text_data.translate(table)  

    # Convert charsets (bảng mã) TCVN3, VIQG... to Unicode
    text_data = unicodedata.normalize('NFC', text_data)  

    # Tokenize text using Mosses tokenizer:
    vi_tokenize = MosesTokenizer('vi')
    text_data_tokenized = vi_tokenize(text_data) 
    vi_tokenize.close()

    if for_training:
        # Have a look at a comment and its tokens
        text = 'đã mang lấy nghiệp vào thân  cũng đừng trách lẫn trời gần trời xa  thiện căn ở tại lòng ta  chữ tâm kia mới bằng ba chữ tài'
        print('\n',text)
        with MosesTokenizer('vi') as vi_tokenize:
            tokens = vi_tokenize(text)
        print('\n',tokens)
        with MosesDetokenizer('vi') as detokenize:
            text_back = detokenize(tokens)
        print('\n',text_back)

        # Create vocabularies:
        vocab, counts = np.unique(text_data_tokenized, return_counts=True)
        vocab_count = {word:count for word, count in zip(vocab, counts)}
        print("full vocab.shape: ", vocab.shape)

        # Truncate the vocabulary (keep only words that appear at least min_occurrences times)
        truncated_vocab = dict(filter(lambda ele: ele[1]>=min_occurrences,vocab_count.items()))
        truncated_vocab = dict(sorted(truncated_vocab.items(), key=lambda item: item[1], reverse=True)) # Just to have low ids for most appeared words
        vocab_size = len(truncated_vocab)
        print("truncated vocal_size:", vocab_size)
        
        # Create vocal list to convert words to ids:
        # Preserve 0 for oov-word token
        vocab_list = ['oov']
        vocab_list.extend(list(truncated_vocab.keys()))
        joblib.dump(vocab_list,r'./datasets/vocab_list.joblib')  
        print('Done saving vocab_list.')   

        # Try encode, decoding some samples:
        temp_text = text_data_tokenized[:37]
        print('\ntemp_comment:',temp_text)
        temp_encode = list(map(lambda word: word_to_id(word, vocab_list), temp_text)) 
        print('\ntemp_encode:',temp_encode)
    else:
        vocab_list = joblib.load(r'./datasets/vocab_list.joblib')

    # Convert words (tokens) to ids: list of token ids of text_data_tokenized
    ids = list(map(lambda word: word_to_id(word, vocab_list), text_data_tokenized))  
    
    if for_training:
        print('\nDONE loading and preprocessing data.')
        return np.array(ids), vocab_list
    else: 
        return np.array(ids)

X_processed, vocab_list = preprocess(text_data, for_training=True)
vocab_size = len(vocab_list)

# Chop text into small sequences (N-grams, where N = n_steps HYPERPARAM)
nsteps_grams = [X_processed[i:i+n_steps] for i in range(len(X_processed)-n_steps+1)]   
X_data = np.array(nsteps_grams)

X = X_data[:,:-1]
Y = X_data[:,1:]
#endregion


# In[3]: PART 3. TRAIN A RNN WITH GRU CELLS
# Hyperparameters:
embed_size = 20 # NOTE: HYPERPARAM. embedding output size
n_units = 128 # NOTE: HYPERPARAM. Number of units in a RNN layer. 
n_epochs = 300 # NOTE: HYPERPARAM. Number of epochs to run training.
batch_size = 32 # NOTE: HYPERPARAM. batch_size
#region
# 3.1. Create model
model = keras.models.Sequential([
    keras.layers.Input(shape=[None]),
    keras.layers.Embedding(vocab_size, embed_size, mask_zero=False), 
    keras.layers.GRU(n_units, return_sequences=True),
    keras.layers.GRU(n_units, return_sequences=True),
    keras.layers.GRU(int(n_units/2), return_sequences=True),
    keras.layers.GRU(int(n_units/2), return_sequences=True),
    keras.layers.Dense(vocab_size, activation="softmax") ])
model.summary()

# 3.2. Train the model
new_training = True
if new_training:
    optimizer = 'nadam'
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    checkpoint_name = r'models/truyenKieu_GRU'+'_epoch{epoch:02d}_accuracy{accuracy:.4f}'+'.h5'
    model_checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_name, monitor='accuracy',save_best_only=True)
    early_stop = keras.callbacks.EarlyStopping(monitor='accuracy',patience=10,restore_best_weights=True)
    tensorboard = keras.callbacks.TensorBoard(r'logs/truyenKieu_GRU_log',embeddings_freq=1, embeddings_metadata='embed_file')
    
    model.fit(X, Y, epochs=n_epochs, batch_size=batch_size, callbacks = [model_checkpoint, early_stop, tensorboard],  )
    model.save(r'models/truyenKieu_GRU_last_epoch.h5')
    print('DONE training.')
else:
    print('NO new training.')
#endregion


# In[4]: PART 4. TRAIN A WaveNet-STYLE MODEL
# Hyperparameters:
n_filters = 48 # NOTE: HYPERPARAM. Number of filters
n_epochs = 300 # NOTE: HYPERPARAM. Number of epochs to run training.
batch_size = 32 # NOTE: HYPERPARAM. batch_size
dilation_rates = (1, 2, 4, 8) # NOTE: HYPERPARAM. Dilation rates for 4 Conv1D layers
#region
# 4.1. Create the model
model = keras.models.Sequential()
model.add(keras.layers.Input(shape=[None, 1]))
for rate in dilation_rates*2:
    model.add(keras.layers.Conv1D(filters=n_filters, kernel_size=2, padding="causal", activation="relu", dilation_rate=rate))
model.add(keras.layers.Dense(vocab_size, activation="softmax"))
model.summary()
new_training = True
if new_training:
    model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam")

    checkpoint_name = r'models/truyenKieu_GRU_wavenet'+'_epoch{epoch:02d}_loss{loss:.4f}'+'.h5'
    model_checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_name, monitor='loss',save_best_only=True)  
    early_stop = keras.callbacks.EarlyStopping(monitor='loss',patience=10,restore_best_weights=True)
    tensorboard = keras.callbacks.TensorBoard(r'logs/truyenKieu_GRU_wavenet_log',embeddings_freq=1, embeddings_metadata='embed_file')
    
    model.fit(X[:,:,np.newaxis], Y[:,:,np.newaxis], epochs=n_epochs, batch_size=batch_size, callbacks = [model_checkpoint,early_stop, tensorboard], )
    model.save(r'models/truyenKieu_WaveNetlike.h5')
#endregion


# In[5]: PART 5. TRAIN A STATEFUL RNN
n_units = 128 # NOTE: HYPERPARAM. Number of units in each layer. 
n_epochs = 300 # NOTE: HYPERPARAM. Number of epochs to run training.
batch_size = 32 # NOTE: HYPERPARAM. batch_size
#region
# 5.1. Create proper input sequences (Sequential and Nonoverlapping): 
# Chop text into small sequences (N-grams, where N = n_steps HYPERPARAM):
nsteps_grams = [X_processed[i:i+n_steps] for i in range(0,len(X_processed)-n_steps+1,n_steps-1)]   
X_data = np.array(nsteps_grams)
# Remove last batch (not enough samples for 1 batch):
n_samples = X_data.shape[0]
n_samples_kept = (n_samples // batch_size)*batch_size
X = X_data[:n_samples_kept,:-1]
Y = X_data[:n_samples_kept,1:]

# 5.2. Create a stateful model
model = keras.models.Sequential([
        keras.layers.Input(batch_shape = [batch_size, None, 1]), # stateful require batch_size
        keras.layers.GRU(n_units, return_sequences=True, stateful=True), # NOTE: specify stateful=True
        keras.layers.GRU(n_units, return_sequences=True, stateful=True),
        keras.layers.GRU(n_units, return_sequences=True, stateful=True),
        keras.layers.GRU(n_units, return_sequences=True, stateful=True),
        keras.layers.Dense(vocab_size, activation="softmax") ])

# 5.3. Reset state at the beginning of each epoch
class ResetStatesCallback(keras.callbacks.Callback): 
    def on_epoch_begin(self, epoch, logs): 
        self.model.reset_states()
model.summary()

# 5.4. Train the model
new_training = 10
if new_training:
    model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam")
    steps_per_epoch = X.shape[0] // batch_size // n_steps

    checkpoint_name = r'models/truyenKieu_GRU_stateful'+'_epoch{epoch:02d}_loss{loss:.4f}'+'.h5'
    model_checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_name, monitor='loss',save_best_only=True) 
    early_stop = keras.callbacks.EarlyStopping(monitor='loss',patience=10,restore_best_weights=True)
    tensorboard = keras.callbacks.TensorBoard(r'logs/truyenKieu_GRU_stateful_log',embeddings_freq=1, embeddings_metadata='embed_file')
    
    model.fit(X[:,:,np.newaxis], Y[:,:,np.newaxis], epochs=n_epochs,            batch_size=batch_size, callbacks = [model_checkpoint,early_stop, tensorboard, ResetStatesCallback()], )
    # model.save(r'models/truyenKieu_statefulRNN.h5')

# 5.5. Convert to stateless version to use with different batch size (eg, = 1)
#   Steps to convert:
#       1. Create a stateless model
#       2. Build the model (to create weights)
#       3. Copy the weights from stateful model
stateless_model = keras.models.Sequential([
        keras.layers.Input(shape=[None, 1]),
        keras.layers.GRU(n_units, return_sequences=True),
        keras.layers.GRU(n_units, return_sequences=True),
        keras.layers.GRU(n_units, return_sequences=True),
        keras.layers.GRU(n_units, return_sequences=True),
        keras.layers.Dense(vocab_size, activation="softmax") ])
stateless_model.build(tf.TensorShape([None, None, 1]))
stateless_model.set_weights(model.get_weights())
stateless_model.save(r'models/truyenKieu_statefulRNN_converted.h5')

# In[6]: PART 6. GENERATE VERSES USING TRAINED MODELS
##### NOTE: specify correct model file name below: #####
# model = keras.models.load_model(r'models/truyenKieu_GRU_epoch262_accuracy0.8973.h5')
model = keras.models.load_model(r'models/truyenKieu_statefulRNN_converted.h5')
# model = keras.models.load_model(r'models/truyenKieu_GRU_wavenet_epoch108_loss3.7185.h5')
use_stateful_RNN = False
vocab_list = joblib.load(r'datasets/vocab_list.joblib')
#%%
#region
def next_word(text, model, uniform_randomness=1):
    '''
    Select next char randomly with a probability = y_proba. If we select highest y_proba char, we often get repeated words. To get interesting results, select next char randomly with a probability = y_proba
    Inputs: 
        text: input text
        uniform_randomness: =1: sampling prob = y_proba
                            >>: sampling prob ~ [0, 0, 0,...]: uniform dist.
    Output: next char after 'text'
    (Reference: Géron 2019 Hands-On Machine Learning)
    '''
    X_new = preprocess(text)
    if use_stateful_RNN:
        X_new = np.array([X_new[:, np.newaxis]]) ##### ONLY FOR statefulRNN_converted. # turn to batch of 1 sample of shape [None, 1]
    else:
        X_new = np.array([X_new]) # turn to batch of 1 sample

    y_proba = model.predict(X_new)[0, -1:, :]
    new_y_proba = y_proba / uniform_randomness    
    new_y_proba = new_y_proba.reshape(-1) # change to 1D array
    word_id = np.random.choice( 
        a=np.arange(len(vocab_list)), size=1,  
        p= new_y_proba/np.sum(new_y_proba) # to ensure sum(p) == 1
        ) 
    return id_to_word(word_id[0], vocab_list)

def generate_text(starting_text, model, n_words=12, uniform_randomness=1):
    text =  starting_text
    for _ in range(n_words):  
        word =  next_word(text, model, uniform_randomness)      
        text += ' '+word 
    return text

starting_text = "Trăm năm" # "Chữ tâm"  "Hỏi" "Thanh minh trong"
next_text = generate_text(starting_text, model, n_words=12, uniform_randomness=1.5)
print('\nGenerated next text starting with "{}":\n "{}"'.format(starting_text,next_text))   

#endregion


# %%
