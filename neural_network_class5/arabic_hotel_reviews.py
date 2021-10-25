    
   
#-----# Importing packages #-----#

# General packages
import pandas as pd 
import string
import numpy as np

# For tokenisation
import nltk
from nltk import word_tokenize
nltk.download('punkt')

# TF-IDF vectoriser
from sklearn.feature_extraction.text import TfidfVectorizer
# One-hot encoder
from sklearn.preprocessing import OneHotEncoder

# Loading dataset
from datasets import load_dataset

# Import nn class
import neural_network_as_nnmodule as nnm

# Import torch
import torch
import torch.nn as nn


#-----# Defining main function #-----#

# Defining main function
def main():

    ArabicReviews()


#-----# Defining class #-----#

class ArabicReviews:

    def __init__(self):
        # Defining number of input feature
        self.features = 100
        
        # Loading data
        data = load_dataset('labr')

        # Splitting data
        X_train_text = pd.DataFrame(data.get('train'))['text']
        X_test_text = pd.DataFrame(data.get('test'))['text']
        y_train_labels = pd.DataFrame(data.get('train'))['label']
        y_test_labels = pd.DataFrame(data.get('test'))['label']

        # TF-IDF vectoring
        tfidf_train, tfidf_test = self.tfidf_vectorisation(X_train_text, X_test_text)

        # Initialising model
        model = nnm.Model(n_input_features=self.features)

        # Turn training data into tensors
        X_train = torch.tensor(tfidf_train, dtype=torch.float)
        y_train = torch.tensor(y_train_labels, dtype=torch.float)
        y_train = OneHotEncoder(sparse=False).fit_transform(y_train.reshape(-1,1))
        y_train = [np.reshape(x, (5,1)) for x in y_train]

        # Turn test data into tensors
        X_test = torch.tensor(tfidf_test, dtype=torch.float)
        y_test = torch.tensor(y_train_labels, dtype=torch.float)
        y_test = y_test.view(y_test.shape[0], 1)


        # define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())

        # train
        epochs = 10000
        for epoch in range(epochs):
            # forward
            y_pred = model(X_train) 

            # backward
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # some print to see that it is running
            if (epoch+1) % 1000 == 0:
                print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')




#-----# Helper functions #-----#

    def tfidf_vectorisation(self, train, test , ngram_range=2):

        # Defining TF-IDF vectorizer 
        tfidf_vect = TfidfVectorizer(max_features = self.features, 
                            tokenizer = self._tokenizer_better, 
                            ngram_range = (1,ngram_range), # Include uni-, bi- or trigrams
                            max_df = 0.8)
        
        # Fit vectorizer to train notes
        tfidf_vect.fit(train)

        # Transform our notes into numerical matrices
        tfidf_vect_notes = tfidf_vect.transform(train)

        tfidf_vect_test_notes = tfidf_vect.transform(test)

        # Convert to arrays
        tfidf_vect_notes_array = tfidf_vect_notes.toarray()

        tfidf_vect_notes_test_array = tfidf_vect_test_notes.toarray()

        return tfidf_vect_notes_array, tfidf_vect_notes_test_array


    # Define a tokenizer function
    def _tokenizer_better(self, text):    

        # Define punctuation list
        punc_list = string.punctuation+'0123456789'

        t = str.maketrans(dict.fromkeys(punc_list, ''))

        # Remove punctuaion
        text = text.translate(t)

        # Tokenise 
        tokens = word_tokenize(text)

        return tokens
        
 # Run if run as main
if __name__ == '__main__':
    main()