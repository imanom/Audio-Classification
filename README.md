# Audio-Classification
I've created an audio classifier here - which will classify the audio into two categories - neutral and angry.
Fot the dataset, I've used my own voice samples - 60 Neutral audio files and 50 Angry audio files.

The audio recordings have to be converted into '.wav' format. Any free online converter can be used for that.
I stored the neutral audio files in the path 'Audio_data/neutral' and angry audio files in 'Audio_data/angry', and files in 'Audio_data/test' are for testing our model with individual audio file.

For an in-depth view, please see my python notebook file - 'Voice Classification.ipynb'

### Import libraries

```
import numpy as np
import pandas as pd
import random
import itertools
import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow
import keras
import pickle #To store our model
```

### Load the data
```
import os
path = 'Audio_data/'
neutral= os.listdir(path+'neutral')
angry= os.listdir(path+'angry')
```

### Classifier

I used a sequential model so the model can be built layer by layer.

It has of three layers - an input layer, a hidden layer and an output layer. 

The first layer will receive the input shape. As each sample contains 40 MFCCs (or columns) we have a shape of (1x40) this means we will start with an input shape of 40.

The first two layers will have 256 nodes. The activation function we will be using for our first 2 layers is the ReLU, or Rectified Linear Activation. This activation function has been proven to work well in neural networks.

A Dropout value of 50% will be applied on the first two layers. This will remove random nodes from each cycle, to reduce the likeliness of overfitting.

The output layer will have 2 nodes which matches the number of possible classifications.

```
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from sklearn import metrics 

num_labels = y_train.shape[1]
filter_size = 2

model = Sequential()

model.add(Dense(256, input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels))
model.add(Activation('softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# Calculate pre-training accuracy 
score = model.evaluate(X_test, y_test, verbose=0)
accuracy = 100*score[1]

print("Pre-training accuracy: %.4f%%" % accuracy)
```
### Train the model

Starting with 100 epochs which is the number of times the model will cycle through the data. 
Also beginning with a low batch size, as having a large batch size can reduce the generalisation ability of the model.

```
from datetime import datetime 

num_epochs = 100
num_batch_size = 32
start = datetime.now()

model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test))


duration = datetime.now() - start
print("Training completed in time: ", duration)
```

```
# Evaluating the model on the training and testing set
score = model.evaluate(X_train, y_train, verbose=0)
print("Training Accuracy: ", score[1])

score = model.evaluate(X_test, y_test, verbose=0)
print("Testing Accuracy: ", score[1])
```

Store the model in a pickle file so that we don't have to train the model each time.
```
if not os.path.isdir("result"):
    os.mkdir("result")

pickle.dump(model, open("result/mlp_classifier.model", "wb"))
```

That's it! Now, we can test it with sample data and you get your own personal bit of code which can tell you whether you're angry or not.
