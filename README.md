# Image-Captioning
This repository implements Vanilla RNN, LSTM and GRU in raw numpy and build image captioning models using those. A part of this project was originally build in my [cs231n assignments repo](https://github.com/divyanshj16/cs231n/tree/master/assignment3).

## What is Image Captioning?
Image Captioning is the process of generating textual description of an image. It uses both Natural Language Processing and Computer Vision to generate the captions.

![Image captioning Examples][im-examples]

[im-examples]: https://cdn-images-1.medium.com/max/1600/1*6BFOIdSHlk24Z3DFEakvnQ.png "Some Captioned Image Examples"

## Model Architechture

The model architechture I have used is CNN(VGG16) + RNN. This RNN can be Vanilla RNN, LSTM or GRU.
The VGG extracts features from the image which are fed to the RNN. The RNN then generates caption one word at each time step. The word embeddings are used as input to the RNN. These word embeddings are trained from scratch however they could also have been initialised to Word2Vec or GloVe embeddings.

![Model Architecture][https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/image_folder_6/CNN_RNN.png "CNN + RNN"]

# How to run?

## Requirements

`numpy`
``
``
``

