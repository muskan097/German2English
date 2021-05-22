# German2English

# NMT-German2English
![](https://i.imgur.com/fK8LE2y.jpg)

## Table of Content
  * [Overview](#overview)
  * [Technical Aspect](#technical-aspect)
  * [Predictions](#predictions)
 

## Overview
“Sprichst du mit jemandem in einer Sprache, die er versteht, so erreichst du seinen Kopf. Sprichst du mit ihm in seiner eigenen Sprache, so erreichst du sein Herz.” -  NELSON MANDELA

This quote is so right. 

But most of us are not able to understand this because we are not familiar with German language. What if the same quote is written in english:
"If you talk to a man in a language he understands, that goes to his head. If you talk to him in his language, that goes to his heart." -  NELSON MANDELA

Now we got this right. 

Language translation plays a vital role in communication among the residents of different nation’s.  Machine translation can help decrease or even eliminate the language barrier in communication.

What is Machine translation?
Machine translation is the task of automatically converting source text in one language to text in another language

In this project, we will make a deep learning model that will translate German sentences to English Sentences. 

## Technical Aspect
We have used language dataset available on :
Link: [http://www.manythings.org/anki/](http://www.manythings.org/anki/)

The data is a raw text file consisting 227080 sentences. We need to clean and transform the data. English an German sentences are separated to different lists and then converted into data frame. Since the dataset is huge, we have used nearly one fourth of it. 

Data Cleaning: Converted into lower case, removed all punctuations, unnecessary letters an digits. Converted string input to a numerical list using Tokenizer provided by keras-preprocessing library. It is mandatory to have an equal length of all input sequences in sequence-to-sequence models. So padded extra ‘0s’ to make the sequence of the same length using pad_sequence.

Training the model:  split dataset into training and testing with a ratio of 0.1 for precise results. 
we will be using Attention Mechanism. Keras does not officially support attention layer. So, we have used a third-party implementation. We have used "rmsprop" optimizer


## Predictions

![](https://i.imgur.com/vZXZwZt.png)
