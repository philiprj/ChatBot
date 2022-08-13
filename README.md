# ChatBot
Chat bot using PyTorch 

## Setup
Create a Conda virtual enviornment with Python 3.9 or greater, then install the requirements for this project. 

```console
conda env -n <env_name> python=3.9
conda activate <my_name>
```

```python
pip install -r requirements.txt
```

## Training
To train, simply run
```python
python3 src/train.py
```
## Chat
To chat with the trained chat bot, run
```python
python3 src/chat.py
```

## Libraries
* Python
* PyTorch
* NLTK

## Approach
This method uses a classifier to tag input propmts. The chat responses are then randomly selected from a preset list of predetermined responsed. 

## Methods

Pre-processing
* Stemming - reduce words to common stem, lower case, punctuation
* Tokenization - split sentence into words and punctuation 
* Bag of words - represent words without positional context

Model
* 2 Layer feed-foward neural network using softmax output. 

Later attempts may utilise pre-trained transformers, making use out of sentence structure rather than using bag of words. 