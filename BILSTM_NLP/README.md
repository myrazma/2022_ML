# Bi-LSTM for Named Entity Recognition (NER)
The main focus on this model is the code for working with a Bi-LSTM. Therefore, my focus here is not on analyzing and fine-tuning the model but rather on the logic behind this method. That is also the reason I used a simple Python script and not Jupyter Lab as the objective of this task is not the analysis but rather the code. Therefore, I also **do not provide argument setting from outside the code** here, which I would usually do in a bigger project.

## Model
I am using a single layer Bi-LSTM with 100 hidden units. It will be trained using Adam optimizer and cross-entropy loss over 20 epochs with a batch size of 1. To evaluate the model, the macro-F1 score will be reported.

### Implementation
To implement the model, I will use PyTorch for the model and numpy as well as matplotlib for the evaluation.

## Data 
The data should be stored in the folder 'data/' can be downloaded or viewed here: 
https://www.kaggle.com/alincijov/conll-huggingface-named-entity-recognition/data?select=test.txt 

It should have to form of the following, where the first input is the word and the last input is the NER tag.

>-DOCSTART- -X- -X- O
>
>SOCCER NN B-NP O  
>JAPAN NNP B-NP B-LOC  
>. . O O  
>  
>Nadim NNP B-NP B-PER  
>Ladki NNP I-NP I-PER  

### Embeddings
You can download the embeddings here:
Store the file 'glove.6B.50d.txt' in the folder 'data/' and you're good to go :)


## Run
You can run the code by creating and running a Docker container as follows:
1. Create a Docker image by running
```
docker build -t <docker-name> .
```

and run in bash

```
docker run --rm -it -v "$PWD":/src <docker-name> bash
```

to run with gpu use `--gpus all`

From the bash you can call the sequence tagger directly with

```
python sequence_tagger.py
```
