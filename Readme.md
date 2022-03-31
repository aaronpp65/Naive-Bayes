# A sentiment analysis system with Naïve Bayes

A simple Java RMI application that involves both the client and server side of a distributed application: a simple sorting server.

The sorting server operates a stack and clients push values and operations on to the stack and each client will have its own stack. Operations are always sensible: that is, we will only push an operator after pushing at least on value and we will only pop when there is a value on the stack.

### Dataset

Download ad extract the IMDB dataset to project folder

```
imdb
```

### Running


To run 
```
python main.py
```

### Input

The the format of IMDB dataset is  

```
Data_ID, Type, Review Text, label, file_name
```
                                                			 
```Label``` is the class label. In this dataset, there are only two classes, ```pos``` or ```neg```. 50% of samples
are labelled ```unsup``` which means that they are unlabelled samples. You can simply ignore
those samples.
