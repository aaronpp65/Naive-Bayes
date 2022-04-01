# IMDB sentiment analysis system with Naïve Bayes

### Dataset

Extract the IMDB dataset to project folder

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
