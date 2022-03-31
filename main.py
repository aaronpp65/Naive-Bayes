from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import re
from collections import Counter

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import stem
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction import stop_words
import seaborn as sns

# stop words corpus
nltk_sw = list(stopwords.words('english'))
sk_sw = list(stop_words.ENGLISH_STOP_WORDS)

# lemmatization
wnl = stem.WordNetLemmatizer()
# stemming
stemer = stem.porter.PorterStemmer()

def cleaner(review):
    # remove special characters characters like "?",":" etc
    # arguments (String) : review 
    # return (String) : cleaned 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', review)
    cleaned = re.sub(r'[?|!|\'|"|#|&|:|-]',r'',cleantext)
    cleaned = re.sub(r'[.|,|)|(|\|/|*]',r' ',cleaned)

    return cleaned

def tokenizer(cleaned):
    # function to perform tokenization and stop words removal
    # arguments (String) : cleaned 
    # return (String) : words 

    word_tokens = word_tokenize(cleaned.lower())
    words = []

    for w in word_tokens:
        if w not in sk_sw:
            words.append(w)

    return word_tokens

def lemmatizer(words):
    # function to perform lemmatization
    # arguments (String) : words 
    # return (String) : lemmatized 
    lemmatized = [wnl.lemmatize(word) for word in words]
    # lemmatized = [stemer.stem(word) for word in words]
    return lemmatized

# bag of words
bag_words = []

# positive and negative reviews
pos_reviews=[]
neg_reviews=[]

# reading csv 
df=pd.read_csv("imdb_master.csv",encoding='ISO-8859-1')
print(df[(df.label == "pos") & (df.type =="train")].sum())

positive_count =0
negative_count =0

# training
for index, row in df.iterrows():

    if(row['type']=='train'):

        # text pre processing
        cleaned = cleaner(row['review'])
        words = tokenizer(cleaned)
        # stm = lemmatizer(words)
        stm = words
                
        # only postitive reviews
        if(row['label']=="pos"):
            pos_reviews+=stm
            positive_count+=1

        # only negative reviews
        elif(row['label']=="neg"):
            neg_reviews+=stm
            negative_count+=1

        bag_words+=stm

        str = " ".join(stm) 




#vocabulary
bag_words=set(bag_words)

# vocabulary size
vocab_size  = len(bag_words)

# no of documents in each class
no_doc_pos = len(pos_reviews)
no_doc_neg = len(neg_reviews)

# initlaizing counter for counting occurences of documents in each class

pos_reviews_counter = Counter()
for word in pos_reviews:
    pos_reviews_counter[word] += 1

neg_reviews_counter = Counter()
for word in neg_reviews:
    neg_reviews_counter[word] += 1

y_test=[]
y_pred=[]


# evaluating classifier against the test data
for index, row in df.iterrows():
    if(row['type']=='test'):

        y_test.append(row['label'])
        
        # text pre processing
        cleaned = cleaner(row['review'])
        words = tokenizer(cleaned)
        # stm = lemmatizer(words)
        stm = words

        # calculating posterior probabilities
        pos_posterior = (positive_count/(positive_count+negative_count))
        neg_posterior =(negative_count/(positive_count+negative_count))

        for word in stm:
            pos_posterior*=(pos_reviews_counter[word]+1)/(no_doc_pos+(1*vocab_size))
            neg_posterior*=(neg_reviews_counter[word]+1)/(no_doc_neg+(1*vocab_size))


        # calculating prediction
        if pos_posterior>neg_posterior:
            predcition="pos"  
        else:
            predcition="neg"

        y_pred.append(predcition)

# creating a confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred)
print(cf_matrix)

# plotting confusion matrix
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                    cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
            zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
ax.set_title('Confusion Matrix\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
plt.show()

# metrics

accuracy = (cf_matrix[0][0]+cf_matrix[1][1])/(cf_matrix[0][0]+cf_matrix[1][1]+cf_matrix[0][1]+cf_matrix[1][0])
print(f"accuracy : {accuracy*100}")

precision = cf_matrix[1][1] / (cf_matrix[1][1]+cf_matrix[0][1])
print(f"precision : {precision}")

recall = cf_matrix[1][1] / (cf_matrix[1][1]+cf_matrix[1][0])
print(f"recall : {recall}")

f1_score = (precision*recall)/(precision+recall)
print(f"f1_score : {2*f1_score}")
