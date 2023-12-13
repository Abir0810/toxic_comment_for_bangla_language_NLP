#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import seaborn as sb
import tensorflow as tf
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train_data = pd.read_csv(r'E:\AI\toxic comment prediction using nlp\dataset_final.csv')
test_data_final = pd.read_csv(r'E:\AI\toxic comment prediction using nlp\dataset_final.csv')
test_labels = pd.read_csv(r'E:\AI\toxic comment prediction using nlp\dataset_final.csv')


# In[3]:


print(any(train_data['comment'].duplicated())) # see if id values are repeated
print(train_data.isnull().values.any()) # check for any n/a values
display(train_data.describe()) # check statistics of each classification


# In[4]:


train_size = train_data.shape[0]
test_size = test_data_final.shape[0]
total_count = train_size + test_size
print(f"Number of training examples {train_size}")
print(f"Number of test examples {test_size}")
print(f"Train/Test Ratio: {round(train_size/total_count * 100, 2)}% / {round(test_size/total_count * 100, 2)}%")


# In[5]:


train_data_new = train_data.copy()
comment_category_totals=train_data.iloc[:,2:].sum(axis=1)
train_data_new['none']=(comment_category_totals==0)
print(f"Comments with no toxicity: {train_data_new['none'].sum()}")
print(f"% of total training dataset marked as not toxic: {train_data_new['none'].sum()/len(train_data_new)}")


# In[6]:


print(train_data_new[['vulgar', 'not_bully', 'religious', 'hate', 'threat', 'insult', 'none']].sum())


# In[7]:


x_vals=train_data_new.iloc[:,1:].sum()


# In[8]:


labels = ['vulgar', 'not_bully', 'religious', 'hate', 'threat', 'insult']
plt.title('Ratio of Positive Toxicity Labels in Training Dataset')
plt.pie(train_data_new[labels].sum(), labels=labels, autopct='%1.1f%%')
plt.show()


# In[9]:


# Word count in each comment:
train_data_new['count_word'] = train_data_new["comment"].apply(lambda x: len(str(x).split()))

# Unique word count
train_data_new['count_unique_word'] = train_data_new["comment"].apply(lambda x: len(set(str(x).split())))

# percentage of unique words
train_data_new['word_unique_percent'] = train_data_new['count_unique_word']*100/train_data_new['count_word']

# marking comments without any tags as "clean".
rowsums = train_data_new.iloc[:,2:7].sum(axis=1)
train_data_new['clean'] = np.logical_not(rowsums).astype('int')


# In[10]:


import seaborn as sns
# word count
axes = sns.violinplot(y='count_word', x='clean', data=train_data_new,split=True, inner="quart")
axes.set_xlabel('Type of comment', fontsize=12)
axes.set_ylabel('# of words', fontsize=12)
axes.set_ylim([0, 250])
axes.set_xticklabels(['Toxic', 'Clean'], rotation='vertical', fontsize=10)
plt.title("Number of words in each comment", fontsize=15)

plt.show()


# In[11]:


# unique word count
plt.title("Number of unique words in each comment", fontsize=15)
axes = sns.violinplot(y='count_unique_word', x='clean', data=train_data_new,split=True, inner="quart")
axes.set_xlabel('Type of comment', fontsize=12)
axes.set_ylabel('# of words', fontsize=12)
axes.set_ylim([0, 200])
axes.set_xticklabels(['Toxic', 'Clean'], rotation='vertical', fontsize=10)
plt.show()

# percentage of unique words
plt.title("Percentage of unique words of total words in comment")
axes = sns.violinplot(y='word_unique_percent', x='clean', data=train_data_new,split=True, inner="quart")
axes.set_xlabel('Type of comment', fontsize=12)
axes.set_ylabel('% of words', fontsize=12)
axes.set_xticklabels(['Toxic', 'Clean'], rotation='vertical', fontsize=10)
plt.show()


# In[12]:


from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# combine training and test comments so we have the entire corpus of words and characters for vectorization
combined_comments = pd.concat([train_data['comment'], test_data_final['comment']])
word_vec = TfidfVectorizer(stop_words='english', analyzer='word', strip_accents='unicode', sublinear_tf=True,
                           token_pattern=r'\w{1,}', max_features=10000, ngram_range=(1,1))
word_vec.fit(combined_comments)
train_word_features = word_vec.transform(train_data['comment'])
test_word_features = word_vec.transform(test_data_final['comment'])

character_vec = TfidfVectorizer(analyzer='char', stop_words='english', strip_accents='unicode', sublinear_tf=True,
                               max_features=50000, ngram_range=(2,6))
character_vec.fit(combined_comments)
train_char_features = character_vec.transform(train_data['comment'])
test_char_features = character_vec.transform(test_data_final['comment'])

train_data_features = hstack([train_char_features, train_word_features])
test_data_features = hstack([test_char_features, test_word_features])


# In[13]:


class_names = ['vulgar', 'not_bully', 'religious', 'hate', 'threat', 'insult']


# In[14]:


from sklearn.metrics import classification_report


# # LogisticRegression

# In[15]:


get_ipython().run_cell_magic('time', '', "from sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import cross_val_score\n\nscores = []\npredictions = pd.DataFrame.from_dict({'comment': test_data_final['comment']})\nfor class_name in class_names:\n    train_target = train_data[class_name]\n    classifier = LogisticRegression(C=0.1, solver='sag')\n\n    cv_score = np.mean(cross_val_score(classifier, train_data_features, train_target, cv=3, scoring='roc_auc'))\n    scores.append(cv_score)\n    print('CV score for class {} is {}'.format(class_name, cv_score))\n\n    classifier.fit(train_data_features, train_target)\n    predictions[class_name] = classifier.predict_proba(test_data_features)[:, 1]\n\nprint('Total CV score is {}'.format(np.mean(scores)))\n")


# In[16]:


from sklearn.metrics import roc_auc_score
def test_score(predicted_vals):
    '''Actual Kaggle competion score using test set labels provided after competition close'''
    actuals = test_labels.copy()
    actuals.drop(actuals.loc[actuals['comment']==-1].index, inplace=True)
    scores_test = []
    for class_name in class_names:
        score = roc_auc_score(actuals[class_name], predicted_vals[class_name])
        scores_test.append(score)
    print('Test set score is {}'.format(np.mean(scores_test)))


# In[17]:


test_score(predictions)


# In[18]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


# In[19]:


import re
import nltk
from nltk.util import pr


# In[20]:


df = pd.read_csv(r'E:\AI\toxic comment prediction using nlp\da.csv')


# In[21]:


x=np.array(df["comment"])
y=np.array(df["label"])


# In[22]:


cv=CountVectorizer()


# In[23]:


x=cv.fit_transform(x)


# In[24]:


from sklearn.model_selection import train_test_split


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(
x, y, test_size=0.33, random_state=42)


# In[26]:


from sklearn.linear_model import LogisticRegression


# In[27]:


clg = LogisticRegression(random_state=0)


# In[28]:


clg.fit(X_train, y_train)


# In[29]:


test_data="ক্যাপ্টেন অফ বাংলাদেশ"
df=cv.transform([test_data]).toarray()
print(clg.predict(df))


# In[30]:


predictions = clg.predict(X_test)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predictions, labels=clg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clg.classes_)
disp.plot()
plt.show()


# In[31]:


y_predict = classifier.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_predict))


# # Support Vector Machine

# In[32]:


get_ipython().run_cell_magic('time', '', "from sklearn.svm import LinearSVC\nfrom sklearn.calibration import CalibratedClassifierCV\n\nscores = []\npredictions = pd.DataFrame.from_dict({'comment': test_data_final['comment']})\nfor class_name in class_names:\n    train_target = train_data[class_name]\n    classifier = CalibratedClassifierCV(LinearSVC(), method='sigmoid',cv=4)\n\n    cv_score = np.mean(cross_val_score(classifier, train_data_features, train_target, cv=3, scoring='roc_auc'))\n    scores.append(cv_score)\n    print('CV score for class {} is {}'.format(class_name, cv_score))\n\n    classifier.fit(train_data_features, train_target)\n    predictions[class_name] = classifier.predict_proba(test_data_features)[:, 1]\n\nprint('Total CV score is {}'.format(np.mean(scores)))\n")


# In[33]:


test_score(predictions)


# In[36]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


# In[37]:


import re
import nltk
from nltk.util import pr


# In[38]:


df = pd.read_csv(r'E:\AI\toxic comment prediction using nlp\da.csv')


# In[39]:


x=np.array(df["comment"])
y=np.array(df["label"])


# In[40]:


cv=CountVectorizer()


# In[41]:


x=cv.fit_transform(x)


# In[42]:


from sklearn.model_selection import train_test_split


# In[43]:


X_train, X_test, y_train, y_test = train_test_split(
x, y, test_size=0.33, random_state=42)


# In[44]:


from sklearn import svm


# In[45]:


sv = svm.SVC()


# In[46]:


sv.fit(X_train, y_train)


# In[48]:


test_data="ক্যাপ্টেন অফ বাংলাদেশ"
df=cv.transform([test_data]).toarray()
print(sv.predict(df))


# In[49]:


predictions = sv.predict(X_test)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predictions, labels=sv.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=sv.classes_)
disp.plot()
plt.show()


# In[51]:


y_predict = sv.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_predict))


# # ANN

# In[52]:


get_ipython().run_cell_magic('time', '', "from sklearn.neural_network import MLPClassifier\nfrom sklearn.model_selection import cross_val_score\n\nscores = []\npredictions = pd.DataFrame.from_dict({'comment': test_data_final['comment']})\nfor class_name in class_names:\n    train_target = train_data[class_name]\n    classifier = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)\n\n    cv_score = np.mean(cross_val_score(classifier, train_data_features, train_target, cv=3, scoring='roc_auc'))\n    scores.append(cv_score)\n    print('CV score for class {} is {}'.format(class_name, cv_score))\n\n    classifier.fit(train_data_features, train_target)\n    predictions[class_name] = classifier.predict_proba(test_data_features)[:, 1]\n\nprint('Total CV score is {}'.format(np.mean(scores)))\n")


# In[53]:


test_score(predictions)


# In[54]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


# In[55]:


import re
import nltk
from nltk.util import pr


# In[56]:


df = pd.read_csv(r'E:\AI\toxic comment prediction using nlp\da.csv')


# In[57]:


x=np.array(df["comment"])
y=np.array(df["label"])


# In[58]:


cv=CountVectorizer()


# In[59]:


x=cv.fit_transform(x)


# In[60]:


from sklearn.model_selection import train_test_split


# In[61]:


X_train, X_test, y_train, y_test = train_test_split(
x, y, test_size=0.33, random_state=42)


# In[62]:


from sklearn.neural_network import MLPClassifier


# In[63]:


anna = MLPClassifier(solver='lbfgs', alpha=1e-5,
hidden_layer_sizes=(5, 2), random_state=1)


# In[64]:


anna.fit(X_train, y_train)


# In[65]:


test_data="ক্যাপ্টেন অফ বাংলাদেশ"
df=cv.transform([test_data]).toarray()
print(anna.predict(df))


# In[66]:


predictions = anna.predict(X_test)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predictions, labels=anna.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=anna.classes_)
disp.plot()
plt.show()


# In[67]:


from sklearn.metrics import classification_report


# In[68]:


classification_report(y_test,predictions)


# In[69]:


classifier_tree = MLPClassifier(max_iter=500,activation='relu',hidden_layer_sizes=(2, 2))


# In[70]:


y_predict = classifier_tree.fit(X_train, y_train).predict(X_test)


# In[71]:


print(classification_report(y_test, y_predict))


# # Naive bayes

# In[72]:


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 


# In[73]:


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]


# In[74]:


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 

#from scikit-learn website
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]

def main():
   
   #3. Pre-processing?(could skip)
   #4. Use sklearn.feature_extraction.text.CountVectorizer to make matrix of words based of comments

   #1. Group by toxic for the training data
   train_data = pd.read_csv(r'E:\AI\toxic comment prediction using nlp\da.csv')
   train = train_data[["comment"]]
   print(train.shape)
   train_labels = train_data[["label"]]
   print(train_labels.shape)

   #2. Use train_test_split to split into train/test
   comment_train, comment_test, labels_train, labels_test = train_test_split(train, train_labels, test_size = 0.2, random_state=42)
   #Transpose and flatten so it fits the correct dimensions
   labels_train = np.transpose(labels_train)
   labels_train = np.ravel(labels_train)
   labels_test = np.transpose(labels_test)
   labels_test = np.ravel(labels_test)

   #3. CountVectorizer
   #Create a count matrix for each comment
   count_vect = CountVectorizer(tokenizer = LemmaTokenizer(),
                                strip_accents = 'unicode', # works 
                                lowercase = True)
   comment_train_counts = count_vect.fit_transform(comment_train.comment)

   #4. TfidfTransformer
   #Use tf-idf instead
   tf_transformer = TfidfTransformer(use_idf=False).fit(comment_train_counts)
   comment_train_tf = tf_transformer.transform(comment_train_counts)
   tfidf_transformer = TfidfTransformer()
   comment_train_tfidf = tfidf_transformer.fit_transform(comment_train_counts)

   #5 Train a classifier
   #create the model
   clf = MultinomialNB().fit(comment_train_tfidf, labels_train)

   #make the bag of words for the test data
   comment_test_new_counts = count_vect.transform(comment_test.comment)
   comment_test_new_tfidf = tfidf_transformer.transform(comment_test_new_counts)

   print(comment_test_new_tfidf)
   
   #6 Prediction:
   prediction = clf.predict(comment_test_new_tfidf)
   
   print(prediction)

   print("Accuracy:", np.mean(prediction == labels_test), "\n")

   print("Precision, Recall, and F1 Score:\n", metrics.classification_report(labels_test, prediction), "\n")
   print("Confusion Matrix:\n", metrics.confusion_matrix(labels_test, prediction), "\n")

   #count_vect.vocabulary_.get(u'algorithm')
   #print(comment_train_counts)

   #print(count_vect.vocabulary_)

   #print(X_train_counts.shape())
   # test_data = pd.read_csv("jigsaw-toxic-comment-classification-challenge/test.csv")
   # test = test_data.drop(["id"], axis=1)
   # print(test.shape)
   # test_labels = pd.read_csv("jigsaw-toxic-comment-classification-challenge/test_labels.csv")
   # #test_labels = test_labels.drop(["id"], axis=1)
   # test_labels = test_labels[["toxic"]]
   # print(test_labels)



   #bayes_model = MultinomialNB().fit(train, train_labels)
   #predictions = bayes_model.score(test_data, test_labels)


if __name__ == '__main__':
   main()


# In[75]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


# In[76]:


import re
import nltk
from nltk.util import pr


# In[77]:


df = pd.read_csv(r'E:\AI\toxic comment prediction using nlp\da.csv')


# In[78]:


df.head()


# In[79]:


df['label'].value_counts()


# In[80]:


import re


# In[81]:


def get_clean(x):
    x = str(x).lower().replace('\\', '').replace('_', ' ')
    x = re.sub(r'https?://[A-Za-z0-9./]+', '', x)
    x = re.sub("(.)\\1{2,}", "\\1", x)
    return x


# In[82]:


TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(x):
    return TAG_RE.sub('', x)


# In[83]:


df['Comment']=df['comment'].apply(lambda x: get_clean(x))


# # Decision Tree

# In[84]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[85]:


from sklearn.model_selection import train_test_split


# In[86]:


from sklearn import tree


# In[87]:


from sklearn.metrics import classification_report


# In[88]:


tfidf=TfidfVectorizer(max_features=2000,ngram_range=(1,3),analyzer='char')


# In[89]:


X=tfidf.fit_transform(df['Comment'])
y=df['label']


# In[90]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[91]:


logmodel = tree.DecisionTreeClassifier()


# In[92]:


logmodel.fit(X_train, y_train)


# In[93]:


predictions = logmodel.predict(X_test)


# In[94]:


from sklearn.metrics import accuracy_score


# In[95]:


accuracy_score(y_test,predictions)


# In[96]:


from sklearn.metrics import accuracy_score


# In[97]:


accuracy_score(y_test,predictions)


# In[98]:


from sklearn.metrics import classification_report


# In[99]:


classification_report(y_test,predictions)


# In[100]:


x='গালি দিলাম না বাইন্সদ'
x=get_clean(x)
vec=tfidf.transform([x])
logmodel.predict(vec)


# In[101]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


# In[102]:


import re
import nltk
from nltk.util import pr


# In[103]:


df = pd.read_csv(r'E:\AI\toxic comment prediction using nlp\da.csv')


# In[104]:


x=np.array(df["comment"])
y=np.array(df["label"])


# In[105]:


cv=CountVectorizer()


# In[106]:


x=cv.fit_transform(x)


# In[107]:


from sklearn.model_selection import train_test_split


# In[108]:


X_train, X_test, y_train, y_test = train_test_split(
x, y, test_size=0.33, random_state=42)


# In[109]:


from sklearn import tree


# In[110]:


clf = tree.DecisionTreeClassifier()


# In[111]:


clf = clf.fit(X_train, y_train)


# In[112]:


clf.score(X_train, y_train)


# In[113]:


test_data="ক্যাপ্টেন অফ বাংলাদেশ"
df=cv.transform([test_data]).toarray()
print(clf.predict(df))


# In[114]:


predictions = clf.predict(X_test)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
disp.plot()
plt.show()


# In[115]:


from sklearn.metrics import classification_report


# In[116]:


classification_report(y_test,predictions)


# In[117]:


y_predict = clf.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_predict))


# # KNN

# In[118]:


from sklearn.neighbors import KNeighborsClassifier


# In[119]:


df = pd.read_csv(r'E:\AI\toxic comment prediction using nlp\da.csv')


# In[120]:


x=np.array(df["comment"])
y=np.array(df["label"])


# In[121]:


cv=CountVectorizer()


# In[122]:


x=cv.fit_transform(x)


# In[123]:


from sklearn.model_selection import train_test_split


# In[124]:


X_train, X_test, y_train, y_test = train_test_split(
x, y, test_size=0.33, random_state=42)


# In[125]:


neigh = KNeighborsClassifier(n_neighbors=1)


# In[126]:


neigh.fit(X_train, y_train)


# In[127]:


neigh.score(X_train, y_train)


# In[128]:


test_data="ক্যাপ্টেন অফ বাংলাদেশ"
df=cv.transform([test_data]).toarray()
print(neigh.predict(df))


# In[129]:


predictions = neigh.predict(X_test)


# In[130]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[131]:


cm = confusion_matrix(y_test, predictions, labels=neigh.classes_)


# In[132]:


disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=neigh.classes_)


# In[133]:


disp.plot()
plt.show()


# In[134]:


from sklearn.metrics import classification_report


# In[135]:


classification_report(y_test,predictions)


# In[136]:


y_predict = neigh.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_predict))


# # LSTM

# In[137]:


import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sqlite3 import Error
from sklearn.ensemble import RandomForestClassifier
import sqlite3
import pickle
import nltk
nltk.download('stopwords')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[138]:


df = pd.read_csv(r'E:\AI\toxic comment prediction using nlp\da.csv')


# In[139]:


df.groupby('label').comment.count().plot.bar(ylim=0)
plt.show()


# In[140]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import seaborn as sns
import re,nltk,json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras import models
from keras import layers
from tensorflow.keras.layers import LSTM,GRU
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from sklearn.metrics import average_precision_score,roc_auc_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
np.random.seed(42)
class color: # Text style
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


# In[141]:


data = pd.read_csv(r'E:\AI\toxic comment prediction using nlp\da.csv')


# In[142]:


data.columns


# In[143]:


# Cleaning Data [Remove unncessary symbols]
def cleaning_data(row):
      comment = re.sub('[^\u0980-\u09FF]',' ',str(row)) #removing unnecessary punctuation
      return comment
# Apply the function into the dataframe
data['cleaned'] = data['comment'].apply(cleaning_data)  

# print some cleaned reviews from the dataset
sample_data = [2000,5000,10000,20000,30000,35000,40000,45000,50000]
for i in sample_data:
  print('Original: ',data.comment[i],'\nCleaned:',
           data.cleaned[i],'\n','label:-- ',data.label[i],'\n')   


# In[144]:


# Length of each headlines
data['length'] = data['cleaned'].apply(lambda x:len(x.split()))
# Remove the headlines with least words
dataset = data.loc[data.length>2]
dataset = dataset.reset_index(drop = True)
print("After Cleaning:","\nRemoved {} Small comment".format(len(data)-len(dataset)),
      "\nTotal comment:",len(dataset))


# In[145]:


def data_summary(dataset):
    
    """
    This function will print the summary of the headlines and words distribution in the dataset. 
    
    Args:
        dataset: list of cleaned sentences   
        
    Returns:
        Number of documnets per class: int 
        Number of words per class: int
        Number of unique words per class: int
    """
    documents = []
    words = []
    u_words = []
    total_u_words = [word.strip().lower() for t in list(dataset.cleaned) for word in t.strip().split()]
    class_label= [k for k,v in dataset.label.value_counts().to_dict().items()]
  # find word list
    for label in class_label: 
        word_list = [word.strip().lower() for t in list(dataset[dataset.label==label].cleaned) for word in t.strip().split()]
        counts = dict()
        for word in word_list:
                counts[word] = counts.get(word, 0)+1
        # sort the dictionary of word list  
        ordered = sorted(counts.items(), key= lambda item: item[1],reverse = True)
        # Documents per class
        documents.append(len(list(dataset[dataset.label==label].cleaned)))
        # Total Word per class
        words.append(len(word_list))
        # Unique words per class 
        u_words.append(len(np.unique(word_list)))
       
        print("\nClass Name : ",label)
        print("Number of Documents:{}".format(len(list(dataset[dataset.label==label].cleaned))))  
        print("Number of Words:{}".format(len(word_list))) 
        print("Number of Unique Words:{}".format(len(np.unique(word_list)))) 
        print("Most Frequent Words:\n")
        for k,v in ordered[:10]:
              print("{}\t{}".format(k,v))
    print("Total Number of Unique Words:{}".format(len(np.unique(total_u_words))))           
   
    return documents,words,u_words,class_label

#call the fucntion
documents,words,u_words,class_names = data_summary(dataset)    


# In[146]:


data_matrix = pd.DataFrame({'Total Toxic comments':documents,
                            'Total comments':words,
                            'Unique Words':u_words,
                            'Class Names':class_names})
df = pd.melt(data_matrix, id_vars="Class Names", var_name="Category", value_name="Values")
plt.figure(figsize=(8, 6))
ax = plt.subplot()

sns.barplot(data=df,x='Class Names', y='Values' ,hue='Category')
ax.set_xlabel('Class Names') 
ax.set_title('Data Statistics')

ax.xaxis.set_ticklabels(class_names, rotation=45);


# In[147]:


# Calculate the Review of each of the Review
dataset['HeadlineLength'] = dataset.cleaned.apply(lambda x:len(x.split()))
frequency = dict()
for i in dataset.HeadlineLength:
    frequency[i] = frequency.get(i, 0)+1

plt.bar(frequency.keys(), frequency.values(), color ="b")
plt.xlim(1, 20)
# in this notbook color is not working but it should work.
plt.xlabel('Length of the Comment')
plt.ylabel('Frequency')
plt.title('Length-Frequency Distribution')
plt.show()  
print(f"Maximum Length of a headline: {max(dataset.HeadlineLength)}")
print(f"Minimum Length of a headline: {min(dataset.HeadlineLength)}")
print(f"Average Length of a headline: {round(np.mean(dataset.HeadlineLength),0)}")


# In[148]:


def label_encoding(category,bool):
    """
    This function will return the encoded labels in array format. 
    
    Args:
        category: series of class names(str)
        bool: boolean (True or False)
        
    Returns:
        labels: numpy array 
    """
    le = LabelEncoder()
    le.fit(category)
    encoded_labels = le.transform(category)
    labels = np.array(encoded_labels) # Converting into numpy array
    class_names =le.classes_ ## Define the class names again
    if bool == True:
        print("\n\t\t\t===== Label Encoding =====","\nClass Names:-->",le.classes_)
        for i in sample_data:
            print(category[i],' ', encoded_labels[i],'\n')

    return labels



                           #===========================================================
                           ################# Dataset Splitting Function ###############
                           #=========================================================== 

def dataset_split(headlines,category):
    """
    This function will return the splitted (90%-10%-10%) feature vector . 
    
    Args:
        headlines: sequenced headlines 
        category: encoded lables (array) 
        
    Returns:
        X_train: training data 
        X_valid: validation data
        X_test : testing feature vector 
        y_train: training encoded labels (array) 
        y_valid: training encoded labels (array) 
        y_test : testing encoded labels (array) 
    """

    X,X_test,y,y_test = train_test_split(headlines,category,train_size = 0.9,
                                                  test_size = 0.1,random_state =0)
    X_train,X_valid,y_train,y_valid = train_test_split(X,y,train_size = 0.8,
                                                  test_size = 0.2,random_state =0)
    print(color.BOLD+"\nDataset Distribution:\n"+color.END)
    print("\tSet Name","\t\tSize")
    print("\t========\t\t======")

    print("\tFull\t\t\t",len(headlines),
        "\n\tTraining\t\t",len(X_train),
        "\n\tTest\t\t\t",len(X_test),
        "\n\tValidation\t\t",len(X_valid))
  
    return X_train,X_valid,X_test,y_train,y_valid,y_test


# In[149]:


labels = label_encoding(dataset.label,True)


# In[150]:


X_train,X_valid,X_test,y_train,y_valid,y_test = dataset_split(dataset.comment,labels)


# In[151]:


vocab_size = 45000
embedding_dim = 64
max_length = 21
trunc_type='post'
padding_type='post'
oov_tok = ""

def padded_headlines(original,encoded,padded):
  '''
  print the samples padded headlines
  '''
  print(color.BOLD+"\n\t\t\t====== Encoded Sequences ======"+color.END,"\n")  
  print(original,"\n",encoded) 
  print(color.BOLD+"\n\t\t\t====== Paded Sequences ======\n"+color.END,original,"\n",padded)  


# In[152]:


# Train Data Tokenization
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index
train_sequences = tokenizer.texts_to_sequences(X_train)
train_padded = pad_sequences(train_sequences, padding=padding_type, maxlen=max_length)


# In[153]:


#============================== Tokenizer Info =================================
(word_counts,word_docs,word_index,document_count) = (tokenizer.word_counts,
                                                       tokenizer.word_docs,
                                                       tokenizer.word_index,
                                                       tokenizer.document_count)
def tokenizer_info(mylist,bool):
  ordered = sorted(mylist.items(), key= lambda item: item[1],reverse = bool)
  for w,c in ordered[:10]:
    print(w,"\t",c)
  #=============================== Print all the information =========================
print(color.BOLD+"\t\t\t====== Tokenizer Info ======"+color.END)   
print("Words --> Counts:")
tokenizer_info(word_counts,bool =True )
print("\nWords --> Documents:")
tokenizer_info(word_docs,bool =True )
print("\nWords --> Index:")
tokenizer_info(word_index,bool =True )    
print("\nTotal Documents -->",document_count)
print(f"Found {len(word_index)} unique tokens")


# In[154]:


padded_headlines(X_train[10],train_sequences[10],train_padded[10]) 


# In[155]:


# Validation Data Tokenization
validation_sequences = tokenizer.texts_to_sequences(X_valid)
validation_padded = pad_sequences(validation_sequences, padding=padding_type , maxlen=max_length)


# In[156]:


# Test Data Tokenization
test_sequences = tokenizer.texts_to_sequences(X_test)
test_padded = pad_sequences(test_sequences, padding=padding_type , maxlen=max_length)


# In[157]:


# Labels Tokenization
#label_tokenizer = Tokenizer()
#label_tokenizer.fit_on_texts(dataset.category)

train_label_seq = y_train
valid_label_seq = y_valid
testing_label_seq = y_test

#print(train_label_seq.shape)
#print(valid_label_seq.shape)
#print(testing_label_seq.shape)


# In[158]:


path='E:\AI'


# In[159]:


keras.backend.clear_session()
accuracy_threshold = 0.97
vocab_size = 45000
embedding_dim = 64
max_length = 21
num_category = 6

class myCallback(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
      if(logs.get('accuracy')>accuracy_threshold):
        print("\nReached %2.2f%% accuracy so we will stop trianing" % (accuracy_threshold*100))
        self.model.stop_training = True

acc_callback = myCallback()
# Saved the Best Model
filepath = path+"Model.h5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=2, save_best_only=True, 
                                             save_weights_only=False, mode='max')
callback_list = [acc_callback, checkpoint] 
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(GRU(64,dropout=0.2)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(num_category, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()


# In[160]:


num_epochs = 10
batch = 64
history = model.fit(train_padded, train_label_seq, 
                    epochs=num_epochs,
                    batch_size = batch,
                    validation_data=(validation_padded, valid_label_seq), 
                    verbose=1,
                    callbacks = callback_list)


# In[161]:


from sklearn.metrics import classification_report, confusion_matrix
# load the Saved model from directory
model = load_model(path+"Model.h5")
predictions = model.predict(test_padded)
y_pred = np.argmax(predictions, axis=1)

cm = confusion_matrix(testing_label_seq, y_pred) 

# Transform to df for easier plotting
cm_df = pd.DataFrame(cm,
                     
                     index = ['vulgar' ,'not bully' ,'religious', 'hate', 'threat', 'insult'], 
                     columns = ['vulgar' ,'not bully' ,'religious', 'hate', 'threat', 'insult'])

plt.figure(figsize=(8,6))
sns.heatmap(cm_df, annot=True,cmap="YlGnBu", fmt='g')
plt.title('GRU \nAccuracy: {0:.2f}'.format(accuracy_score(testing_label_seq, y_pred)*100))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.xticks(rotation = 45)
plt.yticks(rotation = 45)
plt.show()


# In[162]:


report = pd.DataFrame(classification_report(y_true = testing_label_seq, y_pred = y_pred, output_dict=True)).transpose()
report = report.rename(index={'0': 'vulgar','1':'not bully','2':'religious','3':'hate','4':'threat','5':'insult'})
report[['precision','recall','f1-score']]=report[['precision','recall','f1-score']].apply(lambda x: round(x*100,2))
report


# In[194]:


hist = model.history.history

plt.figure(figsize=(20, 5))

plt.subplot(1, 2, 1)
plt.title("Loss")
plt.plot(hist["loss"], label="Training loss")
plt.plot(hist["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(range(len(hist["loss"])))
plt.legend(loc="upper right")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.title("Accuracy")
plt.plot(hist["accuracy"], label="Training accuracy")
plt.plot(hist["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.xticks(range(len(hist["accuracy"])))
plt.legend(loc="lower right")
plt.grid(True)

plt.show()


# In[164]:


import numpy as np
import pandas as pd

#data visualisation libraries
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

#to avoid warnings
import warnings
warnings.filterwarnings('ignore')


# In[165]:


data = pd.read_csv(r'E:\AI\toxic comment prediction using nlp\dataset_final.csv')
#df = df.drop(['Toxic', 'NonToxic'], axis=1 )
#reset index
#df.set_index('id', inplace = True)
data.head()


# In[166]:


column_labels = data.columns.tolist()[2:]
label_counts = data[column_labels].sum().sort_values()
 
 
# Create a black background for the plot
plt.figure(figsize=(7, 5))
 
# Create a horizontal bar plot using Seaborn
ax = sns.barplot(x=label_counts.values,
                 y=label_counts.index, palette='viridis')
 
 
# Add labels and title to the plot
plt.xlabel('Number of Occurrences')
plt.ylabel('Labels')
plt.title('Distribution of Label Occurrences')
 
# Show the plot
plt.show()


# In[167]:


data[column_labels].sum().sort_values()


# In[168]:


train_toxic = data[data[column_labels].sum(axis=1) > 0]
train_clean = data[data[column_labels].sum(axis=1) == 0]
 
# Number of toxic and clean comments
num_toxic = len(train_toxic)
num_clean = len(train_clean)
 
# Create a DataFrame for visualization
plot_data = pd.DataFrame(
    {'Category': ['Toxic', 'Clean'], 'Count': [num_toxic, num_clean]})
 
# Create a black background for the plot
plt.figure(figsize=(7, 5))
 
# Horizontal bar plot
ax = sns.barplot(x='Count', y='Category', data=plot_data, palette='viridis')
 
 
# Add labels and title to the plot
plt.xlabel('Number of Comments')
plt.ylabel('Category')
plt.title('Distribution of Toxic and Clean Comments')
 
# Set ticks' color to white
ax.tick_params()
 
# Show the plot
plt.show()


# In[169]:


print(train_toxic.shape)
print(train_clean.shape)


# In[170]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


# In[171]:


df = pd.read_csv(r'E:\AI\toxic comment prediction using nlp\dataset_final.csv')
df.head()


# In[172]:


train_df = pd.read_csv(r'E:\AI\toxic comment prediction using nlp\dataset_final.csv')
train_df.head(1)


# In[173]:


labels = train_df.columns[1:].to_numpy()
labels


# In[174]:


train_sentences = df["comment"]
train_labels = df.iloc[:, 1:]

print(f"Train sentences: {train_sentences.shape}")
print(f"Train labels: {train_labels.shape}")


# In[175]:


from sklearn.model_selection import train_test_split

train_pct = 0.4

train_sentences, val_sentences, train_labels, val_labels = train_test_split(
    train_sentences,
    train_labels,
    train_size=0.4
)

print(f"Train sentences: {train_sentences.shape}")
print(f"Train labels: {train_labels.shape}")
print(f"Validation sentences: {val_sentences.shape}")
print(f"Validation labels: {val_labels.shape}")


# In[176]:


import string

def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))

def remove_special_chars(text):
    return text.replace("\n", " ").replace("\t", " ").strip()

def normalize_spacing(text):
    return " ".join(text.split())

def process_text(text):
    text = remove_punctuation(text)
    text = remove_special_chars(text)
    text = normalize_spacing(text)
    return text


# In[177]:


train_sentences = train_sentences.map(process_text)
val_sentences = val_sentences.map(process_text)


# In[178]:


print("Word count statistics:\n")
train_sentences.apply(len).describe()


# In[179]:


train_sentences = train_sentences.to_numpy()
train_labels = train_labels.to_numpy()
val_sentences = val_sentences.to_numpy()
val_labels = val_labels.to_numpy()


# In[180]:


train_sentences


# In[181]:


# let's figure out how many unique words there are in our vocabulary

vocab = set()
for sentence in train_sentences:
    for word in sentence.split():
        vocab.add(word)
        
vocab_size = len(vocab)
print(f"Vocab size: {vocab_size}")


# In[182]:


vocab_size=100000


# In[183]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(train_sentences)


# In[184]:


import json

vocab = {}
for word, index in tokenizer.word_index.items():
    if index <= vocab_size:
        vocab[word] = index
        
print(len(vocab))

with open("tokenizer_dictionary.json", "w") as file:
    json.dump(vocab, file)


# In[185]:


# convert sentences to integer sequences
train_sequences = tokenizer.texts_to_sequences(train_sentences)
val_sequences = tokenizer.texts_to_sequences(val_sentences)

# pad integer sequences into fixed length
max_length = 300
padding_type = "post"
trunc_type = "post"

train_sequences = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
val_sequences = pad_sequences(val_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

print(f"Train sentences (vectorized): {train_sequences.shape}")
print(f"Validation sentences (vectorized): {val_sequences.shape}")


# In[186]:


import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, TextVectorization, Input


# In[187]:


embedding_dim = 200


# In[188]:


# define and compile model

model = tf.keras.Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    LSTM(16, activation="tanh"),
    Dense(6, activation="sigmoid"), # we are predicting for 6 classes
])

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)


# In[189]:


model.summary()


# In[190]:


epochs = 5 # if you want to improve performance, try increasing the number of training epochs
batch_size = 64
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3),
]


# In[191]:


model.evaluate(val_sequences, val_labels)


# In[192]:


model.fit(
    train_sequences,
    train_labels,
    validation_data=(val_sequences, val_labels),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=callbacks,
)


# In[193]:


hist = model.history.history

plt.figure(figsize=(20, 5))

plt.subplot(1, 2, 1)
plt.title("Loss")
plt.plot(hist["loss"], label="Training loss")
plt.plot(hist["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(range(len(hist["loss"])))
plt.legend(loc="upper right")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.title("Accuracy")
plt.plot(hist["accuracy"], label="Training accuracy")
plt.plot(hist["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.xticks(range(len(hist["accuracy"])))
plt.legend(loc="lower right")
plt.grid(True)

plt.show()


# In[195]:


def predict(sentence):
    # convert sentence to sequence
    sequences = tokenizer.texts_to_sequences([sentence])
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    
    # get predictions for toxicity
    predictions = model.predict(padded_sequences)[0]
    
    return list(zip(labels, predictions))


# In[196]:


predict("তুই আসলেই খারাপ ")


# In[ ]:





# In[200]:


def generate_response():
    input_sentence = input('Enter input news: ')
    Xi_token = tokenizer.texts_to_sequences([input_sentence])
    Xi_pad = pad_sequences(Xi_token, padding='post', maxlen=maxlen)
    print('Model predicts')
    preds = model.predict(Xi_pad)
    print('Confidence :')
    print(preds)
    preds = preds
    total = 0
    for k in range(len(preds[0])):
        print(encoder.inverse_transform([[k]]))
        print('%f %%' %(preds[0,k]*100))
        total += preds[0,k]*100
    #print(total)
    print('Predicted class: %s'%(encoder.inverse_transform(model.predict_classes(Xi_pad))))


# In[ ]:





# In[197]:


loss, accuracy = model.evaluate(val_sequences, val_labels)
print(f"Accuracy on validation sentences: {accuracy}")


# # RNN

# In[198]:


import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, LSTM, Dropout, Embedding, Dense, Bidirectional,Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Precision, CategoricalAccuracy, Recall
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# In[199]:


import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
import pandas as pd


# In[203]:


dataset = pd.read_csv(r'E:\AI\toxic comment prediction using nlp\dataset_final.csv')
dataset.head()


# In[204]:


plt.figure(figsize=(8,4))
sns.barplot(dataset)
plt.show()


# In[205]:


X = dataset['comment']
y = dataset[dataset.columns[1:]].values


# In[206]:


max_length = 200000


# In[207]:


vectorizer = TextVectorization(max_tokens = max_length, output_sequence_length = 1800, output_mode='int')
vectorizer.adapt(X.values)
vectorizer_text = vectorizer(X.values)


# In[208]:


vectorizer = TextVectorization(max_tokens = max_length, output_sequence_length = 1800, output_mode='int')
vectorizer.adapt(X.values)
vectorizer_text = vectorizer(X.values)


# In[209]:


#MCSHBAP - map, chache, shuffle, batch, prefetch  from_tensor_slices, list_file
data_set = tf.data.Dataset.from_tensor_slices((vectorizer_text, y))
data_set = data_set.cache()
data_set = data_set.shuffle(160000)
data_set = data_set.batch(16)
data_set = data_set.prefetch(8) # helps bottlenecks


# In[210]:


train = data_set.take(int(len(data_set)*.7))
val = data_set.skip(int(len(data_set)*.7)).take(int(len(data_set)*.2))
test = data_set.skip(int(len(data_set)*.9)).take(int(len(data_set)*.1))


# In[211]:


model = Sequential([
    Embedding(max_length+1, 32),
    Bidirectional(LSTM(32,activation = 'tanh')),
    Dense(128,activation='relu'),
    Dense(256,activation = 'relu'),
    Dense(128,activation = 'relu'),
    Dense(6,activation = 'sigmoid')
])
model.compile(loss = 'BinaryCrossentropy',optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
             metrics = ['accuracy'])
model.summary()


# In[212]:


history = model.fit(train, epochs = 1 ,validation_data = val)


# In[214]:


pre = Precision()
re = Recall()
acc = CategoricalAccuracy()


# In[215]:


for batch in test.as_numpy_iterator(): 
    # Unpack the batch 
    X_true, y_true = batch
    # Make a prediction 
    yhat = model.predict(X_true)
    
    # Flatten the predictions
    y_true = y_true.flatten()
    yhat = yhat.flatten()
    
    pre.update_state(y_true, yhat)
    re.update_state(y_true, yhat)
    acc.update_state(y_true, yhat)


# In[216]:


print(f'Precision: {pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')


# In[217]:


import numpy as np


# In[218]:


input_text = vectorizer('তুই আসলেই খারাপ')
res = model.predict(np.expand_dims(input_text,0))
(res > 0.5).astype(int)
print(dataset.columns[1:])
batch_X, batch_y = test.as_numpy_iterator().next()
(model.predict(batch_X) > 0.5).astype(int)
res.shape
print(res)


# In[ ]:




