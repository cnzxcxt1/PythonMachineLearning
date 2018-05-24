# Chapter 8 - Applying Machine Learning To Sentiment Analysis
import pandas as pd
#
df = pd.read_csv('movie_data.csv', encoding='utf-8')
df.head(3)
#

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
#
count = CountVectorizer()
docs = np.array([
    'The sun is shining',
    'The weather is sweet',
    'The sun is shining, the weather is sweet, and one and one is two'])
bag = count.fit_transform(docs)
# Now let us print the contents of the vocabulary to get a better understanding of the underlying concepts:

print(count.vocabulary_)
#
print(bag.toarray())
#
## Assessing word relevancy via term frequency-inverse document frequency
#
np.set_printoptions(precision=2)
#
from sklearn.feature_extraction.text import TfidfTransformer
#
tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())

#
tf_is = 3
n_docs = 3
idf_is = np.log((n_docs + 1) / (3 + 1))
tfidf_is = tf_is * (idf_is + 1)
print('tf-idf of term "is" = %.2f' % tfidf_is)
#

tfidf = TfidfTransformer(use_idf=True, norm=None, smooth_idf=True)
raw_tfidf = tfidf.fit_transform(count.fit_transform(docs)).toarray()[-1]
raw_tfidf
#
l2_tfidf = raw_tfidf / np.sqrt(np.sum(raw_tfidf ** 2))
l2_tfidf
#
## Cleaning text data
#
df.loc[0, 'review'][-50:]
#
import re
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    return text
#
preprocessor(df.loc[0, 'review'][-50:])
#
#
preprocessor("</a>This :) is :( a test :-)!")
#
df['review'] = df['review'].apply(preprocessor)
#
## Processing documents into tokens
#
from nltk.stem.porter import PorterStemmer
#
porter = PorterStemmer()
#
#
def tokenizer(text):
    return text.split()
#
#
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
#
tokenizer('runners like running and thus they run')
#
tokenizer_porter('runners like running and thus they run')
#
import nltk
#
nltk.download('stopwords')
#
from nltk.corpus import stopwords
stop = stopwords.words('english')
[w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:] if w not in stop]
#
# Training a logistic regression model for document classification
#
#
X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values
#
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]}
              ]
#
lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(random_state=0))])

gs_lr_tfidf = GridSearchCV(lr_tfidf,
                           param_grid, scoring='accuracy',
                           cv=5, verbose=1,
                           n_jobs=3)

gs_lr_tfidf.fit(X_train, y_train)

#
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
#
tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)
param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf': [False],
               'vect__norm': [None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              ]
lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(random_state=0))])
gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=1,
                           n_jobs=-1)
gs_lr_tfidf.fit(X_train, y_train)
import os
if 'TRAVIS' in os.environ:
    gs_lr_tfidf.verbose = 2
    X_train = df.loc[:250, 'review'].values
    y_train = df.loc[:250, 'sentiment'].values
    X_test = df.loc[25000:25250, 'review'].values
    y_test = df.loc[25000:25250, 'sentiment'].values

gs_lr_tfidf.fit(X_train, y_train)
#
# In[27]:
#
#
print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)

# In[28]:

clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, y_test))


from sklearn.linear_model import LogisticRegression
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

np.random.seed(0)
np.set_printoptions(precision=6)
y = [np.random.randint(3) for i in range(25)]
X = (y + np.random.randn(25)).reshape(-1, 1)
#
cv5_idx = list(StratifiedKFold(n_splits=5, shuffle=False, random_state=0).split(X, y))

cross_val_score(LogisticRegression(random_state=123), X, y, cv=cv5_idx)
#
# # By executing the code above, we created a simple data set of random integers that shall represent our class labels. Next, we fed the indices of 5 cross-validation folds (`cv3_idx`) to the `cross_val_score` scorer, which returned 5 accuracy scores -- these are the 5 accuracy values for the 5 test folds.
# Next, let us use the `GridSearchCV` object and feed it the same 5 cross-validation sets (via the pre-generated `cv3_idx` indices):
#
# # In[30]:

from sklearn.model_selection import GridSearchCV

gs = GridSearchCV(LogisticRegression(), {}, cv=cv5_idx, verbose=3).fit(X, y)

# # As we can see, the scores for the 5 folds are exactly the same as the ones from `cross_val_score` earlier.
#
# # Now, the best_score_ attribute of the `GridSearchCV` object, which becomes available after `fit`ting, returns the average accuracy score of the best model:
#
# # In[31]:
#
#
gs.best_score_

# # As we can see, the result above is consistent with the average score computed the `cross_val_score`.

# In[32]:

cross_val_score(LogisticRegression(), X, y, cv=cv5_idx).mean()

#
# # # Working with bigger data - online algorithms and out-of-core learning
#
# # In[27]:
#
#
import numpy as np
import re
from nltk.corpus import stopwords


def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized


def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)  # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label

# In[28]:

next(stream_docs(path='movie_data.csv'))


# # In[29]:
#
#
def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y


# # In[30]:
#
#
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

vect = HashingVectorizer(decode_error='ignore',
                         n_features=2 ** 21,
                         preprocessor=None,
                         tokenizer=tokenizer)

clf = SGDClassifier(loss='log', random_state=1, n_iter=1)
doc_stream = stream_docs(path='movie_data.csv')

# # **Note**
# # - You can replace `Perceptron(n_iter, ...)` by `Perceptron(max_iter, ...)` in scikit-learn >= 0.19. The `n_iter` parameter is used here deriberately, because some people still use scikit-learn 0.18.
# #
# # In[31]:


import pyprind

pbar = pyprind.ProgBar(45)
classes = np.array([0, 1])
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()

# In[32]:


X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print('Accuracy: %.3f' % clf.score(X_test, y_test))

# # In[33]:


clf = clf.partial_fit(X_test, y_test)
# # ## Topic modeling

# # ### Decomposing text documents with Latent Dirichlet Allocation

# # ### Latent Dirichlet Allocation with scikit-learn
#
# # In[1]:
#
#
import pandas as pd
df = pd.read_csv('movie_data.csv', encoding='utf-8')
df.head(3)
#
# # In[ ]:
#
#
# ## @Readers: PLEASE IGNORE THIS CELL
# ##
# ## This cell is meant to create a smaller dataset if
# ## the notebook is run on the Travis Continuous Integration
# ## platform to test the code on a smaller dataset
# ## to prevent timeout errors and just serves a debugging tool
# ## for this notebook
#

if 'TRAVIS' in os.environ:
    df.loc[:500].to_csv('movie_data.csv')
    df = pd.read_csv('movie_data.csv', nrows=500)
    print('SMALL DATA SUBSET CREATED FOR TESTING')

# # In[2]:


from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english',
                        max_df=.1,
                        max_features=5000)
X = count.fit_transform(df['review'].values)
# # In[3]:


from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_topics=10,
                                random_state=123,
                                learning_method='batch')
X_topics = lda.fit_transform(X)

# # In[4]:


lda.components_.shape

# # In[5]:


n_top_words = 5
feature_names = count.get_feature_names()

for topic_idx, topic in enumerate(lda.components_):
    print("Topic %d:" % (topic_idx + 1))
    print(" ".join([feature_names[i]
                    for i in topic.argsort() \
                        [:-n_top_words - 1:-1]]))

# # Based on reading the 5 most important words for each topic, we may guess that the LDA identified the following topics:
# #
# # 1. Generally bad movies (not really a topic category)
# # 2. Movies about families
# # 3. War movies
# # 4. Art movies
# # 5. Crime movies
# # 6. Horror movies
# # 7. Comedies
# # 8. Movies somehow related to TV shows
# # 9. Movies based on books
# # 10. Action movies
#
# # To confirm that the categories make sense based on the reviews, let's plot 5 movies from the horror movie category (category 6 at index position 5):
#
# # In[6]:


horror = X_topics[:, 5].argsort()[::-1]

for iter_idx, movie_idx in enumerate(horror[:3]):
    print('\nHorror movie #%d:' % (iter_idx + 1))
    print(df['review'][movie_idx][:300], '...')

# Using the preceeding code example, we printed the first 300 characters from the top 3 horror movies and indeed, we can see that the reviews -- even though we don't know which exact movie they belong to -- sound like reviews of horror movies, indeed. (However, one might argue that movie #2 could also belong to topic category 1.)
#
#
# # # Summary