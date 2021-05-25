import os
from time import time
import numpy as np
from scipy import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from lightgbm import LGBMClassifier
import pandas as pd
import pdb
from multiprocessing import Pool, cpu_count


print("CPU count:", cpu_count())

nrows = None
train_df = pd.read_csv('corpus/news/train.csv', sep='\t', nrows=nrows)
val_df = pd.read_csv('corpus/news/val.csv', sep='\t', nrows=None)
test_df = pd.read_csv('corpus/news/test_a.csv', sep='\t', nrows=nrows)
print(len(train_df), len(val_df), len(test_df))



print("TfidfVectorizer...")
use_cache = False
cache_files = ['save/{}.npy'.format(split) for split in ['train', 'val', 'test']]
params = dict(
    max_features=10000,
    max_df=0.5,
    min_df=0.001,
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1,3))
since = time()
if os.path.exists(cache_files[0]) and use_cache:
    print('load cached matrix...')
    # train_tfidf = io.mmread("train_tfidf.mtx")
    # val_tfidf = io.mmread("val_tfidf.mtx")
    # test_tfidf = io.mmread("test_tfidf.mtx")
    train_tfidf = np.load(cache_files[0])
    val_tfidf = np.load(cache_files[1])
    test_tfidf = np.load(cache_files[2])
else:
    all_df = pd.concat([train_df, val_df, test_df], sort=False)
    print(len(all_df))
    tfidf = TfidfVectorizer(**params).fit(all_df['text'].iloc[:].values)
    train_tfidf = tfidf.transform(train_df['text'].iloc[:].values).toarray()
    val_tfidf = tfidf.transform(val_df['text'].iloc[:].values).toarray()
    test_tfidf = tfidf.transform(test_df['text'].iloc[:].values).toarray()
    # io.mmwrite("train_tfidf.mtx", train_tfidf)
    # io.mmwrite("val_tfidf.mtx", val_tfidf)
    # io.mmwrite("test_tfidf.mtx", test_tfidf)
    np.save(cache_files[0], train_tfidf)
    np.save(cache_files[1], val_tfidf)
    np.save(cache_files[2], test_tfidf)
print('time:', time()-since)



# clf = RidgeClassifier()
# clf = LogisticRegression()
clf = LGBMClassifier(n_jobs=16, feature_fraction=0.7, bagging_fraction=0.4, lambda_l1=0.001, lambda_l2=0.01, n_estimators=600)
# clf = RandomForestClassifier(n_estimators=100, max_depth=None, max_features='auto',
#                              min_samples_split=2, min_samples_leaf = 1, verbose=1, n_jobs=16)

print("training...")
since = time()
clf.fit(train_tfidf, train_df['label'].iloc[:].values)
print('time:', time()-since)


print("eval...")
preds = clf.predict(val_tfidf)
labels = val_df['label'].values
equ = preds == labels
acc = np.sum(equ) / len(equ)
fscore = metrics.f1_score(labels, preds, average='macro')
print('acc:', acc, 'f1:', fscore)


print("inference...")
df = pd.DataFrame()
df['label'] = clf.predict(test_tfidf)
df.to_csv('save/submission.csv', index=None)