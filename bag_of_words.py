import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import matplotlib.pyplot as plt
import pylab
import seaborn as sns

def review_to_wordlist(i, raw, remove_stopwords=False, join_with_space=False):
        ''' Clean one review with optional removing stop words, & join the output with space. '''

        if pd.isnull(raw):
            print i, ':', raw
        else:
            letters_only = re.sub('[^a-zA-Z]', ' ', raw)
            words = letters_only.lower().split()
            # Make a set of stopwords
            if remove_stopwords:
                stop_words = set(stopwords.words("english"))
                meaningful_words = [w for w in words if not w in stop_words]
            if join_with_space:
                return ' '.join(meaningful_words)
            else:
                return words


def get_all_cleaned(raw, remove_stopwords=False, join_with_space=False):
    ''' Loop through all reviews, and preprocess all reviews. '''

    num_reviews = raw.size
    clean_reviews = []
    print "Cleaning and parsing the data set movie reviews...\n"
    for i, raw_item in enumerate(raw):
        if i%5000 == 0:
            print 'Review %d of %d\n' %(i, num_reviews)
        clean_reviews.append(review_to_wordlist(i, raw_item, \
                                                    remove_stopwords, \
                                                    join_with_space))
    return clean_reviews

if __name__ == '__main__':

    print 'Step 1) Concatenating files...'
    files = []
    for i in range(12):
        if i < 10:
            files.append(pd.read_csv('Kickstarter00{}.csv'.format(i)))
        else:
            files.append(pd.read_csv('Kickstarter0{}.csv'.format(i)))
    data = pd.concat(files)
    success = data[data['state'] == 'successful']
    fail = data[data['state'] == 'failed']
    
    print 'Step 2.1) Cleaning success blurb...'
    success_blurbs = get_all_cleaned(success['blurb'], True, True)
    print 'Step 2.2) Cleaning fail blurb...'
    fail_blurbs = get_all_cleaned(fail['blurb'], True, True)
    
    success_blurbs = [x for x in success_blurbs if x is not None]
    fail_blurbs = [x for x in fail_blurbs if x is not None]
    
    print 'Step 3) Use tf-idf to create scaled bag of words model...'
    success_tf = TfidfVectorizer(min_df=2, \
                                max_df=0.75, \
                                max_features=500, \
                                ngram_range=(1, 4), \
                                sublinear_tf=True)
    success_tf_bow = success_tf.fit_transform(success_blurbs)

    fail_tf = TfidfVectorizer(min_df=2, \
                                max_df=0.75, \
                                max_features=500, \
                                ngram_range=(1, 4), \
                                sublinear_tf=False)
    fail_tf_bow = fail_tf.fit_transform(fail_blurbs)
    
    print 'Step 4.1) Converting Scipy sparse matrix to Pandas dataframe...'
    # For fail.
    df_tf_fail = pd.DataFrame([ pd.Series(fail_tf_bow[i].toarray().ravel()) \
                              for i in np.arange(fail_tf_bow.shape[0]) ])
    # For success.
    df_tf_success = pd.DataFrame([ pd.Series(success_tf_bow[i].toarray().ravel()) \
                                  for i in np.arange(success_tf_bow.shape[0]) ])

    print 'Step 4.2) Sort the words in descending order, from most frequent to least.'
    top_ten_tf_fail = df_tf_fail.sum(axis=0).sort_values(ascending=False)
    top_ten_tf_success = df_tf_success.sum(axis=0).sort_values(ascending=False)
    # Reverse dictionary for Fail, and print top 10 words.
    new_tf_fail = {v: k for k, v in fail_tf.vocabulary_.iteritems()}

    # Reverse dictionary for Success, and print top 10 words.
    new_tf_success = {v: k for k, v in success_tf.vocabulary_.iteritems()}

    print 'Step 5) Print the top 20 most frequent words in successful/failed campaigns'
    print '\nTop 20 words for failed campaigns: '
    fail_words = []
    for index in top_ten_tf_fail.index[0:20]:
        print new_tf_fail[index]
        fail_words.append(new_tf_fail[index])
    for value in top_ten_tf_fail.head(20):
        print value

    print '\nTop 20 words for successful campaigns: '
    success_words = []
    for index in top_ten_tf_success.index[0:20]:
        print new_tf_success[index]
        success_words.append(new_tf_success[index])
    for value in top_ten_tf_success.head(20):
        print value
