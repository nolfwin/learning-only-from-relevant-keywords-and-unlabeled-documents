from utils import extract_glove_feature
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from nltk.classify.naivebayes import NaiveBayesClassifier
import numpy as np


def get_baseline_method(x_train, y_train, x_test, y_test, method=None, keywords=None):
    def transform_features(sentence):
        words = sentence.lower().split()
        return dict(('contains(%s)' % w, True) for w in words)

    if 'nb' in method:
        x_train = list(map(transform_features, x_train))
        x_test = list(map(transform_features, x_test))
        train_set = list(zip(x_train, y_train))
        clf = NaiveBayesClassifier.train(train_set)
        score_test = np.array([clf.prob_classify(t).prob(1.0) for t in x_test])
        score_train = np.array([clf.prob_classify(t).prob(1.0) for t in x_train])
    else:
        x_train = [extract_glove_feature(text) for text in x_train]
        x_test = [extract_glove_feature(text) for text in x_test]
        if 'randomforest' in method:
            clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1).fit(x_train, y_train)
            score_train = clf.predict_proba(x_train)[:, 1]
            score_test = clf.predict_proba(x_test)[:, 1]
        elif 'knn' in method:
            clf = KNeighborsClassifier(10).fit(x_train, y_train)
            score_train = clf.predict_proba(x_train)[:, 1]
            score_test = clf.predict_proba(x_test)[:, 1]
        elif 'gloverank' in method:
            from sklearn.metrics.pairwise import cosine_similarity
            keyword_doc = extract_glove_feature(keywords).reshape(1, 50)
            score_train = cosine_similarity(keyword_doc, x_train)[0]
            score_test = cosine_similarity(keyword_doc, x_test)[0]

    return score_test, y_test, score_train, y_train