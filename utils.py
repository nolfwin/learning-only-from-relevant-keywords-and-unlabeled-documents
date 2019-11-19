import torch
import numpy as np
import io
import pandas as pd
import random
import nltk
from torchtext import data
from torchtext.vocab import GloVe
from sklearn.datasets import fetch_20newsgroups
from config import dataset_dir
from gensim.models import KeyedVectors # load the Stanford GloVe model

print('BUILD GLOVE MODEL')
filename = dataset_dir+'/Glove/glove.6B.50d.txt.word2vec'
glove_model = KeyedVectors.load_word2vec_format(filename, binary=False)
all_datasets = ['20_newsgroups', 'ayi', 'custrev', 'mpqa', 'subj']


def get_extended_keyword(key, vocab=None, weight=3, n=5):
    keyword = ""
    for k in key:
        keyword += (k+' ')*weight
        if n > 0:
            for x in glove_model.similar_by_word(k, topn=n, restrict_vocab=vocab):
                keyword += x[0]+' '
    return keyword


def get_dataset(dataset_name, key, threshold=0.8, pseudo=True, flip=False,):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    doc_u, lbl, X_test, Y_test = get_nlp_datasets(dataset_name)
    if flip:
        lbl = -lbl
        Y_test = -Y_test
    if pseudo:
        doc = [key]
        doc.extend(doc_u)
        tfidf = TfidfVectorizer(stop_words='english')
        doc_tfidf = tfidf.fit_transform(doc)
        sim_score = linear_kernel(doc_tfidf[0], doc_tfidf[1:])[0]
        ind, = np.where(np.array(sim_score) > 0.0)
        doc_u = np.array(doc_u)
        sim_score = list(enumerate(sim_score))
        sim_scores = sorted(sim_score, key=lambda x: x[1], reverse=True)
        kk = int(len(ind)*threshold)
        sim_scores = sim_scores[:kk]
        pos = np.array([i[0] for i in sim_scores])
        ind_neg, = np.where(np.isin(np.arange(len(sim_score)), pos, invert=True))
        lp = lbl[pos]
        ln = lbl[ind_neg]
        ind = pos
        id = np.concatenate((ind, ind_neg))
        X_train = doc_u[id]
        sim = np.array([sim_score[i][1] for i in id])
    else:
        doc_u = np.array(doc_u)
        ind, = np.where(lbl==1)
        ind_neg, = np.where(lbl==-1)
        id = np.concatenate((ind, ind_neg))
        X_train = doc_u[id]
        lp = lbl[ind]
        ln = lbl[ind_neg]
        sim = 0
    return X_train, X_test, Y_test, lp, ln, sim


def get_nlp_datasets(dataset_name):
    if dataset_name == '20_newsgroups':
        all_categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
                          'comp.sys.mac.hardware',
                          'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball',
                          'rec.sport.hockey', 'sci.crypt',
                          'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns',
                          'talk.politics.mideast', 'talk.politics.misc',
                          'talk.religion.misc']

        sub_cat = all_categories
        fetch_data = fetch_20newsgroups(data_home=dataset_dir, categories=sub_cat, subset='all',
                                        remove=('headers', 'footers'))
        data_dict = dict()
        sub_cat = list(fetch_data.target_names)
        for i, cat in enumerate(sub_cat):
            data_dict[cat] = i

        pos_categories = ['rec.sport.baseball', 'rec.sport.hockey']
        pos_indx = []
        for pos_cat in pos_categories:
            pos_indx.append(data_dict[pos_cat])
        pos_indx = np.array(pos_indx)

        features = fetch_data.data
        labels = fetch_data.target
        labels = np.isin(labels, pos_indx).astype(int)
        labels = 2 * labels - 1

        n_p = np.count_nonzero(labels + 1)
        n_n = np.count_nonzero(labels - 1)
        x_p = []
        x_n = []
        for i, news in enumerate(features):
            if labels[i] == 1:
                x_p.append(news)
            else:
                x_n.append(news)
        random.shuffle(x_p), random.shuffle(x_n)
        train_xp, test_xp = np.split(x_p, [int(n_p * 0.8)])
        train_xn, test_xn = np.split(x_n, [int(n_n * 0.8)])
        del fetch_data, features, labels
        train_features = np.concatenate((train_xp, train_xn)).tolist()
        train_labels = np.concatenate((np.ones(len(train_xp)), -np.ones(len(train_xn))))
        del train_xp, train_xn
        test_features = np.concatenate((test_xp, test_xn)).tolist()
        test_labels = np.concatenate((np.ones(len(test_xp)), -np.ones(len(test_xn))))
        del test_xp, test_xn

    elif dataset_name == 'ayi':
        filepath_dict = {'yelp': '/sentiment_analysis/yelp_labelled.txt',
                    'amazon': '/sentiment_analysis/amazon_cells_labelled.txt',
                    'imdb':   '/sentiment_analysis/imdb_labelled.txt'}
        df_list = []
        for source, filepath in filepath_dict.items():
            df = pd.read_csv(dataset_dir+filepath, names=['sentences', 'labels'], sep='\t')
            df['source'] = source  # Add another column filled with the source name
            df_list.append(df)

        df = pd.concat(df_list)
        label = 2*df['labels'].values-1
        sentences = df['sentences'].values

        features = sentences
        n_p = np.count_nonzero(label+1)
        n_n = np.count_nonzero(label-1)
        x_p = features[label == 1]
        x_n = features[label == -1]
        perm_p = np.random.permutation(n_p)
        perm_n = np.random.permutation(n_n)
        x_p, x_n = x_p[perm_p], x_n[perm_n]
        train_xp, test_xp = np.split(x_p, [int(n_p * 0.8)])
        train_xn, test_xn = np.split(x_n, [int(n_n * 0.8)])
        del features, label
        train_features = np.concatenate((train_xp, train_xn)).tolist()
        train_labels = np.concatenate((np.ones(len(train_xp)), -np.ones(len(train_xn))))
        del train_xp, train_xn
        test_features = np.concatenate((test_xp, test_xn)).tolist()
        test_labels = np.concatenate((np.ones(len(test_xp)), -np.ones(len(test_xn))))

    elif dataset_name in ['custrev', 'mpqa', 'subj']:
        labels = []
        features = []
        path = dataset_dir + '/harvard-nlp/' + dataset_name + '.all'
        with io.open(path, encoding='utf-8', errors='ignore') as f:
            for i, l in enumerate(f):
                if not len(l.strip()) >= 3:
                    continue
                label, text = l.strip().split(None, 1)
                labels.append(int(label))
                features.append(text)

        features = np.array(features)
        label = np.array(labels) * 2 - 1
        n_p = np.count_nonzero(label+1)
        n_n = np.count_nonzero(label-1)
        x_p = features[label == 1]
        x_n = features[label == -1]
        perm_p = np.random.permutation(n_p)
        perm_n = np.random.permutation(n_n)
        x_p, x_n = x_p[perm_p], x_n[perm_n]
        train_xp, test_xp = np.split(x_p, [int(n_p * 0.8)])
        train_xn, test_xn = np.split(x_n, [int(n_n * 0.8)])
        del features, labels
        train_features = np.concatenate((train_xp, train_xn)).tolist()
        train_labels = np.concatenate((np.ones(len(train_xp)), -np.ones(len(train_xn))))
        del train_xp, train_xn
        test_features = np.concatenate((test_xp, test_xn)).tolist()
        test_labels = np.concatenate((np.ones(len(test_xp)), -np.ones(len(test_xn))))
        del test_xp, test_xn

    return train_features, train_labels, test_features, test_labels


def CustomLoader(doc, lbl, n_tr, fix_length=200, batch_size=32, device=-1, embeds=50):
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    tokenize = lambda x: x.split()
    TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, batch_first=True,
                      fix_length=fix_length, stop_words=stop_words)
    LABEL = data.LabelField(dtype=torch.float, use_vocab=False)
    train_ex = []
    test_ex = []
    for i in range(len(doc)):
        if i < n_tr :
            train_ex += [data.Example.fromlist([doc[i], lbl[i]], [('text', TEXT), ('label', LABEL)])]
        else:
            test_ex += [data.Example.fromlist([doc[i], lbl[i]], [('text', TEXT), ('label', LABEL)])]
    train_dataset = data.Dataset(train_ex, [('text', TEXT), ('label', LABEL)])
    test_dataset = data.Dataset(test_ex, [('text', TEXT), ('label', LABEL)])
    dataset = data.Dataset(train_ex + test_ex, [('text', TEXT), ('label', LABEL)])

    TEXT.build_vocab(dataset, vectors=GloVe(name='6B', dim=embeds))
    word_embeddings = TEXT.vocab.vectors
    vocab_size = len(TEXT.vocab)
    train_loader, test_loader = data.BucketIterator.splits((train_dataset, test_dataset), batch_size=batch_size,
                                                           sort_key=lambda x: len(x.text),device=device, repeat=False, shuffle=True)
    return TEXT, vocab_size, word_embeddings, train_loader, test_loader


def extract_glove_feature(text):
    tokens = nltk.word_tokenize(text.lower())
    words = [word for word in tokens if word.isalpha()]
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w in glove_model]
    if len(words) == 0:
        words.append('none')
    feature = np.mean(glove_model[words], axis=0)
    #feature = glove_model[words]
    return feature
