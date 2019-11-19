Learning Only from Relevant Keywords and Unlabeled Documents (EMNLP-IJCNLP 2019)
=
### Prerequisites
- Python 3.6
- Pytorch 0.4.1 (https://pytorch.org)
- torchtext 0.3.1
    > pip install torchtext
- gensim 3.7.1
    > pip install -U gensim
- The Natural Language Toolkit (NLTK) (http://nltk.org/)
- numpy 1.15.1
- sklearn
- GloVe word2vec text file in ./data/Glove/glove.6B.50d.txt.word2vec

### Setup
Unfortunately since the software size is limited by 10MB in the submission. We can't include the file ./data/Glove/glove.6B.50d.txt.word2vec
Please download glove.6B.50d.txt from the GloVe website and please run the following command:
>
    from gensim.test.utils import datapath, get_tmpfile
    from gensim.models import KeyedVectors
    from gensim.scripts.glove2word2vec import glove2word2vec
    glove_file = datapath('./data/Glove/glove.6B.50d.txt')
    tmp_file = get_tmpfile("./data/Glove/glove.6B.50d.txt.word2vec")
    _ = glove2word2vec(glove_file, tmp_file)
    model = KeyedVectors.load_word2vec_format(tmp_file)

All credits on how to make word2vec file go to https://radimrehurek.com/gensim/scripts/glove2word2vec.html

### Run
> main.py --method sigmoid --dataset mpqa --mode pseudo --weight 3 --extension 5 --threshold 0.9

Note that if there is no keyword, then given keywords (which are same keywords set in the main body) will be used.
### Arguments
* --method : loss function for proposed framework. ['sigmoid', 'log'] or baselines ['nb', 'randomforest', 'knn', 'gloverank']<br>
    - sigmoid : symmetric loss function (default)
    - log (logistic) : non-symmetric loss function
    - nb : NaiveBayes classifier. based on NLTK
    - randomforest : Random Forest based on Glove Feature
    - knn : K-nearest Neighbors (KNN) based on Glove Featrue (k=10)
    - gloverank : GloVe Ranking which is zero-shot baseline.
* --dataset : mpqa(default), ayi, subj, custrev, and 20_newsgroups.
* --mode : pseudo (default) or clean(oracle mode)
* --weight : weight for original keywords. For subj, use 1. For others, use 3 (default).
* --extension : the number of similar words to original keyword. For subj, used 50. For others, use 5 (default).
* --threshold : threshold for pseudo-labeling algorithm, default value is 0.9.
* --keywords : For each dataset, default keywords are given. If some keywords are given, then they will be used in Algorithm 1.

### Note
In config.py, default data location was set ./data. <br>
You might need to change it to run in your own computer. <br>
dataset_dir + /Glove/glove.6B.50d.txt.word2vec are necessary (https://radimrehurek.com/gensim/scripts/glove2word2vec.html).

### Data
We do not create any new datasets and all credits go to the following:
- Harvard-nlp: https://github.com/CS287/HW1/tree/master/data
- Sentiment-analysis: https://github.com/hallr/DAT_SF_19/tree/master/data


Nontawat Charoenphakdee, Jongyeong Lee, Yiping Jin, Dittaya Wanvarie, Masashi Sugiyama. "Learning Only from Relevant Keywords and Unlabeled Documents." Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). 2019.

Paper: [link](https://www.aclweb.org/anthology/D19-1411/)
