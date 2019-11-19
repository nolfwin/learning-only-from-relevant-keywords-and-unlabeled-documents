import argparse

import torch
from pylab import *
from sklearn import metrics

from model import PU_DEEP
from utils import CustomLoader, get_dataset, get_extended_keyword


def restricted_float(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]" % (x,))
    return x


parser = argparse.ArgumentParser(
    description="Text classification based on keyword pseudo-algorithm"
)
parser.add_argument(
    "--dataset",
    type=str,
    choices=["subj", "custrev", "mpqa", "ayi", "20_newsgroups"],
    default="mpqa",
)
parser.add_argument(
    "--keywords",
    type=str,
    nargs="*",
    default=None,
    help="keywords for pesudo-labeling. If input is blank, it will be default values which we set already",
)
parser.add_argument(
    "--mode",
    type=str,
    choices=["clean", "pseudo"],
    default="pseudo",
    help="use corrupted pseudo-labels or use clean labels",
)
parser.add_argument(
    "--weight", type=int, default=3, help="weight for original keywords"
)
parser.add_argument(
    "--extension",
    type=int,
    default=5,
    help="the number of extended keywords which are similar to original one",
)
parser.add_argument(
    "--threshold",
    type=restricted_float,
    default=0.9,
    help="threshold value for pseudo-labeled positive data. If you want to use all value set it as 1.0",
)
parser.add_argument(
    "--method",
    type=str,
    choices=["sigmoid", "log", "nb", "randomforest", "knn", " gloverank"],
    default="sigmoid",
    help="sigmoid is a symmetric loss, log (logistic) is a non-symmetric loss, and baselines (nb, knn, randomforest, gloverank)",
)

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_threshold(score, ratio=0.1):
    n = int(len(score) * ratio)
    sc = sorted(score, reverse=True)
    return sc[n]


def precision_at_k(y_true, y_score, k=10):
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    unique_y = np.unique(y_true)

    if len(unique_y) > 2:
        raise ValueError("Only support two relevance levels.")

    pos_label = unique_y[1]
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    n_relevant = len(y_true[y_true == pos_label])

    return float(n_relevant) / min(k, len(y_true))


if __name__ == "__main__":
    np.random.seed(1)

    ######## Parameters ########
    embedding_dim = 50
    batch_size = 128
    hidden_size = 64
    num_layers = 2
    epoch = 50
    if args.dataset in ["ayi", "custrev"]:
        # use big learning rate for small dataset
        lr = 1e-4
    else:
        lr = 1e-5
    #############################

    proposed_list = ["sigmoid", "log"]
    print("Dataset: %s" % args.dataset)

    if args.keywords is None:
        if args.dataset == "20_newsgroups":
            keywords = ["sports", "baseball", "hockey"]
        elif args.dataset == "subj":
            keywords = [
                "wonderful",
                "terrible",
                "feel",
                "happy",
                "ugly",
                "even",
                "horrible",
                "interesting",
                "funny",
                "dramatic",
                "romantic",
                "compassionate",
            ]
        elif args.dataset == "ayi":
            keywords = [
                "great",
                "best",
                "excellent",
                "friendly",
                "awesome",
                "nice",
                "amazing",
            ]
        elif args.dataset == "mpqa":
            keywords = ["support", "hope", "help", "good", "great", "love"]
        elif args.dataset == "custrev":
            keywords = [
                "easy",
                "excellent",
                "nice",
                "great",
                "good",
                "love",
                "amazing",
                "best",
                "awesome",
                "perfect",
                "definitely",
                "better",
                "happy",
            ]
    else:
        keywords = args.keywords

    pseudo = args.mode == "pseudo"
    flip = args.dataset == "subj"
    keywords = get_extended_keyword(
        keywords, vocab=None, weight=args.weight, n=args.extension
    )
    if pseudo:
        if args.extension > 0:
            print("Extended Keywords :", keywords)
        else:
            print("Original Keywords :", keywords)
    else:
        print("ORACLE MODE")

    x_train, x_test, y_test, lp, ln, sc = get_dataset(
        args.dataset, key=keywords, threshold=args.threshold, pseudo=pseudo, flip=flip
    )
    # check pseudo-labeling method
    tp = len(np.where(lp == 1)[0])
    fp = len(np.where(lp == -1)[0])
    tn = len(np.where(ln == -1)[0])
    fn = len(np.where(ln == 1)[0])
    print("--------------------------------------------")
    print("Result of Pseudo-labeling Algorithm 1")
    print("Pseudo-positive (true-p, false-p) = (%d, %d)" % (tp, fp))
    print("Pseudo-negative (false-n, true-n) = (%d, %d)" % (fn, tn))
    print("--------------------------------------------")
    pi = float(tp / len(lp))
    pi_p = float(fn / len(ln))
    print(
        "Ratio of true positive data in pseudo-labeled data (pi, pi_prime) = (%.2f, %.2f)"
        % (pi, pi_p)
    )
    true_pos = len(np.where(lp == 1)[0]) + len(np.where(ln == 1)[0])
    true_neg = len(np.where(lp == -1)[0]) + len(np.where(ln == -1)[0])

    y_train = np.concatenate((np.ones(len(lp)), -np.ones(len(ln))))
    n_train = len(y_train)
    data = np.concatenate((x_train, x_test))
    label = np.concatenate((y_train, y_test))

    TEXT, vocab_dim, word_embeddings, train_loader, test_loader = CustomLoader(
        doc=data,
        lbl=label,
        n_tr=n_train,
        batch_size=batch_size,
        device=device,
        embeds=embedding_dim,
    )

    if args.method in proposed_list:
        score_test, y_test, score_train, y_train = PU_DEEP(
            train_loader,
            test_loader,
            vocab_dim=vocab_dim,
            weights=word_embeddings,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            epoch=epoch,
            stepsize=lr,
            hidden_size=hidden_size,
            device=device,
            loss=args.method,
        )
    else:
        from baselines import get_baseline_method

        score_test, y_test, score_train, y_train = get_baseline_method(
            x_train, y_train, x_test, y_test, method=args.method, keywords=keywords
        )
    threshold = get_threshold(score_train, float(true_pos / (true_pos + true_neg)))
    # if the value is exactly zero, we take it as positive.
    prediction = np.where(
        np.sign(score_test - threshold) == 0, 1, np.sign(score_test - threshold)
    )

    score = 100 * metrics.roc_auc_score(y_test, score_test)
    precn = 100 * precision_at_k(y_test, score_test, k=100)
    f1 = 100 * metrics.f1_score(y_test, prediction, pos_label=1, average="macro")
    acc = 100 * metrics.accuracy_score(y_test, prediction)

    # for memory
    del (
        x_train,
        x_test,
        y_test,
        lp,
        ln,
        sc,
        data,
        label,
        TEXT,
        vocab_dim,
        word_embeddings,
        train_loader,
        test_loader,
    )
    print("RESULT for %s - %s loss  " % (args.dataset, args.method))
    print("AUC score : %2.1f" % score)
    print("F1 measure: %2.1f" % f1)
    print("Accuracy  : %2.1f" % acc)
    print("Prec@100 : %2.1f" % precn)
