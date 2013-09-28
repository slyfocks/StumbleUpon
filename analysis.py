__author__ = 'slyfocks'
import numpy as np
from sklearn import metrics,preprocessing,cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model as lm
import pandas as pd


loadData = lambda f: np.genfromtxt(open(f, 'r'), delimiter=' ')


def main():
    print("loading data..")
    train_data = list(np.array(pd.read_table('train.tsv'))[:, 2])
    test_data = list(np.array(pd.read_table('test.tsv'))[:, 2])
    y = np.array(pd.read_table('train.tsv'))[:, -1]

    tfv = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode',
                          analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2),
                          use_idf=1, smooth_idf=1, sublinear_tf=1)

    rd = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001,
                               C=1, fit_intercept=True, intercept_scaling=1.0,
                               class_weight=None, random_state=None)

    X_all = train_data + test_data
    len_train = len(train_data)

    print("fitting pipeline")
    tfv.fit(X_all)
    print("transforming data")
    X_all = tfv.transform(X_all)

    X = X_all[:len_train]
    X_test = X_all[len_train:]

    print("20 Fold CV Score: " + str(np.mean(cross_validation.cross_val_score(rd, X, y, cv=20, scoring='roc_auc'))))
    print("training on full data")
    rd.fit(X,y)
    pred = rd.predict_proba(X_test)[:, 1]
    test_file = pd.read_csv('test.tsv', sep="\t", na_values=['?'], index_col=1)
    pred_df = pd.DataFrame(pred, index=test_file.index, columns=['label'])
    pred_df.to_csv('benchmark.csv')
    print("submission file created..")

if __name__ == "__main__":
    main()