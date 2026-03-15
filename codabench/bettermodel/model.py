import numpy as np
import pandas as pd
import re
YOUTUBE_STOPWORDS = {
    "video","subscribe","channel","watch","like","comment",
    "share","follow","instagram","twitter","facebook",
    "link","new","official"
}
class Model:
    

    def __init__(self):
        self.lr = 0.25
        self.epochs = 100
        self.batch_size = 30

        self.vocab = {}
        self.W = None
        self.b = None

        self.label_to_int = {}
        self.int_to_label = {}

    # ---------- text processing ----------
    def _tokenize(self, text):
        text = text.lower()
        text = re.findall(r'\b\w+\b', text)
        tokens = [w for w in text if not w.isdigit()]
        tokens = [w for w in tokens if w not in YOUTUBE_STOPWORDS]
        return tokens

    def _build_vocab(self, texts, min_freq=5):

        word_counts = {}
        doc_counts = {}

        for t in texts:
            words = set(self._tokenize(t))

            for w in words:
                doc_counts[w] = doc_counts.get(w, 0) + 1

            for w in self._tokenize(t):
                word_counts[w] = word_counts.get(w, 0) + 1

        idx = 0
        self.idf = {}

        N = len(texts)

        for w, df in doc_counts.items():

            if word_counts[w] >= min_freq and df <= 0.6 * N:

                self.vocab[w] = idx
                self.idf[idx] = np.log(N / (1 + df))
                idx += 1

    def _vectorize(self, texts):
        X = np.zeros((len(texts), len(self.vocab)))

        for i, t in enumerate(texts):

            words = self._tokenize(t)

            tf = {}

            for w in words:
                if w in self.vocab:
                    j = self.vocab[w]
                    tf[j] = tf.get(j, 0) + 1

            # term frequency normalization
            for j, count in tf.items():
                tf_val = count / len(words)
                X[i, j] = tf_val * self.idf[j]

        return X

    # ---------- softmax ----------
    def _softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        e = np.exp(z)
        return e / np.sum(e, axis=1, keepdims=True)

    # ---------- training ----------
    def fit(self, X_train, y_train):
        
        # testing different combinations
        # texts = (X_train["video_title"] + " " + X_train["video_description"]).values
        texts = X_train["video_title"].values
        #texts = X_train["video_description"].values
        
        # build tfidf
        self._build_vocab(texts)
        X = self._vectorize(texts)

        # encode labels
        labels = y_train['category'].unique()
        self.label_to_int = {l:i for i,l in enumerate(labels)}
        self.int_to_label = {i:l for l,i in self.label_to_int.items()}

        y = y_train['category'].map(self.label_to_int).values

        n_samples, n_features = X.shape
        n_classes = len(labels)

        self.W = np.random.randn(n_features, n_classes) * 0.01
        self.b = np.zeros(n_classes)

        Y = np.zeros((n_samples, n_classes))
        Y[np.arange(n_samples), y] = 1

        # mini-batch GD
        for _ in range(self.epochs):

            perm = np.random.permutation(n_samples)
            X = X[perm]
            Y = Y[perm]

            for start in range(0, n_samples, self.batch_size):

                Xb = X[start:start+self.batch_size]
                Yb = Y[start:start+self.batch_size]

                scores = Xb @ self.W + self.b
                probs = self._softmax(scores)

                m = Xb.shape[0]

                dW = (Xb.T @ (probs - Yb)) / m
                db = np.sum(probs - Yb, axis=0) / m

                self.W -= self.lr * dW
                self.b -= self.lr * db

    # ---------- prediction ----------
    def predict(self, X_test):

        texts = X_test.iloc[:,0].astype(str).values
        X = self._vectorize(texts)

        scores = X @ self.W + self.b
        probs = self._softmax(scores)

        ids = np.argmax(probs, axis=1)

        return np.array([self.int_to_label[i] for i in ids])