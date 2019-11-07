import time
import numpy as np
from numpy.linalg import norm

class NMF(object):
    def __init__(self, n_components, max_iter, tol, cost_function = "euclidean", fix_seed=False):

        if fix_seed: 
            np.random.seed(0)

        self.V = np.array([])
        self.W = np.array([])
        self.H = np.array([])
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.cost_function = cost_function

    def _update_euclidean(self):
        epss = 1e-20
        loss_old = None
        itr = 0
        start_time = time.time()
        while itr < self.max_iter:
            # update H
            WTV = np.dot(self.W.T, self.V)
            WTWH = np.dot(np.dot(self.W.T, self.W), self.H) + epss
            self.H = np.multiply(self.H, np.divide(WTV, WTWH))

            # update W
            VHT = np.dot(self.V, self.H.T)
            WHHT = np.dot(np.dot(self.W, self.H), self.H.T) + epss
            self.W = np.multiply(self.W, np.divide(VHT, WHHT))

            # calculate loss
            # loss_new = np.subtract(self.V, np.dot(self.W, self.H))
            # loss_new = np.sum(loss_new*loss_new)
            loss_new = norm(self.V - np.dot(self.W, self.H), 'fro')**2/2.0
            end_time = time.time()
            print('Step={}, Loss={}, Time={}s'.format(itr, loss_new, end_time-start_time))
            itr += 1

            # check terminate condition
            if loss_old == None:
                loss_old = loss_new
                continue
            elif loss_old - loss_new <= self.tol:
                break
            else:
                loss_old = loss_new

    def _update_kl(self):
        epss = 1e-20
        loss_old = None
        itr = 0
        start_time = time.time()
        while itr < self.max_iter:
            # update H
            WH = np.dot(self.W, self.H) + epss
            VovWH = np.divide(self.V, WH)
            WVovWH = np.dot(self.W.T, VovWH)
            Wka = np.sum(self.W, axis=0) + epss
            self.H = np.multiply(self.H, np.divide(WVovWH.T, Wka).T)

            # update W
            WH = np.dot(self.W, self.H) + epss
            VovWH = np.divide(self.V, WH)
            HVovWH = np.dot(self.H, VovWH.T)
            Hav = np.sum(self.H, axis=1) + epss
            self.W = np.multiply(self.W, np.divide(HVovWH.T, Hav))    

            # calculate loss
            B = np.dot(self.W, self.H) + epss
            VovB = np.log(np.divide(self.V, B) + epss)
            loss_new = np.sum(np.multiply(self.V, VovB) - V + B)
            end_time = time.time()
            print('Step={}, Loss={}, Time={}s'.format(itr, loss_new, end_time-start_time))
            itr += 1

            # check terminate condition
            if loss_old == None:
                loss_old = loss_new
                continue
            elif loss_old - loss_new <= self.tol:
                break
            else:
                loss_old = loss_new        

    def fit(self, V):
        assert type(V) == np.ndarray, 'error: require numpy.array as input'
        self.V = V
        n_row, n_col = self.V.shape
        n_topic = self.n_components
        self.W = np.random.random((n_row, n_topic))
        self.W = self.W * np.sqrt(self.V.mean() / self.n_components)
        self.H = np.random.random((n_topic, n_col))
        self.H = self.H * np.sqrt(self.V.mean() / self.n_components)

        if self.cost_function == "euclidean":
            self._update_euclidean()
        elif self.cost_function == "kullback-leibler":
            self._update_kl()

def text2index(posts, word2id):
    x_train = []
    for post in posts:
        p = []
        for w in post.split():
            idx = word2id.get(w)
            if idx != None:
                p.append(idx)
        x_train.append(p)
    return x_train

if __name__ == '__main__':
    print('Loading Data')
    docs = []
    with open('./data/whole.txt', 'r', encoding='utf-8') as file:
        for line in file.readlines():
            docs.append(line)
    print('Done')
    print(len(docs))

    vobs = []
    with open("./data/vobs.txt", "r") as f:
        for w in f.readlines():
            vobs.append(w.strip())
    print(len(vobs))

    V = len(vobs)

    word2id = {k: v for v, k in enumerate(vobs)}
    id2word = {v: k for k, v in word2id.items()}

    x_train = text2index(docs, word2id)
    x_train = np.array([np.bincount(doc, minlength=V) for doc in x_train])

    nmf = NMF(n_components=10, max_iter = 100, tol = 0.1, cost_function="euclidean")
    nmf.fit(x_train)

    topic_words = nmf.H
    for index, topic in enumerate(topic_words):
        message = "Topic #%d: " % index
        top_words = [id2word[i] for i in topic.argsort()[:-10 - 1:-1]]
        message += " ".join(top_words)
        print(message)