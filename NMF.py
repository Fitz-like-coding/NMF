import time
import numpy as np

class NMF(object):
    def __init__(self, n_components, max_iter, tol):
        self.V = np.array([])
        self.W = np.array([])
        self.H = np.array([])
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def _update(self):
        epss = 1e-20
        loss_old = None
        itr = 0
        start_time = time.time()
        while itr < self.max_iter:
            #update H
            WTV = np.dot(self.W.T, self.V)
            WTWH = np.dot(np.dot(self.W.T, self.W), self.H) + epss
            self.H = np.multiply(self.H, np.divide(WTV, WTWH))
            #update W
            VHT = np.dot(self.V, self.H.T)
            WHHT = np.dot(np.dot(self.W, self.H), self.H.T) + epss
            self.W = np.multiply(self.W, np.divide(VHT, WHHT))

            loss_new = np.subtract(self.V, np.dot(self.W, self.H))
            loss_new = np.sum(loss_new*loss_new)
            end_time = time.time()
            print('Step={}, Loss={}, Time={}s'.format(itr, loss_new, end_time-start_time))
            itr += 1
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
        self.H = np.random.random((n_topic, n_col))

        self._update()


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

    nmf = NMF(n_components=10, max_iter = 50, tol = 0.1)
    nmf.fit(x_train)

    topic_words = nmf.H
    for index, topic in enumerate(topic_words):
        message = "Topic #%d: " % index
        top_words = [id2word[i] for i in topic.argsort()[:-10 - 1:-1]]
        message += " ".join(top_words)
        print(message)