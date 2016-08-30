# -*- coding:utf-8 -*-
import ctypes
import numpy as np
from scipy.linalg import norm
from numpy.ctypeslib import ndpointer
import tempfile
import os
import sys
lib = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__),
                                           'libfasttext.so'))

_get_dim = lib.get_dim
_get_dim.restype = ctypes.c_int32

_mat_rows = lib.mat_rows
_mat_rows.restype = ctypes.c_int64

_mat_cols = lib.mat_cols
_mat_cols.restype = ctypes.c_int64

_mat_data = lib.mat_data

_get_id = lib.get_id
_get_id.restype=ctypes.c_int32

_get_Vector = lib.get_Vector
_get_Vector.argtypes=[ctypes.c_char_p,ctypes.c_int64,ctypes.c_void_p]

_cpredict = lib.cpredict

_cpredict.argtypes = [ctypes.c_char_p,ctypes.c_int32,ctypes.c_void_p]

_get_label = lib.get_label
_get_label.argtyes = ctypes.c_int32
_get_label.restype=ctypes.c_char_p

_get_nlabels = lib.n_labels
_get_nlabels.restype= ctypes.c_int32


class FastText(object):
    def __init__(self):
        self.__dealloc = lib.dealloc
        lib.init_tables()
    @property
    def dim(self):
        return self.__dim

    def train(self,input,model='skipgram',lr=0.05,dim=100,
              window=5,epoch=5,min_count=5,negative=5,
              word_ngrams=1,loss_fn='hs',bucket=2000000,minn=3,
              maxn=6,thread=12,lrUpdateRate=100,t=1e-4,label='__label__',
              verbose=False,
              ):
        """
        train the model using given params

        :param input: training file path
        :param lr: learning rate
        :param dim: size of word vectors
        :param window: size of the context window
        :param epoch: number of epochs
        :param min_count: minimal number of word occurences
        :param negative: number of negatives sampled
        :param word_ngrams: max length of word ngram
        :param loss_fn: loss function {ns, hs, softmax}
        :param model: skip_gram(sg),cbow or supervised
        :param bucket: number of buckets
        :param minn: min length of char ngram
        :param maxn: max length of char ngram
        :param thread: number of threads
        :param lrUpdateRate:
        :param t:  sampling threshold
        :param label:  labels prefix
        :return:
        """
        args = ['fasttext']
        _opt = ('supervised','cbow','skipgram')
        if model not in _opt:
            raise ValueError("model should be one of [{}]".format(",".join(_opt)))
        def add_arg(*options):
            args.extend(options)

        args.append(model)
        add_arg('-input',input)
        add_arg('-output','/tmp')
        add_arg('-lr',lr)
        add_arg('-lrUpdateRate',lrUpdateRate)
        add_arg('-dim',dim)
        add_arg('-ws',window)
        add_arg('-epoch',epoch)
        add_arg('-minCount',min_count)
        add_arg('-neg',negative)
        add_arg('-wordNgrams',word_ngrams)
        _opt = ('ns','hs','softmax')
        if loss_fn not in _opt:
            raise ValueError("loss_fn should be one of [{}]".format(",".join(_opt)))
        add_arg('-loss',loss_fn)
        add_arg('-bucket',bucket)
        add_arg('-minn',minn)
        add_arg('-maxn',maxn)
        add_arg('-thread',thread)
        add_arg('-t',t)
        add_arg('-label',label)
        args = map(lambda opt:str(opt),args)
        if verbose:
            print (args)
        c_args_type = (ctypes.c_char_p * len(args))
        c_args = c_args_type()
        for i,arg in enumerate(args):
            c_args[i] = arg
        lib.train(len(args),c_args)
        self.__read_property()
        return self
    def __read_property(self):
        self.__dim = _get_dim()
        self.__rows = _mat_rows()
        self.__cols = _mat_cols()

    @property
    def data(self):

        mat_data = lib.mat_data
        try:
            mat_data.restype = ndpointer(dtype=ctypes.c_float,
                                        shape=(self.__rows,self.__cols))
        except AttributeError:
            return None
        return mat_data()

    def save(self,filename):
         if lib.saveModel(ctypes.c_char_p(filename))==1:
             raise ValueError("Model file cannot be opened for saving!")

    def load(self,filename):
        if lib.loadModel(ctypes.c_char_p(filename))==1:
            raise ValueError("Model file cannot be opened for loading!")
        self.__read_property()
        return self
    def __getitem__(self, word):
        wd = to_bytes_str(word)
        vec = np.zeros(self.dim,dtype=np.float32)
        _get_Vector(wd,self.dim,vec.ctypes.data)
        return vec

    def transform(self):
        """
        create all word vectors using n-grams.
        :param words: word list to be transform into vectors
        :param release: release n_gram data to save memory
        :return: None
        """
        tmp = os.path.join(tempfile.gettempdir(),"tmp.vec")
        if lib.saveVectors(tmp)==1:
            raise RuntimeError("can't write to temp file")
        if sys.version > "3":
            def compat_splitting(line):
                return line.split()
        else:  # if version is 2
            def compat_splitting(line):
                return line.decode('utf8').split()
        vectors = dict()
        with open(tmp,'r') as fin:
            for line in fin:
                try:
                    tab = compat_splitting(line)
                    vec = np.array(tab[1:], dtype=float)
                    word = tab[0]
                    if not word in vectors:
                        vectors[word] = vec
                except ValueError:
                    continue
                except UnicodeDecodeError:
                    continue

        word_emb = np.zeros((len(vectors),self.__dim),dtype=np.float32)
        vocab = dict()
        self.idx2wd = dict()
        for i,(k,v) in enumerate(vectors.iteritems()):
            vocab[k] = i
            word_emb[i] = v
            self.idx2wd[i] = k
        self.word_emb = word_emb
        self.vocab = vocab

    def __init_sims(self,replace=True):
        if replace:
            self.word_emb/=norm(self.word_emb,axis=1)[:,None]
            self.word_emb_init = self.word_emb
        else:
            self.word_emb_init = self.word_emb/norm(self.word_emb,axis=1)[:,None]

    def most_similar(self,word,topn=10):
        """

        :param word: word to be evaluate
        :param topn: top K most similar words to return
        :return: tuple of most similar (words,similarity)
        """
        if not hasattr(self,"word_emb_unit"):
            self.__init_sims(False)
        sim = np.dot(self.word_emb_init,self.word_emb_init[self.vocab[word]])

        indexes = np.argsort(-sim)[1:topn+1]
        res_words = [self.idx2wd[_] for _ in indexes]
        res_sim = [sim[_] for _ in indexes]
        return res_words,res_sim

    def __init_labels(self):
        if hasattr(self,"_labels"):
            return
        else:
            self._labels = {}
            for i in  range(_get_nlabels()):
                self._labels[i]= unicode(_get_label(i))

    @property
    def labels(self):
        if not hasattr(self,'_labels'):
            self.__init_labels()
        return self._labels

    def predict(self,text,topk=1,class_label=True):
        """
        predict the class label of the given text
        :param text: text to be assign class label
        :param topk:
        :param class_label:
        :return:
        """
        pred = np.zeros(topk,np.int32)
        line = to_bytes_str(text)
        line = line.replace('\n',' ')
        if _cpredict(line,topk,pred.ctypes.data)==-1:
            raise ValueError("all words in line not in dictionary")
        if class_label:
            self.__init_labels()
            return [self._labels[i] for i in pred]
        return pred


    def similarity(self,word1,word2):
        """
        :param word1: str or utf8 string
        :param word2: str or utf8 string
        :return: cosine similarity score of word1 and word2
        """
        v1 = self[word1]
        v2 = self[word2]
        return np.dot(v1,v2)/(norm(v1)*norm(v2))

    def __del__(self):
        lib.free_tables()
        self.__dealloc()

def to_bytes_str(s):
    if isinstance(s,unicode):
        return s.encode('utf8')
    if isinstance(s,str):
        return s
