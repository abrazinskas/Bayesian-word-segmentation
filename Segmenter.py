import numpy as np
from lib.Helpers import insert_symbol, remove_symbol, bern, get_words_freq, get_all_words_freq, get_current_word, put_boundaries_randomly, get_word
from lib.FreqVocab import FreqVocab


class Segmenter:
    __BOUNDARY = '.'  # the boundary symbol
    __RANDOM_BOUND_PROB = 0.05 # the random boundary probability constant that controls what is a probability of putting initially boundaries in the text
    __TOTAL_SYMBOL = '_TOTAL_'

    __T = None # current temperature


    # inputs:
    #   p : beta parameter, i.e. number of heads and tails
    #   alpha : DP hyper-parameter proportional to the probability of visitors to sit on an unoccupied table
    def __init__(self, text, char_freq, p, alpha, p_hash):
        self.char_freq = char_freq
        self.p = p
        self.alpha = alpha
        self.p_hash = p_hash # this hyperparameter is used in P_0

        self.m = np.shape(text)[0] # number of sentences
        self.text = put_boundaries_randomly(text,self.__RANDOM_BOUND_PROB)

        # creating vocab
        freq = get_all_words_freq(self.text, self.__BOUNDARY)
        self.word_freq = FreqVocab(freq, self.__TOTAL_SYMBOL)


    # inputs:
    #   data: the preprocessed array of concatenated string sentences
    def run(self, iter):
        temps = np.arange(0,1,step=1.0/float(iter)) # temperatures
        for i in range(iter):
            self.__T = temps[i]
            self.text = self.__gibbs(self.text)
            print ("Iteration: %d" % i)
        return self.text


    def __gibbs(self, text):
        # for every sentence
        for i in range(self.m):
            sent = text[i]
            j = 0  # position in a sentence
            while True:
                n = len(sent)
                # since we will change the length of the sentence the stopping condition is dynamic
                if j == n - 1: break
                p = self.__boundary_prob(sent,j)
                self.word_freq.update_freq(sent, remove=True)
                sent = self.__action(sent, j, p)
                self.word_freq.update_freq(sent, remove=False)

                # adjust positions based on the decision that has been made
                if len(sent)>n: # we added a boundary
                    j+=1
                if len(sent)<n: # we removed a boundary
                    j-=1
                j+=1
            text[i]=sent
        return text

    # this function flips a coin based on probabiliy of putting a boundary
    # and then takes action (removes or places a new boundary symbol or does not do anything)
    # finally, it return a new sentence (with new or without a boundary symbol)
    # inputs :
    #   sent : current sentence
    #   i : current position in the sentence
    #   p: probability for a new boundary
    def __action(self, sent, i, p):
        if bern(p):
            if not (sent[i+1] == self.__BOUNDARY or sent[i] == self.__BOUNDARY):  # leave it the way it is if the current position has a boundary symbol
                sent = insert_symbol(sent, i, self.__BOUNDARY)
        else:
            if sent[i] == self.__BOUNDARY:
                sent = remove_symbol(sent, i)
        return sent

    # computes a probability to put a boundary in the position i
    # takes temperature into account
    def __boundary_prob(self, sent, i):
        h1 = self.__h1(sent,i)**self.__T
        h2 = self.__h2(sent,i)**self.__T
        if h1+h2 == 0: return 0
        return h2/(h1+h2)

    # hypothesis of NOT putting a boundary
    # inputs :
    #   sent : current sentence
    #   i : current index
    def __h1(self, sent, i):
        # TODO: need to think about the way the current word is produced, this one can be wrong
        temp_sent = self.__action(sent, i, 0)
        cur_word = get_current_word(temp_sent, i)

        m = len(sent)
        n_c_w = self.word_freq.get_freq(cur_word) # freq of the current word in the corpus
        n_w = self.word_freq.get_total_freq()
        n_u = self.m if get_current_word(temp_sent,m-1) == cur_word else n_w - self.m # here we check if the current word is the utterance final

        enum = (n_c_w + self.alpha*self.__P0(cur_word))*(n_u + self.p/2)
        denom =(n_w + self.alpha)*(n_w + self.p)
        return enum/denom

    #Hypothesis of putting a boundary
    def __h2(self,sent,i):
        temp_sent = self.__action(sent, i, 0)
        
        w2 = get_word(temp_sent, i, before=True)
        w3 = get_word(temp_sent, i, before=False)       
        n_w2 = self.word_freq.get_freq(w2)
        n_w3 = self.word_freq.get_freq(w3)
        p_w2 = self.__P0(w2)
        p_w3 = self.__P0(w3)
        m = len(sent)
        n_u = self.m
        n_w = self.word_freq.get_total_freq()

        # If words are the same, I is one   
        if w2 == w3:
            I = 1
        else:
            I = 0
        
        # If second word is final word, I_final is zero
        if get_current_word(temp_sent, m-1) == w3:
            I_final = 0
        else:
            I_final = 1
        
        enum = (n_w2 + self.alpha*p_w2) * (n_w - n_u + (self.p/2)) * (n_w3 + 1 + self.alpha*p_w3) * (n_u +I_final + (self.p/2)) 
        denom = (n_w + self.alpha) * (n_w+self.p) *  (n_w + 1 + self.alpha) *(n_w+1+self.p)
        
        return enum/denom

    # 1. base distribution
    def __P0(self, w):
        m = len(w)
        res = 1.0
        total = self.char_freq[self.__TOTAL_SYMBOL]
        for i in range(m):
            char = w[i]
            count = self.char_freq[char] if char in self.char_freq else 0
            res *= count
        return (res/total**m)*self.p_hash * (1 - self.p_hash)**(m - 1)


