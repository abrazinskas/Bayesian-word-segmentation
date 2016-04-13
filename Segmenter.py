import numpy as np
from lib.Helpers import insert_symbol, remove_symbol, bern, get_words_freq, get_all_words_freq,change_freq, put_boundaries_randomly


class Segmenter:
    boundary = '.'  # the boundary symbol

    # TODO: write the meaning of parameters p and alpha
    def __init__(self,char_freq, p, alpha):
        self.char_freq = char_freq
        self.p = p
        self.alpha = alpha


    # inputs:
    #   data: the preprocessed array of concatenated string sentences
    def run(self, text, iter):
        text = put_boundaries_randomly(text,0.1)
        self.word_freq = get_all_words_freq(text,self.boundary)
        for i in range(iter):
            text = self.__gibbs(text)


    def __gibbs(self, text):
        m = np.shape(text)[0]
        # for every sentence
        for i in range(m):
            sent = text[i]
            j = 0  # position in a sentence
            while True:
                n = len(sent)
                # since we will change the length of the sentence the stopping condition is dynamic
                if j == n - 1: break
                p = self.__boundary_prob(sent,j)
                self.__update_word_freq(sent, remove=True)
                sent = self.__action(sent, j, p)
                self.__update_word_freq(sent, remove=False)

                # adjust positions based on the decision that has been made
                if len(sent)>n: # we added a boundary
                    j+=1
                if len(sent)<n: # we removed a boundary
                    j-=1

                j+=1
            text[i]=sent

    # this function flips a coin based on probabiliy of putting a boundary
    # and then takes action (removes or places a new boundary symbol or does not do anything)
    # finally, it return a new sentence (with new or without a boundary symbol)
    # inputs :
    #   sent : current sentence
    #   i : current position in the sentence
    #   p: probability for a new boundary
    def __action(self, sent, i, p):
        if bern(p):
            if not sent[i] == self.boundary:  # leave it the way it is if the current position has a boundary symbol
                sent = insert_symbol(sent, i, self.boundary)
        else:
            if sent[i] == self.boundary:
                sent = remove_symbol(sent, i)
        return sent

    # computes a probability to put a boundary in the position i
    # note that we negate the probability of NOT putting a boundary
    def __boundary_prob(self, sent, i):

        return 0.5

    # two stage mechanism, the first stage first subtracts all the words in a sentence
    # and the second one adds them back when action has taken place
    # stages are chosen via inserted flag
    def __update_word_freq(self, sent, remove=False):
        freq = get_words_freq(sent,self.boundary)
        if remove:
            self.word_freq=change_freq(self.word_freq,freq, remove=True)
        else:
            self.word_freq=change_freq(self.word_freq,freq, remove=False)






    # 1. base distribution
    def __P0(self, w):
        m = len(w)
        res = 1
        for i in range(m):
            char = w[i].lower()
            res *= self.unigrams(char) / self.unigrams('_TOTAL_')
        return self.p * (1 - self.p) ^ (m - 1)


