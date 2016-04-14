import numpy as np
from lib.Helpers import insert_symbol, remove_symbol, bern, get_words_freq, get_all_words_freq,change_freq, put_boundaries_randomly, get_current_word


class Segmenter:
    BOUNDARY = '.'  # the boundary symbol
    RANDOM_BOUND_PROB = 0.2 # the random boundary probability constant that controls what is a probability of putting initially boundaries in the text
    TOTAL_SYMBOL = '_TOTAL_'

    # inputs:
    #   p : beta parameter, i.e. number of heads and tails
    #   alpha : DP hyperparameter proportional to the probability of visitors to sit on an unoccupied table
    def __init__(self, char_freq, p, alpha):
        self.char_freq = char_freq
        self.p = p
        self.alpha = alpha


    # inputs:
    #   data: the preprocessed array of concatenated string sentences
    def run(self, text, iter):
        self.m = np.shape(text)[0] # number of sentences
        text = put_boundaries_randomly(text,self.RANDOM_BOUND_PROB)
        self.word_freq = get_all_words_freq(text,self.BOUNDARY)

        for i in range(iter):
            text = self.__gibbs(text)


    def __gibbs(self, text):
        # for every sentence
        for i in range(self.m):
            print(i)
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
            if not sent[i] == self.BOUNDARY:  # leave it the way it is if the current position has a boundary symbol
                sent = insert_symbol(sent, i, self.BOUNDARY)
        else:
            if sent[i] == self.BOUNDARY:
                sent = remove_symbol(sent, i)
        return sent

    # computes a probability to put a boundary in the position i
    def __boundary_prob(self, sent, i):
        h1 = self.__h1(sent,i)
        h2 = self.__h2(sent,i)
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
        n_c_w = self.word_freq[cur_word] if cur_word in self.word_freq else 0 # freq of the current word in the corpus
        n_w = self.word_freq[self.TOTAL_SYMBOL]
        n_u = self.m if get_current_word(temp_sent,m-1) == cur_word else n_w - self.m # here we check if the current word is the utterance final

        return (n_c_w + self.alpha*self.__P0(cur_word))*(n_u + self.p/2)/((n_w + self.alpha)*(n_w + self.p))

    def __h2(self,sent,i):
        return 0.2


    # two stage mechanism, the first stage first subtracts all the words in a sentence
    # and the second one adds them back when action has taken place
    # stages are chosen via inserted flag
    def __update_word_freq(self, sent, remove=False):
        freq = get_words_freq(sent,self.BOUNDARY)
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
            count = self.char_freq[char] if char in self.char_freq else 0
            res *= count / self.char_freq[self.TOTAL_SYMBOL]
        return self.p * (1 - self.p)**(m - 1) * res


