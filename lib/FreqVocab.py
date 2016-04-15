from lib.Helpers import get_words_freq, change_freq

class FreqVocab:
    freq_vocab=None

    def __init__(self,freq,total_symbol='_TOTAL_'):
        self.total_symbol=total_symbol
        self.freq_vocab=freq

    # two stages mechanism, the first stage first subtracts all the words in a sentence
    # and the second one adds them back when action has taken place
    # stages are chosen via inserted flag
    def update_freq(self, sent, remove=False, bound_symb='.'):
        # TODO: implement a garbage collector to remove words that have 0 freq
        freq = get_words_freq(sent, bound_symb)
        if remove:
            self.freq_vocab = change_freq(self.freq_vocab, freq, remove=True)
        else:
            self.freq_vocab = change_freq(self.freq_vocab, freq, remove=False)


    def get_freq(self,word):
        word = word.lower()
        if word in self.freq_vocab:
            return self.freq_vocab[word]
        else:
            return 0

    def get_total_freq(self):
        return self.freq_vocab[self.total_symbol]