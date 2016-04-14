import numpy as np

# join parameter controls if the output should be an array of joint strings
# returns data, words, and letters frequencies
def load_text(filename,join=True, total_key = '_TOTAL_'):
    data=[]
    char_freq = {
        total_key:0
    }
    word_freq = {
        total_key:0
    }

    with open(filename) as f:
        for sent in f:
            s = sent.strip().split(' ')
            word_freq[total_key]+=len(s)

            for word in s:
                word = word.lower()
                # storing word frequencies
                if word in word_freq:
                    word_freq[word]+=1
                else:
                    word_freq[word]=1

                # storing character frequencies
                for char in word:
                    if(char in char_freq):
                        char_freq[char]+=1
                    else:
                        char_freq[char]=1
                    char_freq[total_key]+=1 # the total number of words

            if (join): s= "".join(s)
            data.append(s)
    return data, word_freq, char_freq


# returns word freq. from 1 dimensional array of text
def get_all_words_freq(text,sep=' ',total_key='_TOTAL_'):
    word_freq = {
        total_key:0
    }
    n = np.shape(text)[0]
    for i in range(n):
        freq = get_words_freq(text[i],sep)
        word_freq = change_freq(word_freq,freq)
    return word_freq


# returns word frequencies for a sentence
def get_words_freq(sent,sep='.'):
    freq ={}
    words = sent.split(sep)
    for word in words:
        if(word in freq):
            freq[word]+=1
        else:
            freq[word]=1
    return freq


# changes all_freq by subtracting or adding frequencies from freq hash
def change_freq(all_freq,freq,remove=False,total_symb='_TOTAL_'):
    for word in freq:
        if remove:
            if word in all_freq and all_freq[word]>0:
                all_freq[word]-=freq[word]
                all_freq[total_symb]-=freq[word]
        else:
            if(word in all_freq):
                all_freq[word]+=freq[word]
            else:
                all_freq[word]=freq[word]
            all_freq[total_symb]+=freq[word]
    return all_freq


# inputs :
#   p: probability to put a boundary
#   text : 1 dim. array of strings
def put_boundaries_randomly(text,p,symb='.'):
    m = np.shape(text)[0]
    for j in range(m):
        sent = text[j]
        i = 0  # position in a sentence
        while True:
            n = len(sent)
            if i == n - 1: break
            if bern(p):
                 # we don't want to have two boundaries in a row
                if not (i!=0 and sent[i-1]==symb):
                    sent = insert_symbol(sent,i,symb)
            # adjust positions based on the decision that has been made
            if len(sent)>n: # we added a boundary
                i+=1
            if len(sent)<n: # we removed a boundary
                i-=1
            i+=1
        text[j]=sent
    return text


# inputs :
#   st : a string
#   i : position where to insert the symbol
# symb: textual symbol
def insert_symbol(st,i,symb):
    return st[0:i+1]+str(symb)+st[i+1:]

def remove_symbol(st,i):
    return st[0:i]+st[i+1:]

# performs a Bernoulli trial
def bern(p):
    return np.random.binomial(1, p)



# returns the current words between the two boundaries in the [sent]
# based on the current position [i]
def get_current_word(sent, i,bound_symb='.'):
    m = len(sent)
    start = i
    end = i
    # search for the left end
    while True:
        if start == 0 or start > (m-1) : break
        if sent[start] == bound_symb:
            start+=1 # we add one because we don't want to return the boundary symbol in the beginning, ie. :'.dog'
            break
        start-=1
    # search for the right end
    while True:
        if end >= (m - 1): break
        if sent[end] == bound_symb:
            end-=1 # same story as in the previous loop
            break
        end+=1
    return sent[start:end+1]