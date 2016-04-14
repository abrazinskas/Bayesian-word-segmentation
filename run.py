# this is a dummy file to run other scripts
# it contains examples on how to run the main Segmenter class
from lib.Helpers import load_text, put_boundaries_randomly, remove_symbol,get_current_word

from Segmenter import Segmenter

# loading data
#file_path = 'data/br-phono-train.txt'
file_path = 'data/small.txt'
(text,word_freq,char_freq) = load_text(file_path)
segmenter = Segmenter(char_freq=char_freq, p=0.5, alpha=0.5)
segmenter.run(text,10)

text=put_boundaries_randomly(text,0.5)
sent = text[0]
print(get_current_word(text[0],len(sent)-1))

print(text)