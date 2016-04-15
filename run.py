# this is a dummy file to run other scripts
# it contains examples on how to run the main Segmenter class
from lib.Helpers import load_text, put_boundaries_randomly, remove_symbol

from Segmenter import Segmenter

# loading data
#file_path = 'data/br-phono-train.txt'
file_path = 'data/small.txt'
(text,word_freq,char_freq) = load_text(file_path)
segmenter = Segmenter(text,char_freq, p=0, alpha=2000)
segmenter.run(10)