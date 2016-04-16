# this is a dummy file to run other scripts
# it contains examples on how to run the main Segmenter class
from lib.Helpers import load_text, load_file

from Segmenter import Segmenter
from Evaluation import Evaluation

# loading data
file_path = 'data/br-phono-train.txt'
#file_path = 'data/small.txt'
(text,word_freq,char_freq) = load_text(file_path)
segmenter = Segmenter(text=text,char_freq=char_freq, p=2, alpha=20, p_hash = 0.5)
print "Start segmenting \n" 
segm_text = segmenter.run(200)


filetext = load_file(file_path)

print "Start evaluating \n"
evaluation = Evaluation(filetext, segm_text)
P, R, F, BP, BR, BF, LP, LR, LF = evaluation.run()

print "Boundary evaluation: \n Precision: %.2f, Recall: %.2f, F-measure: %.2f \n" % (P, R, F)
print "Ambigious boundary evaluation: \n Precision: %.2f, Recall: %.2f, F-measure: %.2f \n" % (BP, BR, BF)
print "Lexicon evaluation: \n Precision: %.2f, Recall: %.2f, F-measure: %.2f \n" % (LP, LR, LF)

