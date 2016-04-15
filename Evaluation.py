import numpy as np
import re

class Evaluation:
    __BOUNDARY = '.'
    
    #Input:
    #   text = correct text 
    #   segm_text = our segmented text
    def __init__(self, text, segm_text):
        self.text = text
        self.segm_text = segm_text
        self.m = np.shape(text)[0]
        
    def __run(self, text, segm_text):
        for i in range(self.m):
            sent = text[i]
            segm_sent = segm_text[i]
            bound_sent = __index_boundary(self, text)
            (P, R, F) = __boundary_eval(self, bound_sent, bound_segm)
            (BP, BR, BF) = __ambigious_eval(self, bound_sent, bound_segm)
            (LP, LR, LF) = __lexicon_eval(self, sent, text)
            #TODO: calculate precision recall, F0 over all text
            
    
    # Correctly defined boundaries: precision, recall, F0-measure
    # so consider correct boundary before and after word
    #Input:
    #   bound_sent = index of boundaries from correct sentence
    #   bound_segm = index of boundaries from segmented sentence
    def __boundary_eval(self, bound_sent, bound_segm):
        correct_bound = 0
        for i in range(len(bound_sent) - 1):
            if bound_sent[i] == bound_segm[i] and bound_sent[i+1] == bound_segm[i+1]:
                correct_bound += 1
        
        #Do not see beginning as boundary, thus minus one
        p_total = len(bound_segm) - 1
        r_total = len(bound_sent) - 1
        P = correct_bound/p_total
        R = correct_bound/r_total
        F0 = __F0(self, P, R)        
        
        return (P, R, F0)
    
    
    # Precision, recall & F0 on the lexicon: type of words
    def __lexicon_eval(self, sent, segm):
        sent = sent.split(' ')
        segm = segm.split(' ')
        correct_lexi = 0
        for i in range(len(sent)):
            if sent[i] in segm:
                correct_lexi += 1
                
        p_total = len(segm) - 2
        r_total = len(sent) - 2
        P = correct_lexi/p_total
        R = correct_lexi/r_total
        F0 = __F0(self, P, R)
        return (P, R, F0)
        
    # Ambigious boundaries precision, recall & F0
    # All predicted boundaries which are correct, not considering the word
    def __ambigious_eval(self, bound_sent, bound_segm):
         c = np.intersect1d(bound_sent, bound_segm)
         correct_ambi = len(c)
         
         #Do not include first and last boundary
         p_total = len(bound_segm) - 2
         r_total = len(bound_sent) - 2
         P = correct_ambi/p_total
         R = correct_ambi/r_total
         F0 = __F0(self, P, R)
         return (P, R, F0)
        
    #Compute F0 measure from precision and recall    
    def __F0(self, P, R):
        f0 = (2*P*R)/(P+R)
        return float(f0)
        
    #Get indeces of all whitespaces from a sentence
    def __index_boundary(self, sent):
        indeces = [0]
        for m in re.finditer(' ', sent):
            indeces.append(m.start())
        i = len(sent)
        indeces.append[i]
        return indeces