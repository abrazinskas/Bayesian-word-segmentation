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
        
    def run(self):
        total_P = total_R = total_F = total_BP = total_BR = total_BF = total_LP = total_LR = total_LF = 0     
        for i in range(self.m):
            sent = self.text[i]
            segm_sent = self.segm_text[i]
            P, R, F = self.__boundary_eval(sent, segm_sent)
            BP, BR, BF = self.__ambigious_eval(sent, segm_sent)
            LP, LR, LF = self.__lexicon_eval(sent, segm_sent)
            #calculate precision recall, F0 over all text
            total_P += P
            total_R += R
            total_F += F
            total_BP += BP
            total_BR += BR
            total_BF += BF
            total_LP += LP
            total_LR += LR
            total_LF += LF
            
        P = 100* (total_P/float(self.m))
        R = 100 * (total_R/float(self.m))
        F = 100 * (total_F/float(self.m))
        BP = 100 * (total_BP/float(self.m))
        BR = 100* (total_BR/float(self.m))
        BF = 100 *(total_BF/float(self.m))
        LP = 100* (total_LP/float(self.m))
        LR = 100 * (total_LR/float(self.m))
        LF = 100 * (total_LF/float(self.m))
        
            
        return (P, R, F, BP, BR, BF, LP, LR, LF)
            
    
    # Correctly defined boundaries: precision, recall, F0-measure
    # so consider correct boundary before and after word
    #Input:
    #   bound_sent = index of boundaries from correct sentence
    #   bound_segm = index of boundaries from segmented sentence
    def __boundary_eval(self, sent, segm):
        sent = sent.split(' ')
        segm = segm.split('.')
        correct_bound = 0
        for i in range(len(sent)):
            if sent[i] in segm:
                correct_bound += 1
        
        if correct_bound == 0:
            P = 0
            R = 0
            F0 = 0
        else:
            p_total = len(segm)
            r_total = len(sent)
            P = correct_bound/float(p_total)
            R = correct_bound/float(r_total)
            F0 = self.__F0( P, R)        
        
        return (P, R, F0)
        
    # Precision, recall & F0 on the lexicon: type of words
    def __lexicon_eval(self, sent, segm):
        sent = sent.split(' ')
        segm_split = segm.split('\.')
        correct_lexi = 0
        for i in range(len(sent)):
            if sent[i] in segm_split:
                correct_lexi += 1
        
        # Find all words, so exclude the ones with only one letter:
        index_segm = self.__index_boundary(segm, True)
        bounds = [0] + index_segm + [(len(index_segm) + 1)]
        
        count = 0
        for i in range(len(bounds)-1):
            if (bounds[i] - bounds[i+1] == 1):
                count += 1
            
        if correct_lexi == 0:
            P = 0
            R = 0
            F0 = 0
        else:        
            p_total = len(segm_split) - count
            r_total = len(sent)
            P = correct_lexi/float(p_total)
            R = correct_lexi/float(r_total)
            F0 = self.__F0(P, R)
        return (P, R, F0)
        
    # Ambigious boundaries precision, recall & F0
    # All predicted boundaries which are correct, not considering the word
    def __ambigious_eval(self, sent, segm):
         index_sent = self.__index_boundary(sent, False)
         index_segm = self.__index_boundary(segm, True)
         correct_ambi = 0
         for i in range(len(index_sent)):
             ind = index_sent[i]
             letter1 = sent[ind-1]
             letter2 = sent[ind+1]
             for j in range(len(index_segm)):
                 ind_seg = index_segm[j]
                 letter3 = segm[ind_seg-1]
                 letter4 = segm[ind_seg+1]
                 if letter1 == letter3 and letter2 == letter4:
                     correct_ambi += 1

         if correct_ambi == 0:
            P = 0
            R = 0
            F0 = 0
         else: 
            p_total = len(index_segm)
            r_total = len(index_sent)
            P = correct_ambi/float(p_total)
            R = correct_ambi/float(r_total)

            F0 = self.__F0(P, R)
         return (P, R, F0)
        
    #Compute F0 measure from precision and recall    
    def __F0(self, P, R):
        f0 = float(2*P*R)/float(P+R)
        return float(f0)
        
    #Get indeces of all whitespaces from a sentence
    def __index_boundary(self, sent, point):
        indeces = []
        if point:
            for m in re.finditer('\.', sent):
                indeces.append(m.start())
        else: 
            for m in re.finditer(' ', sent):
                indeces.append(m.start())
        return indeces