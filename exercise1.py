from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re
import datetime
import os
import math 

def test (path,pos_voc_rep,pos_dict,p_pos,neg_voc_rep,neg_dict,p_neg,option):
    result = {}
    P = 0
    N = 0
    listing = os.listdir(path)
    for file in listing:
        
        #clean the review under test=========================
        print('testing ........' + file)
        file= open(path+file, "r" , encoding="utf8")
        filewords = file.read().lower()
        file.close()
        for word in stopwords.words('english'):
            regex = r"\b"+re.escape(word)+r"\b"
            filewords = re.sub(regex,'',filewords)
        filter1=RegexpTokenizer(r'[a-z]\w+')
        wordslist=filter1.tokenize(filewords)
        
        #Multinomial=========================================
        pp = math.log10(p_pos)
        pn = math.log10(p_neg)
        for word in wordslist:
            if word in pos_dict:
                pp += math.log10(pos_dict[word])
            else:
                pp += math.log10(1/(len(pos_voc_rep)+len(pos_voc)+len(neg_voc)))
            
            if word in neg_dict:
                pn += math.log10(neg_dict[word])
            else:
                pn += math.log10(1/(len(neg_voc_rep)+len(pos_voc)+len(neg_voc)))
        if pp > pn:
            result[file] = 'Positive'
            P +=1
        else:
            result[file] = 'Negative'
            N += 1    
    #accuracy========================         
    if option == 'Positive':        
        print((P/len(listing))*100)
    else:
        print((N/len(listing))*100)        
    return (result)         


def read_dataset (path):
    listing = os.listdir(path)
    vocab_inits = ''
    vocab_rep_unfs = ''

    print('reading dataset.......')
    for file in listing:
        file= open(path+file, "r" , encoding="utf8")
        filewords = file.read().lower()
        file.close()
        vocab_inits += ' ' + filewords  

    vocab_rep_unfs = vocab_inits
    for word in stopwords.words('english'):
        regex = r"\b"+re.escape(word)+r"\b"
        vocab_rep_unfs = re.sub(regex,'',vocab_rep_unfs)
   
    filter1=RegexpTokenizer(r'[a-z]\w+')
    vocab_rep_l=filter1.tokenize(vocab_rep_unfs)
    
    return(vocab_rep_l)

#read==============================  
T_P_Dir="D:\\CIT\\AI\\Machine_Learning\\Text_classification\\byclassdata\\data\\trainSmall\\pos\\"
T_N_Dir="D:\\CIT\\AI\\Machine_Learning\\Text_classification\\byclassdata\\data\\trainSmall\\neg\\"

#set of all positive vocabulary 
pos_voc_rep=read_dataset(T_P_Dir)
pos_voc = set(pos_voc_rep)

#set of all negative vocabulary 
neg_voc_rep=read_dataset(T_N_Dir)
neg_voc = set(neg_voc_rep)

plisting = os.listdir(T_P_Dir)
nlisting = os.listdir(T_N_Dir)

#p(pos) and p(neg) = (total number of docs in a class/total number of docs)
p_pos = len(plisting)/(len(plisting)+len(nlisting))
p_neg = len(nlisting)/(len(plisting)+len(nlisting))

#create Model=============================
print('Creating positive model.......')
pos_dict={}
for word in pos_voc:
    pos_dict[word] = (pos_voc_rep.count(word) + 1)/(len(pos_voc_rep)+len(pos_voc)+len(neg_voc))

print('Creating negative model........')
neg_dict={}
for word in pos_voc:
    neg_dict[word] = (neg_voc_rep.count(word) + 1)/(len(neg_voc_rep)+len(pos_voc)+len(neg_voc))

#test==============================
result=test("D:\\CIT\\AI\\Machine_Learning\\Text_classification\\byclassdata\\data\\test\\neg\\",pos_voc_rep,pos_dict,p_pos,neg_voc_rep,neg_dict,p_neg,'Negative')    



