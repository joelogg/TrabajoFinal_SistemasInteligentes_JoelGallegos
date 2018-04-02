limit = {
        'maxq' : 25,
        'minq' : 1,
        'maxa' : 25,
        'mina' : 1
        }

UNK = 'unk'
VOCAB_SIZE = 8000#22000 #8000

import nltk
import itertools

import numpy as np

import pickle
import tokenizer as t

import pandas as pd

def reducirPalabras(questions2, answers2):
    questions = []
    answers = []
    
    for lineaq, lineaa in zip(questions2, answers2):
        lineaAuxq = lineaq.split(" ") 
        lineaAuxa = lineaa.split(" ") 
        tamq = len(lineaAuxq)
        tama = len(lineaAuxa)
        if(tamq<=25 and tama<=25):
            questions.append(lineaq)
            answers.append(lineaa)
        else:
            oracionReducidaq = ""
            oracionReducidaa = ""
            i = 1
            if (tamq<=50 and tama<=50):
            #if (tamq<=25 and tama<=25):
                for palabraq, palabraa in zip(lineaAuxq,lineaAuxa):
                    if i<25:
                        if i==24:
                            oracionReducidaq = oracionReducidaq + palabraq
                            oracionReducidaa = oracionReducidaa + palabraa
                        else:
                            oracionReducidaq = oracionReducidaq + palabraq+' '
                            oracionReducidaa = oracionReducidaa + palabraa + ' '
                    i = i+1
                questions.append(oracionReducidaq)
                answers.append(oracionReducidaa)
        
        
    
    
    return questions, answers

'''---leer datos----'''
def gather_dataset3():    
    data = pd.read_csv('interbank_final.csv', index_col=0)    
    questionsAux = data['feed_message'].tolist()
    answersAux = data['comment_message'].tolist()
    
    questions2 = []
    answers2 = []
    for linea in questionsAux:
       questions2.append( t.getSoloLetras(linea) )
    for linea in answersAux:
       answers2.append( t.getSoloLetras(linea) )
    
    '''
    for i in range(1000):
       questions2.append( t.getSoloLetras(questionsAux[i]) )
    for i in range(1000):
       answers2.append( t.getSoloLetras(answersAux[i]) )
    '''
        
    return reducirPalabras(questions2, answers2)
    #return questions2, answers2

questions, answers = gather_dataset3()






''' --- filtrando oraciones largas ---'''
def filter_data(qseq, aseq):
    filtered_q, filtered_a = [], []
    raw_data_len = len(qseq)

    assert len(qseq) == len(aseq)

    for i in range(raw_data_len):
        qlen, alen = len(qseq[i].split(' ')), len(aseq[i].split(' '))
        if qlen >= limit['minq'] and qlen <= limit['maxq']:
            if alen >= limit['mina'] and alen <= limit['maxa']:
                filtered_q.append(qseq[i])
                filtered_a.append(aseq[i])

    # print the fraction of the original data, filtered
    filt_data_len = len(filtered_q)
    filtered = int((raw_data_len - filt_data_len)*100/raw_data_len)
    print(str(filtered) + '% filtered from original data')

    return filtered_q, filtered_a


'''
 read list of words, create index to word,
  word to index dictionaries
    return tuple( vocab->(word, count), idx2w, w2idx )

'''
def index_(tokenized_sentences, vocab_size):
    # get frequency distribution
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    # get vocabulary of 'vocab_size' most used words
    vocab = freq_dist.most_common(vocab_size)
    # index2word
    index2word = ['_'] + [UNK] + [ x[0] for x in vocab ]
    # word2index
    word2index = dict([(w,i) for i,w in enumerate(index2word)] )
    return index2word, word2index, freq_dist

'''
 filter based on number of unknowns (words not in vocabulary)
  filter out the worst sentences

'''
def filter_unk(qtokenized, atokenized, w2idx):
    data_len = len(qtokenized)

    filtered_q, filtered_a = [], []

    for qline, aline in zip(qtokenized, atokenized):
        unk_count_q = len([ w for w in qline if w not in w2idx ])
        unk_count_a = len([ w for w in aline if w not in w2idx ])
        if unk_count_a <= 2:
            if unk_count_q > 0:
                if unk_count_q/len(qline) > 0.2:
                    pass
            filtered_q.append(qline)
            filtered_a.append(aline)

    # print the fraction of the original data, filtered
    filt_data_len = len(filtered_q)
    filtered = int((data_len - filt_data_len)*100/data_len)
    print(str(filtered) + '% filtered from original data')

    return filtered_q, filtered_a




'''
 create the final dataset :
  - convert list of items to arrays of indices
  - add zero padding
      return ( [array_en([indices]), array_ta([indices]) )

'''
def zero_pad(qtokenized, atokenized, w2idx):
    # num of rows
    data_len = len(qtokenized)

    # numpy arrays to store indices
    idx_q = np.zeros([data_len, limit['maxq']], dtype=np.int32)
    idx_a = np.zeros([data_len, limit['maxa']], dtype=np.int32)

    for i in range(data_len):
        q_indices = pad_seq(qtokenized[i], w2idx, limit['maxq'])
        a_indices = pad_seq(atokenized[i], w2idx, limit['maxa'])

        #print(len(idx_q[i]), len(q_indices))
        #print(len(idx_a[i]), len(a_indices))
        idx_q[i] = np.array(q_indices)
        idx_a[i] = np.array(a_indices)

    return idx_q, idx_a


'''
 replace words with indices in a sequence
  replace with unknown if word not in lookup
    return [list of indices]

'''
def pad_seq(seq, lookup, maxlen):
    indices = []
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])
    return indices + [0]*(maxlen - len(seq))





def process_data():

    questions, answers = gather_dataset3()

    # filter out too long or too short sequences
    print('\n>> Filtrando oraciones largas')
    qlines, alines = filter_data(questions, answers)
    print('q : [{0}]; a : [{1}]'.format(qlines[0],alines[0]))

    # convert list of [lines of text] into list of [list of words ]
    print('\n>> Segment lines into words')
    qtokenized = [ [w.strip() for w in wordlist.split(' ') if w] for wordlist in qlines ]
    atokenized = [ [w.strip() for w in wordlist.split(' ') if w] for wordlist in alines ]
    print('\n:: Sample from segmented list of words')

    for q,a in zip(qtokenized[0:1], atokenized[0:1]):
        print('q : [{0}]; a : [{1}]'.format(q,a))

    # indexing -> idx2w, w2idx
    print('\n >> Index words')
    idx2w, w2idx, freq_dist = index_( qtokenized + atokenized, vocab_size=VOCAB_SIZE)

    # filter out sentences with too many unknowns
    print('\n >> Filter Unknowns')
    qtokenized, atokenized = filter_unk(qtokenized, atokenized, w2idx)
    print('\n Final dataset len : ' + str(len(qtokenized)))


    print('\n >> Zero Padding')
    idx_q, idx_a = zero_pad(qtokenized, atokenized, w2idx)

    print('\n >> Save numpy arrays to disk')
    # save them
    np.save('idx_q.npy', idx_q)
    np.save('idx_a.npy', idx_a)

    # let us now save the necessary dictionaries
    metadata = {
            'w2idx' : w2idx,
            'idx2w' : idx2w,
            'limit' : limit,
            'freq_dist' : freq_dist
                }

    # write to disk : data control dictionaries
    with open('metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

    # count of unknowns
    unk_count = (idx_q == 1).sum() + (idx_a == 1).sum()
    # count of words
    word_count = (idx_q > 1).sum() + (idx_a > 1).sum()

    print('% unknown : {0}'.format(100 * (unk_count/word_count)))
    print('Dataset count : ' + str(idx_q.shape[0]))


    return qlines, alines, idx2w, w2idx, freq_dist



if __name__ == '__main__':
   qlines, alines, idx2w, w2idx, freq_dist = process_data()

