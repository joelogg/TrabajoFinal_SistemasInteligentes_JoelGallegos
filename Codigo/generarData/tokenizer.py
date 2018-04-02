from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
#from stop_words import get_stop_words

#stop = stopwords.words('spanish')
#stop = get_stop_words('spanish')
'''stop = stopwords.words('english')
porter = PorterStemmer()

def tokenizerOriginal(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    text = [w for w in text.split() if w not in stop]
    tokenized = [porter.stem(w) for w in text]
    return text'''
def minuscula(text):
    return text.lower()

def getTokens(text):
    text = re.sub('<[^>]*>', '', text)
    #emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower())
    text = [w for w in text.split() ]
    return text

def getSoloLetras(text):
    mitoken = getTokens(text)
    return ' '.join(mitoken)

'''
mitoken = getTokens('mi Estoy :) Joel con estando habrás las comiendo, y is a <a> testing! :-)</br>')
print( mitoken)

#print("---------------")
mitoken2 = getSoloLetras('mi Estoy :) Joel con estando habrás las comiendo, y is a <a> testing! :-)</br>')
print( mitoken2)
'''

