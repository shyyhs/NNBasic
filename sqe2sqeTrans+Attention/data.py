from globalSetting import *

SOSToken = 0
EOSToken = 1

#Level 3.5
class Lang:
    def __init__(self, name):
        self.name = name

        self.wordN = 2
        self.index2word = {0:"SOS", 1:"EOS"}
        self.word2index = {}
        self.wordCount = {}
    def addWord(self, word):
        if (word in self.word2index):
            self.wordCount[word] += 1
            return
        self.index2word[self.wordN] = word
        self.word2index[word] = self.wordN
        self.wordCount[word] = 1
        self.wordN += 1 

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)
    
    def testPrint(self):
        print (self.wordN)
        print (self.wordCount)
        print (self.word2index)
        print (self.index2word)

# https://docs.python.org/2/library/unicodedata.html
# http://www.fileformat.info/info/unicode/category/index.htm
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD',s)
        if unicodedata.category(c)!='Mn')

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r'([.!?])',r' \1',s)
    s = re.sub(r'[^a-z.!?]+',r' ',s)
    return s;

def readLangs(lang1,lang2,reverse=False):
    print ("Reading lines")

    lines = open('data/{}-{}.txt'.format(lang1,lang2),encoding='utf-8').\
        read().strip().split('\n')
    pairs = [[normalizeString(s) for s in line.split('\t')] for line in lines]
    if reverse:
        pairs=[list(reversed(p)) for p in pairs]
        inputLang = Lang(lang2)
        outputLang = Lang(lang1)
    else:
        inputLang = Lang(lang1)
        outputLang = Lang(lang2)
    return inputLang,outputLang,pairs

MAXLEN = 10
engPrefix = ("i am", "i m", "he is", "he s", "she is", "she s","you are",
        "you re", "we are", "we re","they are", "they re")
def filterPair(p):
    return ((len(p[0].split(' '))<MAXLEN) and (len(p[1].split(' '))<MAXLEN) and p[1].startswith(engPrefix))

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1,lang2,reverse=False):
    inputLang,outputLang,pairs = readLangs(lang1,lang2,reverse)
    print ("Read {} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print ("{} pairs after filter".format(len(pairs)))
    print ("Counting words...")
    for pair in pairs:
        inputLang.addSentence(pair[0])
        outputLang.addSentence(pair[1])
    print ("Counted words")
    print (inputLang.name, inputLang.wordN)
    print (outputLang.name, outputLang.wordN)
    return inputLang,outputLang,pairs







if (__name__=="__main__"):
    print ("From Data.py begins") 
    inputLang, outputLang, pairs = prepareData('eng','fra',1)
    print (random.choice(pairs))
    """
    inputLang = Lang("Chinese")
    inputLang.addSentence("There is a a dog !")
    inputLang.testPrint()
    """
    

if (__name__!="__main__"):
    print ("Module: data loaded")
