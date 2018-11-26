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
        if codedata.category(c)!='Mn')

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    


if (__name__=="__main__"):
    print ("From Data.py begins") 
    """
    inputLang = Lang("Chinese")
    inputLang.addSentence("There is a a dog !")
    inputLang.testPrint()
    """
    

if (__name__!="__main__"):
    print ("Module: data loaded")
