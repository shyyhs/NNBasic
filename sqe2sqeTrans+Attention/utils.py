from globalSetting import *

def sent2Tensor(lang,sentence):
    indexes = [lang.word2index[word] for word in sentence.split(' ')]
    indexes.append(EOSToken)
    return torch.tensor(indexes,dtype=torch.long,device=device).view(-1,1)
def pair2Tensor(pair,inputLang,outputLang):
    srcTensor = sent2Tensor(inputLang,pair[0])
    tgtTensor = sent2Tensor(outputLang,pair[1])
    return (srcTensor,tgtTensor)
def timeSince(startTime):
    return time.time()-startTime
