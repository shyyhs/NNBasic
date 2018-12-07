from globalSetting import *

class EncoderRNN(nn.Module):
    def __init__(self, inputSize, hiddenSize):
        super(EncoderRNN,self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.embedding = nn.Embedding(inputSize,hiddenSize)
        self.gru = nn.GRU(hiddenSize,hiddenSize)
    def forward(self,input,hidden):
        embedded = self.embedding(input).view(1,1-1)
        output, hidden = self.gru(embedded,hidden)
        return output,hidden
    def initHidden(self):
        return torch.zeros(1,1,self.hiddenSize,device=device)


if (__name__=="__main__"):
    print ("Test model")
    encoder = EncoderRNN(1000,100)
