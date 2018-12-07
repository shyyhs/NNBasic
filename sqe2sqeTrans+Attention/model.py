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

class DecoderRNN(nn.Module):
    def __init__(self,hiddenSize,outputSize):
        super(DecoderRNN,self).__init__()
        self.hiddenSize = hiddenSize
        self.inputSize = outputSize 
        self.outputSize = outputSize
        self.embedding = nn.Embedding(outputSize, hiddenSize)
        self.relu = F.relu
        self.gru = nn.GRU(hiddenSize,hiddenSize)
        self.out = nn.Linear(hiddenSize,outputSize)
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self,input,hidden):
        embedding = self.embedding(input).view(1,1,-1)
        relu = self.relu(embedding)
        output, hidden = self.gru(relu,hidden)
        output = nn.out(output[0])
        output = nn.softmax(output)
        return output,hidden
    def initHidden(self):
        return torch.zeros(1,1,self.hiddenSize,device=device)



if (__name__=="__main__"):
    print ("Test model")

    encoder = EncoderRNN(1000,100)
    decoder = DecoderRNN(1000,100)
