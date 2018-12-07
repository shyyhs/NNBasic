from globalSetting import *

class EncoderRNN(nn.Module):
    def __init__(self, inputSize, hiddenSize):
        super(EncoderRNN,self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.embedding = nn.Embedding(inputSize,hiddenSize)
        self.gru = nn.GRU(hiddenSize,hiddenSize)
    def forward(self,input,hidden):
        embedded = self.embedding(input).view(1,1,-1)
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
        # here input means context vector from encoder
        embedding = self.embedding(input).view(1,1,-1)
        relu = self.relu(embedding)
        output, hidden = self.gru(relu,hidden)
        output = nn.out(output[0])
        output = nn.softmax(output)
        return output,hidden
    def initHidden(self):
        return torch.zeros(1,1,self.hiddenSize,device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self,hiddenSize,outputSize,dropout=0.1,maxLen=MAXLEN):
        # outputSize: both input and output word size
        super(AttnDecoderRNN,self).__init__()
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.dropout = dropout
        self.maxLen = maxLen
        
        self.relu = F.relu
        self.dropout = nn.Dropout(self.dropout)
        self.softmax = F.log_softmax
        self.embedding = nn.Embedding(self.outputSize,self.hiddenSize)
        self.attn = nn.Linear(self.hiddenSize*2,self.maxLen)
        self.attnCombine = nn.Linear(self.hiddenSize*2,self.hiddenSize)
        self.gru = nn.GRU(self.hiddenSize,self.hiddenSize)
        self.out = nn.Linear(self.hiddenSize,self.outputSize)

    def forward(self,input,hidden,encoderOutput):
        embedded = self.embedding(input).view(1,1,-1)
        embedded = self.dropout(embedded)
        attnWeight = self.attn(torch.cat((embedded[0],hidden[0]),1))
        attnWeight = F.softmax(attnWeight,dim=1)
        attn = torch.bmm(attnWeight.unsqueeze(0),encoderOutput.unsqueeze(0))
        attnComb = torch.cat((embedded[0],attn[0]),1)
        output = self.attnCombine(attnComb).unsqueeze(0)
        output = self.relu(output)
        output,hidden = self.gru(output,hidden)
        output = F.log_softmax(self.out(output[0]),dim=1)
        return output,hidden,attnWeight
    def initHidden(self):
        return torch.zeors(1,1,self.hiddenSize,device=device)



if (__name__=="__main__"):
    print ("Test model")

    encoder = EncoderRNN(1000,100)
    decoder = AttnDecoderRNN(1000,100)
