from globalSetting import *
from utils import *
from data import *
from model import *

teacherForcingRatio = 0.5

def train(srcTensor,tgtTensor,encoder,decoder,encoderOpt,decoderOpt,criterion,maxLen=MAXLEN):
    encoderHidden = encoder.initHidden()
    srcLen = srcTensor.size(0)
    tgtLen = tgtTensor.size(0)
    encoderOutputs = torch.zeros(maxLen,encoder.hiddenSize)

    loss=0

    for ei in range(srcLen):
        encoderOutput,encoderHidden = encoder(srcTensor[ei],encoderHidden)
        encoderOutputs[ei]=encoderOutput[0,0]
    decoderInput = torch.tensor([[SOSToken]],device=device)
    decoderHidden=encoderHidden
    teacherForcing=True if random.random()< teacherForcingRatio else False
    if (teacherForcing):
        for di in range(tgtLen):
            decoderOutput,decoderHidden,decoderAttn = decoder(decoderInput,\
                    decoderHidden,encoderOutputs)
            loss +=criterion(decoderOutput,tgtTensor[di])
            decoderInput = tgtTensor[di]
    else:
        for di in range(tgtLen):
            decoderOutput,decoderHidden,decoderAttn = decoder(decoderInput,\
                    decoderHidden,encoderOutputs)
            topv,topi = decoderOutput.topk(1)
            decoderInput = topi.squeeze.detach()
            loss +=criterion(decoderOutput,tgtTensor[di])
            if (decoderInput.item()==EOSToken): break

    encoderOpt.zero_grad()
    decoderOpt.zero_grad()
    loss.backward()
    encoderOpt.step()
    decoderOpt.step()
    return loss.item()/tgtLen
    
def trainIters(encoder,decoder,iterN,printEvery=1000,plotEvery=100,lr=0.01):
    startTime = time.time()
    plotLosses = []
    plotLossTotal=0
    printLossTotal=0
    encoderOpt = optim.SGD(encoder.parameters(),lr=lr)
    decoderOpt = optim.SGD(decoder.parameters(),lr=lr)
    criterion = nn.NLLLoss()
    trainPairs = [pair2Tensor(random.choice(pairs)) for i in range(iterN)]
    for i in range(iterN):
        trainPair = trainPairs[i]
        srcTensor = trainPair[0]
        tgtTensor = trainPair[1]
        loss = train(srcTensor,tgtTensor,encoder,decoder,encoderOpt,decoderOpt,\
                criterion)
        printLossTotal+=loss
        plotLossTotal+=loss
        if ((i+1)%printEvery==0):
            printLossAvg = printLossTotal/printEvery
            printLossTotal=0
            print ("{} {} {:.4f}".format(timeSince(startTime),i+1,printLossAvg))
        if ((i+1)%plotEvery==0):
            plotLossAvg = plotLossTotal/plotEvery
            plotLosses.append(plotLossAvg)
            plotLossTotal=0
    showPlot(plotLosses)



