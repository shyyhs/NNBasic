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
    
