import random
import math

class Cell:
    def __init__(self, weight=None, bias=None, learning=None, truely=1, momentumexc=0.9):
        self.weight = float(weight) if weight is not None else random.random()
        self.bias = float(bias) if bias is not None else (random.random()*2)-1
        self.learning = float(learning) if learning is not None else random.random()
        self.mw = 0
        self.mb = 0
        self.truely = truely
        self.momentumexc = momentumexc
    def limitation(self, limit=512):
        self.weight = max(min(self.weight, limit), 1/limit)
        self.bias = max(min(self.bias, limit), -limit)
    def process(self, input, target=None, train=True):
        input = (input*2)-1
        if target is None:
            target = input
        z = (input*self.weight)+(self.bias*self.truely)
        output = 1/(1+math.exp(-z))
        if not train:
            return output
        error = target-output
        deriv = output*(1-output)
        delta = error*deriv
        graw = delta*input
        grab = delta*self.truely
        self.mw = (self.momentumexc*self.mw)+graw
        self.mb = (self.momentumexc*self.mb)+grab
        self.weight += self.mw*self.learning
        self.bias += self.mb*self.learning
        return output

class DataCell:
    def __init__(self, maxdatac=64):
        self.outputs = []
        self.maxdatac = maxdatac
    def process(self, input, target=None, train=True, rangc=128):
        if target is None:
            target = input
        olist = self.outputs
        maxdiff = max([abs(input-dat) for dat, _ in olist]+[0])
        outputs = [0.5]
        for dat, out in olist:
            seq = (1-(abs(input-dat)/maxdiff))*rangc
            outputs += [out]*int(seq)
        output = sum(outputs)/len(outputs)
        if train:
            self.outputs = self.outputs+[[input, (output+target)/2]]
            self.outputs = self.outputs[-self.maxdatac:]
        return output

class MultiCell:
    def __init__(self, weights=None, wcount=2, bias=None, learning=None, truely=1, momentumexc=0.9):
        self.weights = list(weights) if weights is not None else [random.random() for _ in range(wcount)]
        self.bias = float(bias) if bias is not None else (random.random()*2)-1
        self.learning = float(learning) if learning is not None else random.random()
        self.wcount = len(weights) if weights is not None else int(wcount)
        self.mw = [0]*self.wcount
        self.mb = 0.0
        self.truely = truely
        self.momentumexc = momentumexc
    def limitation(self, limit=512):
        self.weight = [max(min(w, limit), 1/limit) for w in self.weights]
        self.bias = max(min(self.bias, limit), -limit)
    def process(self, inputs, target=None, train=True):
        inputs = [(i*2)-1 for i in inputs]
        if target is None:
            target = sum(inputs)/len(inputs)
        z = sum([input*self.weights[i] for i, input in enumerate(inputs)])+(self.bias*self.truely)
        output = 1/(1+math.exp(-z))
        if not train:
            return output
        error = target-output
        deriv = output*(1-output)
        delta = error*deriv
        graw = [delta*inputs[i] for i in range(self.wcount)]
        grab = delta*self.truely
        for i in range(self.wcount):
            self.mw[i] = (self.momentumexc*self.mw[i])+graw[i]
            self.weights[i] += self.mw[i]*self.learning
        self.mb = (self.momentumexc*self.mb)+grab
        self.bias += self.mb*self.learning
        return output
