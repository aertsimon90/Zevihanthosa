import random
import math
import numpy 

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
            self.outputs = (self.outputs+[[input, (output+target)/2]])[-self.maxdatac:]
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

class FuncCell:
    def __init__(self, maxdatac=64, range=8, rangc=16, margin=0, traindepth=2):
        self.func = ""
        self.range = range
        self.rangc = rangc
        self.outputs = []
        self.maxdatac = maxdatac
        self.margin = margin
        self.procs = ["+sumr", "*mult", "/dive", "**forc", "%rema"]
        self.traindepth = traindepth
    def comb(self, x="x"):
        return [x+h for h in self.procs]
    def combs(self, depth=1):
        listx = self.comb()
        if depth == 0:
            return listx
        else:
            listx = sum([self.comb(x) for x in listx], start=[])
            for _ in range(depth-1):
                listx = sum([self.comb(x) for x in listx], start=[])
            return listx
    def eval(self, func):
        return eval(func)
    def train(self):
        outs = [i for _, i in self.outputs]
        most = ""
        mostv = float("inf")
        nums = list(range(-self.range, self.range+1))+list(numpy.linspace(-self.range, self.range, self.rangc))
        for combsc in range(self.traindepth):
            for comb in self.combs(combsc):
                sumrs = nums if "sumr" in comb else [0]
                mults = nums if "mult" in comb else [1]
                dives = nums if "dive" in comb else [1]
                forcs = nums if "forc" in comb else [1]
                remas = list(range(-self.range, self.range)) if "rema" in comb else [float("inf")]
                for sumr in sumrs:
                    for mult in mults:
                        for dive in dives:
                            for forc in forcs:
                                for rema in remas:
                                    func = comb.replace("sumr", str(sumr)).replace("mult", str(mult)).replace("dive", str(dive)).replace("forc", str(forc)).replace("rema", str(rema))
                                    try:
                                        outs2 = [self.eval(func.replace("x", str(x))) for x, _ in self.outputs]
                                        diff = sum([abs(i-i2) for i, i2 in zip(outs2, outs)])
                                        if diff <= self.margin:
                                            self.func = func
                                            return
                                        if diff < mostv:
                                            mostv = diff
                                            most = func
                                    except:
                                        pass
        self.func = most
    def process(self, input, target=None, train=True):
        if target is None:
            target = input
        if self.func:
            output = self.eval(self.func.replace("x", str(input)))
            if train:
                self.outputs = (self.outputs+[[input, target]])[-self.maxdatac:]
                self.train()
            return output
        else:
            if train:
                self.outputs = (self.outputs+[[input, target]])[-self.maxdatac:]
                self.train()
            return target
