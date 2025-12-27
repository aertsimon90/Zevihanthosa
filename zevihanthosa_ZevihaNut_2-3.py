import random
import math
import json
import numpy 
from sympy import sympify

def to01(value):
    if value == 0:
        return 0.5
    if value < 0:
        return 1-to01(-value)
    if value < 1:
        return (value/4)+0.5
    value = 1-(1/value)
    return (value/4)+0.75
def from01(value):
    if value == 0.5:
        return 0
    if value < 0.5:
        return -from01(1-value)
    if value < 0.75:
        return 4*(value-0.5)
    denom = 1-4*(value-0.75)
    return 1/denom
def nom(value, min=0, max=1):
    absmx = abs(min-max)
    return (to01(value)*absmx)+min
def denom(value, min=0, max=1):
    absmx = abs(min-max)
    return from01((value-min)/absmx)
def rmtolerance(value, sensivity=128):
    mnx = 1/sensivity
    intvalue = int(value)
    floating = value-intvalue
    if floating < mnx:
        return intvalue
    if floating > 1-mnx:
        return intvalue+1
    return value

# Activation Methods (They all produce an output between 0 and 1)
class Activation:
    def __init__(self):
        self.act = "s"
        self.sh = 1
        self.maxc = 1024
        self.zevian = nom
    def sigmoid(self, value):
        return 1/(1+math.exp(-value))
    def tanh(self, value):
        return (math.tanh(value)+1)/2
    def relu(self, value, maxc=1024):
        return min(maxc, max(0, value))/maxc
    def linear(self, value, maxc=1024):
        return ((max(min(value, maxc), -maxc)/maxc)+1)/2
    def activation(self, value):
        value = value*self.sh
        return self.sigmoid(value) if self.act=="s" else self.tanh(value) if self.act=="t" else self.zevian(value) if self.act=="z" else self.relu(value, maxc=self.maxc) if self.act=="r" else self.linear(value, maxc=self.maxc)
    def save(self):
        return [self.act, self.sh, self.maxc]
    def load(self, data):
        self.act = data[0]
        self.sh = data[1]
        self.maxc = data[2]

def save(obj):
    data = {}
    if obj.type == 0:
        data = {"t": 0, "w": obj.weight, "b": obj.bias, "l": obj.learning, "m": [obj.mw, obj.mb], "tr": obj.truely, "mx": obj.momentumexc, "a": obj.activation.save()}
    elif obj.type == 1:
        data = {"t": 1, "b": obj.bias, "l": obj.learning, "m": obj.mb, "tr": obj.truely, "mx": obj.momentumexc, "a": obj.activation.save()}
    elif obj.type == 2:
        data = {"t": 2, "w": obj.weight, "l": obj.learning, "m": obj.mw, "mx": obj.momentumexc, "a": obj.activation.save()}
    elif obj.type == 3:
        data = {"t": 3, "o": obj.outputs, "md": obj.maxdatac}
    elif obj.type == 4:
        data = {"t": 4, "w": obj.weights, "b": obj.bias, "l": obj.learning, "wc": obj.wcount, "m": [obj.mw, obj.mb], "tr": obj.truely, "mx": obj.momentumexc, "a": obj.activation.save()}
    elif obj.type == 5:
        data = {"t": 5, "w": obj.weights, "b": obj.biases, "l": obj.learning, "pc": obj.pcount, "m": [obj.mw, obj.mb], "tr": obj.truely, "mx": obj.momentumexc, "a": obj.activation.save()}
    elif obj.type == 6:
        data = {"t": 6, "f": obj.func, "r": obj.range, "rc": obj.rangc, "o": obj.outputs, "md": obj.maxdatac, "m": obj.margin, "pr": obj.procs, "ts": obj.trainstart, "td": obj.traindepth, "n": obj.nums, "en": obj.exnums}
    elif obj.type == 7:
        data = {"t": 7, "w": obj.weights, "b": obj.biases, "l": obj.learning, "ic": obj.icount, "oc": obj.ocount, "m": [obj.mw, obj.mb], "tr": obj.truely, "mx": obj.momentumexc, "a": obj.activation.save()}
    return json.dumps(data)
def load(data):
    data = json.loads(data)
    t = data.get("t", None)
    obj = None
    if t == 0:
        obj = Cell(weight=data["w"], bias=data["b"], learning=data["l"], truely=data["tr"], momentumexc=data["mx"])
        obj.activation.load(data["a"])
        obj.mw = data["m"][0]
        obj.mb = data["m"][1]
    elif t == 1:
        obj = LinearSumCell(bias=data["b"], learning=data["l"], truely=data["tr"], momentumexc=data["mx"])
        obj.activation.load(data["a"])
        obj.mb = data["m"]
    elif t == 2:
        obj = LinearMulCell(weight=data["w"], learning=data["l"], momentumexc=data["mx"])
        obj.activation.load(data["a"])
    elif t == 3:
        obj = DataCell()
        obj.outputs = data["o"]
        obj.maxdatac = data["maxdatac"]
    elif t == 4:
        obj = MultiInputCell(weights=data["w"], bias=data["b"], learning=data["l"], wcount=data["wc"], truely=data["tr"], momentumexc=data["mx"])
        obj.activation.load(data["a"])
        obj.mw = data["m"][0]
        obj.mb = data["m"][1]
    elif t == 5:
        obj = MultiOutputCell(weights=data["w"], biases=data["b"], learning=data["l"], pcount=data["pc"], truely=data["tr"], momentumexc=data["mx"])
        obj.activation.load(data["a"])
        obj.mw = data["m"][0]
        obj.mb = data["m"][1]
    elif t == 6:
        obj = FuncCell(maxdatac=data["md"], range=data["r"], rangc=data["rc"], margin=data["m"], trainstart=data["ts"], traindepth=data["td"])
        obj.func = data["f"]
        obj.outputs = data["o"]
        obj.procs = data["pr"]
        obj.nums = data["n"]
        obj.exnums = data["en"]
    elif t == 7:
        obj = MultiCell(weights=data["w"], biases=data["b"], learning=data["l"], icount=data["ic"], ocount=data["oc"], truely=data["tr"], momentumexc=data["mx"])
        obj.activation.load(data["a"])
        obj.mw = data["m"][0]
        obj.mb = data["m"][1]
    return obj
def copy(obj):
    return load(save(obj))

class Cell:
    def __init__(self, weight=None, bias=None, learning=None, truely=1, momentumexc=0.9):
        self.weight = float(weight) if weight is not None else random.random()
        self.bias = float(bias) if bias is not None else (random.random()*2)-1
        self.learning = float(learning) if learning is not None else random.random()
        self.mw = 0
        self.mb = 0
        self.truely = truely
        self.momentumexc = momentumexc
        self.activation = Activation()
        self.type = 0
    def limitation(self, limit=512):
        self.weight = max(min(self.weight, limit), -limit)
        self.bias = max(min(self.bias, limit), -limit)
    def process(self, input, target=None, train=True, perceptron=False):
        input = (input*2)-1
        if target is None:
            target = input
        z = (input*self.weight)+(self.bias*self.truely)
        output = self.activation.activation(z)
        if not train:
            return output
        error = target-output
        if perceptron:
            self.mw = (self.momentumexc*self.mw)+(error*input)
            self.mb = (self.momentumexc*self.mb)+error
            self.weight += self.mw*self.learning
            self.bias += self.mb*self.learning
        else:
            deriv = output*(1-output)
            delta = error*deriv
            graw = delta*input
            grab = delta*self.truely
            self.mw = (self.momentumexc*self.mw)+graw
            self.mb = (self.momentumexc*self.mb)+grab
            self.weight += self.mw*self.learning
            self.bias += self.mb*self.learning
        return output

class LinearSumCell:
    def __init__(self, bias=None, learning=None, truely=1, momentumexc=0.9):
        self.bias = float(bias) if bias is not None else (random.random()*2)-1
        self.learning = float(learning) if learning is not None else random.random()
        self.mb = 0
        self.truely = truely
        self.momentumexc = momentumexc
        self.activation = Activation()
        self.type = 1
    def limitation(self, limit=512):
        self.bias = max(min(self.bias, limit), -limit)
    def process(self, input, target=None, train=True, perceptron=False):
        input = (input*2)-1
        if target is None:
            target = input
        z = (input)+(self.bias*self.truely)
        output = self.activation.activation(z)
        if not train:
            return output
        error = target-output
        if perceptron:
            self.mb = (self.momentumexc*self.mb)+error
            self.bias += self.mb*self.learning
        else:
            deriv = output*(1-output)
            delta = error*deriv
            grab = delta*self.truely
            self.mb = (self.momentumexc*self.mb)+grab
            self.bias += self.mb*self.learning
        return output

class LinearMulCell:
    def __init__(self, weight=None, learning=None, momentumexc=0.9):
        self.weight = float(weight) if weight is not None else random.random()
        self.learning = float(learning) if learning is not None else random.random()
        self.mw = 0
        self.momentumexc = momentumexc
        self.activation = Activation()
        self.type = 2
    def limitation(self, limit=512):
        self.weight = max(min(self.weight, limit), -limit)
    def process(self, input, target=None, train=True, perceptron=False):
        input = (input*2)-1
        if target is None:
            target = input
        z = (input*self.weight)
        output = self.activation.activation(z)
        if not train:
            return output
        error = target-output
        if perceptron:
            self.mw = (self.momentumexc*self.mw)+(error*input)
            self.weight += self.mw*self.learning
        else:
            deriv = output*(1-output)
            delta = error*deriv
            graw = delta*input
            self.mw = (self.momentumexc*self.mw)+graw
            self.weight += self.mw*self.learning
        return output

class DataCell:
    def __init__(self, maxdatac=64):
        self.outputs = []
        self.maxdatac = maxdatac
        self.type = 3
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

class MultiInputCell:
    def __init__(self, weights=None, wcount=2, bias=None, learning=None, truely=1, momentumexc=0.9):
        self.weights = list(weights) if weights is not None else [random.random() for _ in range(wcount)]
        self.bias = float(bias) if bias is not None else (random.random()*2)-1
        self.learning = float(learning) if learning is not None else random.random()
        self.wcount = len(weights) if weights is not None else int(wcount)
        self.mw = [0]*self.wcount
        self.mb = 0.0
        self.truely = truely
        self.momentumexc = momentumexc
        self.activation = Activation()
        self.type = 4
    def limitation(self, limit=512):
        self.weights = [max(min(w, limit), -limit) for w in self.weights]
        self.bias = max(min(self.bias, limit), -limit)
    def process(self, inputs, target=None, train=True, perceptron=False):
        inputs = [(i*2)-1 for i in inputs]
        if target is None:
            target = sum(inputs)/len(inputs)
        z = sum([input*self.weights[i] for i, input in enumerate(inputs)])+(self.bias*self.truely)
        output = self.activation.activation(z)
        if not train:
            return output
        error = target-output
        if perceptron:
            for i in range(self.wcount):
                self.mw[i] = (self.momentumexc*self.mw[i])+(error*inputs[i])
                self.weights[i] += self.mw[i]*self.learning
            self.mb = (self.momentumexc*self.mb)+error
            self.bias += self.mb*self.learning
        else:
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

class MultiOutputCell:
    def __init__(self, weights=None, biases=None, pcount=3, learning=None, truely=1, momentumexc=0.9):
        self.weights = weights if weights is not None else [random.random() for _ in range(pcount)]
        self.biases = biases if biases is not None else [(random.random()*2)-1 for _ in range(pcount)]
        self.learning = float(learning) if learning is not None else random.random()
        self.pcount = len(weights) if weights else int(pcount)
        self.mw = [0]*self.pcount
        self.mb = [0]*self.pcount
        self.truely = truely
        self.momentumexc = momentumexc
        self.activation = Activation()
        self.type = 5
    def limitation(self, limit=512):
        self.weights = [max(min(w, limit), -limit) for w in self.weights]
        self.biases = [max(min(b, limit), -limit) for b in self.biases]
    def process(self, input, target=None, train=True, perceptron=False):
        input = (input*2)-1
        if target is None:
            target = [input]*self.pcount
        outs = []
        for i in range(self.pcount):
            x = (input*self.weights[i])+(self.biases[i]*self.truely)
            output = self.activation.activation(x)
            outs.append(output)
            if train:
                error = target[i]-output
                if perceptron:
                    self.mw[i] = (self.momentumexc*self.mw[i])+(error*input)
                    self.weights[i] += self.mw[i]*self.learning
                    self.mb[i] = (self.momentumexc*self.mb[i])+error
                    self.biases[i] += self.mb[i]*self.learning
                else:
                    deriv = output*(1-output)
                    delta = error*deriv
                    graw = delta*input
                    grab = delta*self.truely
                    self.mw[i] = (self.momentumexc*self.mw[i])+graw
                    self.weights[i] += self.mw[i]*self.learning
                    self.mb[i] = (self.momentumexc*self.mb[i])+grab
                    self.biases[i] += self.mb[i]*self.learning
        return outs

class FuncCell:
    def __init__(self, maxdatac=64, range=8, rangc=16, margin=0, trainstart=0, traindepth=2):
        self.func = ""
        self.range = range
        self.rangc = rangc
        self.outputs = []
        self.maxdatac = maxdatac
        self.margin = margin
        self.procs = ["+sumr", "*mult", "/dive", "**forc", "%rema"]
        self.trainstart = trainstart
        self.traindepth = traindepth
        self.nums = []
        self.exnums = [math.pi, math.e]
        self.type = 6
    def comb(self, x="x"):
        return [x+h for h in self.procs]
    def comb2(self, x=[]):
        lx = x if x else self.comb()
        lx2 = []
        nums = self.nums if self.nums else list(range(-self.range, self.range+1))+list(numpy.linspace(-self.range, self.range, self.rangc))+self.exnums
        for comb in lx:
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
                                comb2 = comb.replace("sumr", str(sumr)).replace("mult", str(mult)).replace("dive", str(dive)).replace("forc", str(forc)).replace("rema", str(rema))
                                lx2.append(comb2)
        return lx2
    def combs(self, depth=1):
        lx = self.comb2()
        if depth == 0:
            return lx
        else:
            for _ in range(depth):
                lx = sum([self.comb(x) for x in lx], start=[])
                lx = self.comb2(lx)
            return lx
    def eval(self, func):
        return eval(func)
    def train(self):
        outs = [i for _, i in self.outputs]
        most = ""
        mostv = float("inf")
        for combsc in range(self.trainstart, self.traindepth):
            for func in self.combs(combsc):
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

class MultiCell:
    def __init__(self, weights=None, biases=None, icount=2, ocount=2, learning=None, truely=1, momentumexc=0.9):
        self.icount = len(biases) if biases is not None else int(icount)
        self.ocount = len(weights[0]) if weights is not None else int(ocount)
        self.weights = weights if weights is not None else [[random.random() for _ in range(self.icount)] for _ in range(self.ocount)]
        self.biases = biases if biases is not None else [(random.random()*2)-1 for _ in range(self.ocount)]
        self.learning = float(learning) if learning is not None else random.random()
        self.truely = truely
        self.momentumexc = momentumexc
        self.mw = [[0]*self.icount]*self.ocount
        self.mb = [0]*self.ocount
        self.activation = Activation()
        self.type = 7
    def limitation(self, limit=512):
        for i in range(self.ocount):
            self.weights[i] = [max(min(w, limit), -limit) for w in self.weights[i]]
            self.biases[i] = max(min(self.biases[i], limit), -limit)
    def process(self, input, target=None, train=True, perceptron=False):
        input = [(i*2)-1 for i in input]
        if target is None:
            avg = sum(input)/len(input)
            target = [avg]*self.ocount
        outs = []
        for i in range(self.ocount):
            x = sum(input[j]*self.weights[i][j] for j in range(self.icount))+(self.biases[i]*self.truely)
            output = self.activation.activation(x)
            outs.append(output)
            if train:
                error = target[i]-output
                if perceptron:
                    for j in range(self.icount):
                        self.mw[i][j] = (self.momentumexc*self.mw[i][j])+(error*input[j])
                        self.weights[i][j] += self.mw[i][j]*self.learning
                    self.mb[i] = (self.momentumexc*self.mb[i])+error
                    self.biases[i] += self.mb[i]*self.learning
                else:
                    deriv = output*(1-output)
                    delta = error*deriv
                    for j in range(self.icount):
                        graw = delta*input[j]
                        self.mw[i][j] = (self.momentumexc*self.mw[i][j])+graw
                        self.weights[i][j] += self.mw[i][j]*self.learning
                    grab = delta*self.truely
                    self.mb[i] = (self.momentumexc*self.mb[i])+grab
                    self.biases[i] += self.mb[i]*self.learning
        return outs
