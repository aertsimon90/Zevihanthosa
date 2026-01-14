# Zevihanthosa - ZevihaNut/2.9
import random
import math
import json
import ast
import operator as op

OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}
MIN_DENOM = 1e-308

def local_sympify(expr: str):
    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        elif isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            return OPS[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):
            return OPS[type(node.op)](_eval(node.operand))
        else:
            raise ValueError(f"Unsupported: {ast.dump(node)}")
    tree = ast.parse(expr, mode="eval")
    return _eval(tree)
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
    denom = max(denom, MIN_DENOM)
    return 1/denom
def nom(value, min=0, max=1):
    absmx = abs(min-max)
    return (to01(value)*absmx)+min
def denom(value, min=0, max=1):
    absmx = abs(min-max)
    return from01((value-min)/absmx)
def rmtolerance(value, sensitivity=128):
    if value < 0:
        return -rmtolerance(-value, sensitivity=sensitivity)
    mnx = 1/sensitivity
    intvalue = int(value)
    floating = value-intvalue
    if floating < mnx:
        return intvalue
    if floating > 1-mnx:
        return intvalue+1
    return value
def quantization(value, sensitivity=256):
    return int(value*sensitivity)/(sensitivity)
def linspace_sing(rang):
    return [0]+[(i+1)/rang for i in range(rang)]
def linspace(minx, maxx, rang):
    if rang <= 1:
        return [minx]
    step = (maxx - minx) / (rang - 1)
    return [minx + i * step for i in range(rang)]
def distance_weighted_ratio(x, sharpness=2):
    return max((x-(1-(1/sharpness))), 0)*sharpness
def clustmin(x, k=32):
    if x > 0.5:
        x = 1-x
        x = quantization(x+(1/k), sensitivity=k)
        return (1-x)-(1/k)
    else:
        return quantization(x, sensitivity=k)
def clustmax(x, k=32):
    return min(clustmin(x)+(1/(k/2)), 1)
def sfcmx(valuec, k=32):
    values = valuec*[1]
    val = clustmin(values[0])
    for i, value in enumerate(values[1:]):
        value = clustmax(value)
        val += value/(k**(i+1))
    return val
def squeezeforclust(values, k=32):
    n = len(values)
    val = clustmin(values[0])
    for i, value in enumerate(values[1:]):
        value = clustmax(value)
        val += value/(k**(i+1))
    return val/sfcmx(n,k=k)

# Activation Methods (They all produce an output between 0 and 1)
class Activation:
    def __init__(self, act="s"):
        self.act = act
        self.sh = 1
        self.maxc = 512
        self.zevian = nom
    def sigmoid(self, value):
        return 1/(1+math.exp(-value))
    def softsign(self, value):
        return (value/(1+abs(value))+1)/2
    def tanh(self, value):
        return (math.tanh(value)+1)/2
    def linear(self, value, maxc=512):
        return ((max(min(value, maxc), -maxc)/maxc)+1)/2
    def plinear(self, value, maxc=512):
        return (self.linear(max(value, 0), maxc=maxc)-0.5)*2
    def nlinear(self, value, maxc=512):
        return 1-self.linear(min(value, 0), maxc=maxc)*2
    def swish(self, value):
        sw = value*self.sigmoid(value)
        return (sw + 0.3) / (sw + 5.3)
    def activation(self, value):
        value = value*self.sh
        try:
            if self.act == "s":
                value = self.sigmoid(value)
            elif self.act == "ss":
                value = self.softsign(value)
            elif self.act == "z":
                value = self.zevian(value)
            elif self.act == "t":
                value = self.tanh(value)
            elif self.act == "l":
                value = self.linear(value, maxc=self.maxc)
            elif self.act == "pl":
                value = self.plinear(value, maxc=self.maxc)
            elif self.act == "nl":
                value = self.nlinear(value, maxc=self.maxc)
            elif self.act == "sw":
                value = self.swish(value)
            return value
        except OverflowError:
            return 1 if value > 0 else 0
    def derivation(self, output, input_value=None):
        if self.act in ["s", "z"]:
            return output * (1 - output)
        elif self.act == "t":
            t = 2 * output - 1
            return (1 - t * t)
        elif self.act == "ss":
            if input_value is not None:
                return 1.0 / (1.0 + abs(input_value))**2
            else:
                scaled = 2 * output - 1
                return 1.0 / (1.0 + abs(scaled) * 4)**2 
        elif self.act == "l":
            base_slope = 1.0 / (self.maxc * 2)
            if output < 1e-6 or output > 1 - 1e-6:
                return 0.0
            return base_slope
        elif self.act == "pl":
            base_slope = 1.0 / (self.maxc * 2)
            if output < 1e-6 or output > 1 - 1e-6:
                return 0.0
            if output > 0.99:
                return 0.0
            return base_slope
        elif self.act == "nl":
            base_slope = 1.0 / (self.maxc * 2)
            if output < 1e-6 or output > 1 - 1e-6:
                return 0.0
            if output < 0.01:
                return 0.0
            return base_slope
        elif self.act == "sw":
            y = output
            if y <= 0.0 or y >= 1.0:
                return 0.0
            if y < 0.01 or y > 0.99:
                return 0.0
            deriv = y * (1.0 + (1.0 - y) * (1.0 - 2.0 * y))
            scale_factor = 0.165 
            return deriv * scale_factor
        else:
            return 1.0 
    def normalize(self, value): # normalize for [-1,1] processing
        value = min(max(value, 0), 1)
        return value if self.act=="n" else (value*2)-1 # n=none
    def save(self):
        return [self.act, self.sh, self.maxc]
    def load(self, data):
        self.act = data[0]
        self.sh = data[1]
        self.maxc = data[2]

class DivideCluster:
    def __init__(self, data=[], count=16):
        self.count = count
        self.data = data
    def add_data(self, floatn):
        self.data.append(floatn)
    def result(self):
        minn = min(self.data)
        maxx = max(self.data)
        clusters = {}
        for i in range(self.count):
            clusters[i] = []
        if len(self.data) == 1:
            clusters[int(len(clusters)/2)] = [self.data[0]]
            return clusters
        for data in self.data:
            ratio = (data-minn)/(maxx-minn)
            clusters[min(int(ratio*self.count), self.count-1)].append(data)
        return clusters

def save(obj, wantdict=False):
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
    elif obj.type == 8:
        data = {"t": 8, "w": obj.weight, "b": obj.bias, "l": obj.learning, "tr": obj.truely, "a": obj.activation.save()}
    elif obj.type in [9, 10, 11, 12]:
        data = obj.savedict()
    if not wantdict:
        data = json.dumps(data)
    return data
def load(data, wantdict=False):
    if not wantdict:
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
    elif t == 8:
        obj = NanoCell(weight=data["w"], bias=data["b"], learning=data["l"], truely=data["tr"])
        obj.activation.load(data["a"])
    elif t == 9:
        obj = CellForest(cellscount=0)
        obj.loaddict(data)
    elif t == 10:
        obj = MultiCellForest(cellscount=0)
        obj.loaddict(data)
    elif t == 11:
        obj = CellNetwork(layers=[0])
        obj.loaddict(data)
    elif t == 12:
        obj = CellNetworkForest(layers=[0])
        obj.loaddict(data)
    return obj
def copy(obj):
    return load(save(obj))

class Cell:
    def __init__(self, weight=None, bias=None, learning=None, truely=1, momentumexc=0.9):
        self.weight = float(weight) if weight is not None else (random.random()*2)-1
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
    def forget(self):
        self.mw = 0
        self.mb = 0
    def rmtolerance(self, sensitivity=128):
        self.weight = rmtolerance(self.weight, sensitivity=sensitivity)
        self.bias = rmtolerance(self.bias, sensitivity=sensitivity)
        self.learning = max(rmtolerance(self.learning, sensitivity=sensitivity), 1/sensitivity)
        self.truely = max(rmtolerance(self.truely, sensitivity=sensitivity), 1/sensitivity)
    def quantization(self, sensitivity=128):
        self.weight = quantization(self.weight, sensitivity=sensitivity)
        self.bias = quantization(self.bias, sensitivity=sensitivity)
        self.learning = max(quantization(self.learning, sensitivity=sensitivity), 1/sensitivity)
        self.truely = max(quantization(self.truely, sensitivity=sensitivity), 1/sensitivity)
    def process(self, input, target=None, train=True, trainrate=1, perceptron=False):
        input = self.activation.normalize(input)
        if target is None:
            target = input
        z = (input*self.weight)+(self.bias*self.truely)
        output = self.activation.activation(z)
        if not train:
            return output
        error = target-output
        if perceptron:
            self.mw = (self.momentumexc*self.mw)+((error*input)*trainrate)
            self.mb = (self.momentumexc*self.mb)+(error*trainrate)
            self.weight += self.mw*self.learning
            self.bias += self.mb*self.learning
        else:
            deriv = self.activation.derivation(output)
            delta = error*deriv
            graw = delta*input
            grab = delta*self.truely
            self.mw = (self.momentumexc*self.mw)+(graw*trainrate)
            self.mb = (self.momentumexc*self.mb)+(grab*trainrate)
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
    def forget(self):
        self.mb = 0
    def rmtolerance(self, sensitivity=128):
        self.bias = rmtolerance(self.bias, sensitivity=sensitivity)
        self.learning = max(rmtolerance(self.learning, sensitivity=sensitivity), 1/sensitivity)
        self.truely = max(rmtolerance(self.truely, sensitivity=sensitivity), 1/sensitivity)
    def quantization(self, sensitivity=128):
        self.bias = quantization(self.bias, sensitivity=sensitivity)
        self.learning = max(quantization(self.learning, sensitivity=sensitivity), 1/sensitivity)
        self.truely = max(quantization(self.truely, sensitivity=sensitivity), 1/sensitivity)
    def process(self, input, target=None, train=True, trainrate=1, perceptron=False):
        input = self.activation.normalize(input)
        if target is None:
            target = input
        z = (input)+(self.bias*self.truely)
        output = self.activation.activation(z)
        if not train:
            return output
        error = target-output
        if perceptron:
            self.mb = (self.momentumexc*self.mb)+(error*trainrate)
            self.bias += self.mb*self.learning
        else:
            deriv = self.activation.derivation(output)
            delta = error*deriv
            grab = delta*self.truely
            self.mb = (self.momentumexc*self.mb)+(grab*trainrate)
            self.bias += self.mb*self.learning
        return output

class LinearMulCell:
    def __init__(self, weight=None, learning=None, momentumexc=0.9):
        self.weight = float(weight) if weight is not None else (random.random()*2)-1
        self.learning = float(learning) if learning is not None else random.random()
        self.mw = 0
        self.momentumexc = momentumexc
        self.activation = Activation()
        self.type = 2
    def limitation(self, limit=512):
        self.weight = max(min(self.weight, limit), -limit)
    def forget(self):
        self.mw = 0
    def rmtolerance(self, sensitivity=128):
        self.weight = rmtolerance(self.weight, sensitivity=sensitivity)
        self.learning = max(rmtolerance(self.learning, sensitivity=sensitivity), 1/sensitivity)
    def quantization(self, sensitivity=128):
        self.weight = quantization(self.weight, sensitivity=sensitivity)
        self.learning = max(quantization(self.learning, sensitivity=sensitivity), 1/sensitivity)
    def process(self, input, target=None, train=True, trainrate=1, perceptron=False):
        input = self.activation.normalize(input)
        if target is None:
            target = input
        z = (input*self.weight)
        output = self.activation.activation(z)
        if not train:
            return output
        error = target-output
        if perceptron:
            self.mw = (self.momentumexc*self.mw)+((error*input)*trainrate)
            self.weight += self.mw*self.learning
        else:
            deriv = self.activation.derivation(output)
            delta = error*deriv
            graw = delta*input
            self.mw = (self.momentumexc*self.mw)+(graw*trainrate)
            self.weight += self.mw*self.learning
        return output

class DataCell:
    def __init__(self, maxdatac=64):
        self.data = []
        self.maxdatac = maxdatac
        self.type = 3
    def process(self, input, target=None, train=True, rangc=128, fallbackavg=True, average=True, avgdirect=True):
        if target is None:
            target = input
        dataset = list(self.data)
        if average:
            outputs = []
            maxdiff = max([abs(input-inp) for inp, _ in (dataset+[(0, [0])])])
            for inp, out in dataset:
                seq = 1-(abs(inp-input)/maxdiff)
                outputs += out*int(seq*rangc)
            if fallbackavg:
                outputs = outputs if outputs else sum([out for _, out in dataset], start=[])
            outputs = outputs if outputs else [0.5]
            output = sum(outputs)/len(outputs)
        else:
            output = 0.5
            outputbs = float("inf")
            for inp, out in dataset:
                if avgdirect:
                    out = sum(out)/len(out)
                else:
                    out = out[-1]
                bs = abs(inp-input)
                if bs < outputbs:
                    output = out
                    outputbs = bs
        if train:
            self.data += [(input, [target])]
            self.data = self.data[-self.maxdatac:]
        return output

class MultiInputCell:
    def __init__(self, weights=None, wcount=2, bias=None, learning=None, truely=1, momentumexc=0.9):
        self.weights = list(weights) if weights is not None else [(random.random()*2)-1 for _ in range(wcount)]
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
    def forget(self):
        self.mw = [0]*self.wcount
        self.mb = 0.0
    def rmtolerance(self, sensitivity=128):
        self.weights = [rmtolerance(w, sensitivity=sensitivity) for w in self.weights]
        self.bias = rmtolerance(self.bias, sensitivity=sensitivity)
        self.learning = max(rmtolerance(self.learning, sensitivity=sensitivity), 1/sensitivity)
        self.truely = max(rmtolerance(self.truely, sensitivity=sensitivity), 1/sensitivity)
    def quantization(self, sensitivity=128):
        self.weights = [quantization(w, sensitivity=sensitivity) for w in self.weights]
        self.bias = quantization(self.bias, sensitivity=sensitivity)
        self.learning = max(quantization(self.learning, sensitivity=sensitivity), 1/sensitivity)
        self.truely = max(quantization(self.truely, sensitivity=sensitivity), 1/sensitivity)
    def process(self, inputs, target=None, train=True, trainrate=1, perceptron=False):
        inputs = [self.activation.normalize(i) for i in inputs]
        if target is None:
            target = sum(inputs)/len(inputs)
        z = sum([input*self.weights[i] for i, input in enumerate(inputs)])+(self.bias*self.truely)
        output = self.activation.activation(z)
        if not train:
            return output
        error = target-output
        if perceptron:
            for i in range(self.wcount):
                self.mw[i] = (self.momentumexc*self.mw[i])+((error*inputs[i])*trainrate)
                self.weights[i] += self.mw[i]*self.learning
            self.mb = (self.momentumexc*self.mb)+(error*trainrate)
            self.bias += self.mb*self.learning
        else:
            deriv = self.activation.derivation(output)
            delta = error*deriv
            graw = [delta*inputs[i] for i in range(self.wcount)]
            grab = delta*self.truely
            for i in range(self.wcount):
                self.mw[i] = (self.momentumexc*self.mw[i])+(graw[i]*trainrate)
                self.weights[i] += self.mw[i]*self.learning
            self.mb = (self.momentumexc*self.mb)+(grab*trainrate)
            self.bias += self.mb*self.learning
        return output

class MultiOutputCell:
    def __init__(self, weights=None, biases=None, pcount=3, learning=None, truely=1, momentumexc=0.9):
        self.weights = weights if weights is not None else [(random.random()*2)-1 for _ in range(pcount)]
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
    def forget(self):
        self.mw = [0]*self.pcount
        self.mb = [0]*self.pcount
    def rmtolerance(self, sensitivity=128):
        self.weights = [rmtolerance(w, sensitivity=sensitivity) for w in self.weights]
        self.biases = [rmtolerance(b, sensitivity=sensitivity) for b in self.biases]
        self.learning = max(rmtolerance(self.learning, sensitivity=sensitivity), 1/sensitivity)
        self.truely = max(rmtolerance(self.truely, sensitivity=sensitivity), 1/sensitivity)
    def quantization(self, sensitivity=128):
        self.weights = [quantization(w, sensitivity=sensitivity) for w in self.weights]
        self.biases = [quantization(b, sensitivity=sensitivity) for b in self.biases]
        self.learning = max(quantization(self.learning, sensitivity=sensitivity), 1/sensitivity)
        self.truely = max(quantization(self.truely, sensitivity=sensitivity), 1/sensitivity)
    def process(self, input, target=None, train=True, trainrate=1, perceptron=False):
        input = self.activation.normalize(input)
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
                    self.mw[i] = (self.momentumexc*self.mw[i])+((error*input)*trainrate)
                    self.weights[i] += self.mw[i]*self.learning
                    self.mb[i] = (self.momentumexc*self.mb[i])+(error*trainrate)
                    self.biases[i] += self.mb[i]*self.learning
                else:
                    deriv = self.activation.derivation(output)
                    delta = error*deriv
                    graw = delta*input
                    grab = delta*self.truely
                    self.mw[i] = (self.momentumexc*self.mw[i])+(graw*trainrate)
                    self.weights[i] += self.mw[i]*self.learning
                    self.mb[i] = (self.momentumexc*self.mb[i])+(grab*trainrate)
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
        nums = self.nums if self.nums else list(range(-self.range, self.range+1))+list(linspace(-self.range, self.range, self.rangc))+self.exnums
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
        return local_sympify(func)
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
        self.weights = weights if weights is not None else [[(random.random()*2)-1 for _ in range(self.icount)] for _ in range(self.ocount)]
        self.biases = biases if biases is not None else [(random.random()*2)-1 for _ in range(self.ocount)]
        self.learning = float(learning) if learning is not None else random.random()
        self.truely = truely
        self.momentumexc = momentumexc
        self.mw = [[0.0]*self.icount for _ in range(self.ocount)]
        self.mb = [0]*self.ocount
        self.activation = Activation()
        self.type = 7
    def limitation(self, limit=512):
        for i in range(self.ocount):
            self.weights[i] = [max(min(w, limit), -limit) for w in self.weights[i]]
            self.biases[i] = max(min(self.biases[i], limit), -limit)
    def forget(self):
        self.mw = [[0.0]*self.icount for _ in range(self.ocount)]
        self.mb = [0]*self.ocount
    def rmtolerance(self, sensitivity=128):
        for i in range(self.ocount):
            self.weights[i] = [rmtolerance(w, sensitivity=sensitivity) for w in self.weights[i]]
            self.biases[i] = rmtolerance(self.biases[i], sensitivity=sensitivity)
        self.learning = max(rmtolerance(self.learning, sensitivity=sensitivity), 1/sensitivity)
        self.truely = max(rmtolerance(self.truely, sensitivity=sensitivity), 1/sensitivity)
    def quantization(self, sensitivity=128):
        for i in range(self.ocount):
            self.weights[i] = [quantization(w, sensitivity=sensitivity) for w in self.weights[i]]
            self.biases[i] = quantization(self.biases[i], sensitivity=sensitivity)
        self.learning = max(quantization(self.learning, sensitivity=sensitivity), 1/sensitivity)
        self.truely = max(quantization(self.truely, sensitivity=sensitivity), 1/sensitivity)
    def process(self, input, target=None, train=True, trainrate=1, perceptron=False):
        input = [self.activation.normalize(i) for i in input]
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
                        self.mw[i][j] = (self.momentumexc*self.mw[i][j])+((error*input[j])*trainrate)
                        self.weights[i][j] += self.mw[i][j]*self.learning
                    self.mb[i] = (self.momentumexc*self.mb[i])+(error*trainrate)
                    self.biases[i] += self.mb[i]*self.learning
                else:
                    deriv = self.activation.derivation(output)
                    delta = error*deriv
                    for j in range(self.icount):
                        graw = delta*input[j]
                        self.mw[i][j] = (self.momentumexc*self.mw[i][j])+(graw*trainrate)
                        self.weights[i][j] += self.mw[i][j]*self.learning
                    grab = delta*self.truely
                    self.mb[i] = (self.momentumexc*self.mb[i])+(grab*trainrate)
                    self.biases[i] += self.mb[i]*self.learning
        return outs

class NanoCell:
    def __init__(self, weight=None, bias=None, learning=None, truely=1):
        self.weight = float(weight) if weight is not None else (random.random()*2)-1
        self.bias = float(bias) if bias is not None else (random.random()*2)-1
        self.learning = float(learning) if learning is not None else random.random()
        self.truely = truely
        self.activation = Activation()
        self.type = 8
    def limitation(self, limit=512):
        self.weight = max(min(self.weight, limit), -limit)
        self.bias = max(min(self.bias, limit), -limit)
    def rmtolerance(self, sensitivity=128):
        self.weight = rmtolerance(self.weight, sensitivity=sensitivity)
        self.bias = rmtolerance(self.bias, sensitivity=sensitivity)
        self.learning = max(rmtolerance(self.learning, sensitivity=sensitivity), 1/sensitivity)
        self.truely = max(rmtolerance(self.truely, sensitivity=sensitivity), 1/sensitivity)
    def quantization(self, sensitivity=128):
        self.weight = quantization(self.weight, sensitivity=sensitivity)
        self.bias = quantization(self.bias, sensitivity=sensitivity)
        self.learning = max(quantization(self.learning, sensitivity=sensitivity), 1/sensitivity)
        self.truely = max(quantization(self.truely, sensitivity=sensitivity), 1/sensitivity)
    def process(self, input, target=None, train=True, trainrate=1, perceptron=False):
        input = self.activation.normalize(input)
        if target is None:
            target = input
        z = (input*self.weight)+(self.bias*self.truely)
        output = self.activation.activation(z)
        if not train:
            return output
        error = target-output
        if perceptron:
            self.weight += error*input*self.learning*trainrate
            self.bias += error*self.learning*trainrate
        else:
            deriv = self.activation.derivation(output)
            delta = error*deriv
            graw = delta*input
            grab = delta*self.truely
            self.weight += graw*self.learning*trainrate
            self.bias += grab*self.learning*trainrate
        return output

class CellForest:
    def __init__(self, cellscount=32, weight=None, bias=None, learning=None, truely=1, momentumexc=0.9):
        self.cells = [Cell(weight=weight, bias=bias, learning=learning, truely=truely, momentumexc=momentumexc) for _ in range(cellscount)]
        self.cellscount = cellscount
        self.activation = Activation()
        self.update_activation(self.activation)
        self.type = 9
    def limitation(self, limit=512):
        for cell in self.cells:
            cell.limitation(limit=limit)
    def forget(self):
        for cell in self.cells:
            cell.forget()
    def rmtolerance(self, sensitivity=128):
        for cell in self.cells:
            cell.rmtolerance(sensitivity=sensitivity)
    def quantization(self, sensitivity=128):
        for cell in self.cells:
            cell.quantization(sensitivity=sensitivity)
    def update_activation(self, obj):
        for i in range(self.cellscount):
            self.cells[i].activation = obj
        self.activation = obj
    def get_target_cells(self, inp, top_k=4):
        inp = max(min(inp, 1), 0)
        cells = []
        inpr = inp*self.cellscount
        for i in range(self.cellscount):
            seq = 1-(abs((i+0.5)-inpr)/self.cellscount)
            cells.append([self.cells[i], seq])
        cells.sort(key=lambda x: x[1], reverse=True)
        cells = cells[:top_k]
        return [[cell[0], 1-(i/len(cells))] for i, cell in enumerate(cells)]
    def process(self, input, target=None, train=True, trainrate=1, perceptron=False, top_k=4, pointweight=True):
        results = []
        main = 0.5
        cells = self.get_target_cells(input, top_k=top_k)
        best_cell = cells[0][0]
        for cell in cells:
            result = cell[0].process(input, target=target, train=train, trainrate=trainrate*cell[1], perceptron=perceptron)
            if cell[0] == best_cell:
                main = result
            results.append(result)
        results = results if results else [0.5]
        return ((sum(results)/len(results))+main)/2 if pointweight else sum(results)/len(results)
    def savedict(self):
        return {"t": 9, "cs": [save(cell, wantdict=True) for cell in self.cells], "a": self.activation.save()}
    def loaddict(self, data):
        self.cells = [load(cell, wantdict=True) for cell in data["cs"]]
        self.cellscount = len(data["cs"])
        self.activation.load(data["a"])
        self.update_activation(self.activation)

class MultiCellForest:
    def __init__(self, cellscount=32, weights=None, biases=None, icount=2, ocount=2, learning=None, truely=1, momentumexc=0.9, forest_depth=None):
        if forest_depth is None:
            forest_depth = icount
        self.cells = [MultiCell(weights=weights, biases=biases, icount=icount, ocount=ocount, learning=learning, truely=truely, momentumexc=momentumexc) for _ in range(cellscount**forest_depth)]
        self.cellscount = cellscount
        self.activation = Activation()
        self.update_activation(self.activation)
        self.icount = icount
        self.ocount = ocount
        self.forest_depth = forest_depth
        self.type = 10
    def limitation(self, limit=512):
        for cell in self.cells:
            cell.limitation(limit=limit)
    def forget(self):
        for cell in self.cells:
            cell.forget()
    def rmtolerance(self, sensitivity=128):
        for cell in self.cells:
            cell.rmtolerance(sensitivity=sensitivity)
    def quantization(self, sensitivity=128):
        for cell in self.cells:
            cell.quantization(sensitivity=sensitivity)
    def update_activation(self, obj):
        for i in range(len(self.cells)):
            self.cells[i].activation = obj
        self.activation = obj
    def get_target_cells(self, inp, top_k=4, clusters=32):
        inp = squeezeforclust(inp, k=clusters)
        cells = []
        cellscount = len(self.cells)
        inpr = inp*cellscount
        for i in range(cellscount):
            seq = 1-(abs((i+0.5)-inpr)/cellscount)
            cells.append([self.cells[i], seq])
        cells.sort(key=lambda x: x[1], reverse=True)
        cells = cells[:top_k]
        return [[cell[0], 1-(i/len(cells))] for i, cell in enumerate(cells)]
    def process(self, input, target=None, train=True, trainrate=1, perceptron=False, top_k=4, clusters=32, pointweight=True):
        results = []
        main = [0.5]*self.ocount
        cells = self.get_target_cells(input, top_k=top_k, clusters=clusters)
        best_cell = cells[0][0]
        for cell in cells:
            result = cell[0].process(input, target=target, train=train, trainrate=trainrate*cell[1], perceptron=perceptron)
            if cell[0] == best_cell:
                main = result
            results.append([[i] for i in result])
        if results:
            result = results[0]
            for h in results[1:]:
                for i, h2 in enumerate(h):
                    result[i] += h2
            result = [sum(i)/len(i) for i in result]
        else:
            result = [0.5]*self.ocount
        return [(i+i2)/2 for i, i2 in zip(result, main)] if pointweight else result
    def savedict(self):
        return {"t": 10, "ci": self.cellscount, "cs": [save(cell, wantdict=True) for cell in self.cells], "a": self.activation.save(), "ic": self.icount, "oc": self.ocount, "fd": self.forest_depth}
    def loaddict(self, data):
        self.cells = [load(cell, wantdict=True) for cell in data["cs"]]
        self.cellscount = data["ci"]
        self.activation.load(data["a"])
        self.update_activation(self.activation)
        self.icount = data["ic"]
        self.ocount = data["oc"]
        self.forest_depth = data["fd"]

class CellNetwork:
    def __init__(self, layers=[1, 16, 1], activations=None, learning=None, truely=1, momentumexc=0.9):
        if activations is None:
            activations = ["s"]*len(layers)
        self.layers = []
        self.rawlayers = layers
        for i in range(len(layers)-1):
            input_count = layers[i]
            output_count = layers[i+1]
            self.layers.append([MultiInputCell(wcount=input_count, learning=learning, truely=truely, momentumexc=momentumexc) for _ in range(output_count)])
        self.activations = [Activation(act) for act in activations]
        self.update_activations(self.activations)
        self.type = 11
    def limitation(self, limit=512):
        for cells in self.layers:
            for cell in cells:
                cell.limitation(limit=limit)
    def forget(self):
        for cells in self.layers:
            for cell in cells:
                cell.forget()
    def rmtolerance(self, sensitivity=128):
        for cells in self.layers:
            for cell in cells:
                cell.rmtolerance(sensitivity=sensitivity)
    def quantization(self, sensitivity=128):
        for cells in self.layers:
            for cell in cells:
                cell.quantization(sensitivity=sensitivity)
    def update_activations(self, objs):
        self.activations = objs
        for cells, obj in zip(self.layers, objs):
            for cell in cells:
                cell.activation = obj
    def process(self, input, target=None, train=True, trainrate=1, perceptron=False):
        if target is None:
            target = [0.5]*self.rawlayers[-1]
        current = input
        layer_outs = [current]
        for layer in self.layers:
            outs = []
            for cell in layer:
                out = cell.process(current, train=False)
                outs.append(out)
            current = outs
            layer_outs.append(current)
        final_out = current
        if not train:
            return final_out
        errors = [t-o for t, o in zip(target, final_out)]
        for i in reversed(range(len(self.layers))):
            next_layer_errors = [0.0]*self.rawlayers[i]
            current_layer = self.layers[i]
            prev_layer_acts = layer_outs[i]
            for j, cell in enumerate(current_layer):
                cell_error = errors[j]
                cell.process(prev_layer_acts, target=cell_error+layer_outs[i+1][j], train=True, trainrate=trainrate, perceptron=perceptron)
                for k in range(len(cell.weights)):
                    next_layer_errors[k] += cell.weights[k]*cell_error
            errors = next_layer_errors
        return final_out
    def savedict(self):
        return {"t": 11, "rl": self.rawlayers, "l": [[save(cell, wantdict=True) for cell in layer] for layer in self.layers], "a": [a.save() for a in self.activations]}
    def loaddict(self, data):
        self.layers = [[load(cell, wantdict=True) for cell in layer] for layer in data["l"]]
        self.rawlayers = data["rl"]
        self.activations = [Activation() for _ in range(len(data["a"]))]
        [a.load(d) for d, a in zip(data["a"], self.activations)]
        self.update_activations(self.activations)

class CellNetworkForest:
    def __init__(self, networkcount=32, layers=[1, 16, 1], activations=None, learning=None, truely=1, momentumexc=0.9, forest_depth=None):
        if activations is None:
            activations = ["s"]*len(layers)
        if forest_depth is None:
            forest_depth = layers[0]
        self.networks = [CellNetwork(layers=layers, learning=learning, truely=truely, momentumexc=momentumexc) for _ in range(networkcount**forest_depth)]
        self.networkcount = networkcount
        self.activations = [Activation(act) for act in activations]
        self.update_activations(self.activations)
        self.layers = layers
        self.forest_depth = forest_depth
        self.type = 12
    def limitation(self, limit=512):
        for network in self.networks:
            network.limitation(limit=limit)
    def forget(self):
        for network in self.networks:
            network.forget()
    def rmtolerance(self, sensitivity=128):
        for n in self.networks:
            n.rmtolerance(sensitivity=sensitivity)
    def quantization(self, sensitivity=128):
        for n in self.networks:
            n.quantization(sensitivity=sensitivity)
    def update_activations(self, objs):
        for i in range(len(self.networks)):
            self.networks[i].update_activations(objs)
        self.activations = objs
    def get_target_networks(self, inp, top_k=4, clusters=32):
        inp = squeezeforclust(inp, k=clusters)
        ns = []
        networkcount = len(self.networks)
        inpr = inp*networkcount
        for i in range(networkcount):
            seq = 1-(abs((i+0.5)-inpr)/networkcount)
            ns.append([self.networks[i], seq])
        ns.sort(key=lambda x: x[1], reverse=True)
        ns = ns[:top_k]
        return [[n[0], 1-(i/len(ns))] for i, n in enumerate(ns)]
    def process(self, input, target=None, train=True, trainrate=1, perceptron=False, top_k=4, clusters=32, pointweight=True):
        results = []
        main = [0.5]*self.layers[-1]
        ns = self.get_target_networks(input, top_k=top_k, clusters=clusters)
        best_network = ns[0][0]
        for network in ns:
            result = network[0].process(input, target=target, train=train, trainrate=trainrate*network[1], perceptron=perceptron)
            if network[0] == best_network:
                main = result
            results.append([[i] for i in result])
        if results:
            result = results[0]
            for h in results[1:]:
                for i, h2 in enumerate(h):
                    result[i] += h2
            result = [sum(i)/len(i) for i in result]
        else:
            result = [0.5]*self.layers[-1]
        return [(i+i2)/2 for i, i2 in zip(result, main)] if pointweight else result
    def savedict(self):
        return {"t": 12, "ni": self.networkcount, "ns": [save(network, wantdict=True) for network in self.networks], "ls": self.layers, "fd": self.forest_depth, "a": [a.save() for a in self.activations]}
    def loaddict(self, data):
        self.networks = [load(network, wantdict=True) for network in data["ns"]]
        self.networkcount = data["ni"]
        self.layers = data["ls"]
        self.forest_depth = data["fd"]
        self.activations = [Activation() for _ in range(len(data["a"]))]
        [a.load(d) for d, a in zip(data["a"], self.activations)]
        self.update_activations(self.activations)
