# Zevihanthosa - AI Model
"""
The ZevihaNut/1.3 model is an evolution of the ZevihaNut/1.2 version, with a few key differences. Here are the notable features of ZevihaNut/1.3:

1. New Activation Function: Bisigmoid

The ZevihaNut/1.3 model introduces a new activation function called bisigmoid, which enhances scalability. This function is a variant of the traditional sigmoid but differs by producing outputs in the range of -1 to 1, instead of the usual 0 to 1. This allows for a more flexible decision-making process and enables the model to make finer, more adaptive decisions with a broader value range.

2. New Tokenizer Splitting Methods

ZevihaNut/1.3 also includes new splitting methods for the Tokenizer, which allow for more flexible and efficient text parsing:

Sentence: This method splits the text based on "." (period) characters. Each sentence is treated as a separate unit, which helps the model process text at the sentence level, providing better structure and context.

Basicwords: The simplest form of tokenization, where text is split based on " " (space) characters. This method breaks the text down into individual words, treating each word as a separate entity for learning.

Directive: This highly optimized method splits text by considering escape characters, spaces, tabs, and other formatting markers. It mimics the way humans read and parse text, ensuring that the tokenizer works similarly to human reading patterns, providing a more accurate understanding of the text.


3. General Improvements and Performance

ZevihaNut/1.3, in addition to the previously developed features such as vectorization and neural processing, offers faster and more accurate text parsing. The new activation function and splitting methods make decision-making processes more precise and adaptable. Additionally, the model's multi-layered structure and deep learning techniques allow for more efficient learning and adaptation.

4. Summary

ZevihaNut/1.3 builds on the solid foundations of ZevihaNut/1.2 while introducing significant improvements. The new bisigmoid activation function and enhanced tokenizer splitting methods make the model's text processing and learning capabilities stronger and more flexible. As a result, it can generate more accurate, faster, and context-aware responses.


"""

import random
import math
import numpy
import difflib
def General_Activation(self, value, method):
	if method == "sigmoid":
		return 1/(1+math.exp(-value))
	elif method == "tanh":
		return math.tanh(value)
	elif method == "relu":
		return max(0, value)
	elif method == "abs":
		return abs(value)
	elif method == "sin":
		return math.sin(value)
	elif method == "cos":
		return math.cos(value)
	elif method == "zevianthosa":
		return (1/(1+((value)/self.weight)**-(value+(self.momentum/self.weight))))
	elif method == "bisigmoid":
		return (2/(1+math.exp(-value)))-1
	return value
class Cell:
	def __init__(self, weight=None, truely=10, momentum_default=0):
		if weight == None:
			weight = random.uniform(-1, 1)
		self.weight = weight
		self.learning = random.random()/truely
		self.momentum = momentum_default
		self.truely = truely
	def activation(self, value, method):
		return General_Activation(self, value, method)
	def save(self):
		return [self.weight, self.learning, self.momentum, self.truely]
	def load(self, data):
		self.weight = float(data[0])
		self.learning = float(data[1])
		self.momentum = float(data[2])
		self.truely = float(data[3])
	def stabilizer(self, value, alpha=0.1):
		self.weight = self.weight*(1-alpha)+value*alpha
	def process(self, value, target_value=0, train=True, activation_method="sigmoid"):
		if target_value == None:
			target_value = value
		value = self.weight*value
		value = self.activation(value, activation_method)
		if train:
			error = target_value-value
			self.momentum = (self.momentum*(1-(1/self.truely)))+(error*self.learning)
			self.weight += self.momentum
		return value
class Brain:
	def __init__(self, cells=128, minvalue=-1, maxvalue=1, randomize=False, truely=10, momentum_default=0, kingvalue=None):
		if kingvalue == None:
			kingvalue = random.uniform(minvalue, maxvalue)
		self.cells = []
		if randomize:
			for _ in range(cells):
				self.cells.append(Cell(random.uniform(minvalue, maxvalue), truely, momentum_default))
		else:
			for h in numpy.linspace(minvalue, maxvalue, cells):
				self.cells.append(Cell(h, truely, momentum_default))
		if randomize:
			self.king = Cell(random.uniform(minvalue, maxvalue), truely, momentum_default)
		else:
			self.king = Cell(kingvalue, truely, momentum_default)
		self.minvalue = minvalue
		self.maxvalue = maxvalue
		self.momentum_default = momentum_default
		self.truely = truely
	def activation(self, value, method):
		return General_Activation(self, value, method)
	def save(self):
		cells = []
		for cell in self.cells:
			cells.append(cell.save())
		return [cells, self.king.save(), self.minvalue, self.maxvalue, self.momentum_default, self.truely]
	def load(self, data):
		cells = []
		for cell in data[0]:
			cellroot = Cell(0)
			cellroot.load(cell)
			cells.append(cellroot)
		self.king.load(data[1])
		self.minvalue = data[2]
		self.maxvalue = data[3]
		self.momentum_default = data[4]
		self.truely = data[5]
	def clusters_to_temperature(self, clusters):
		return 1/clusters
	def stabilizer(self, value, alpha=0.1, temperature=1):
		temperature = min(max(temperature, 0), 1)
		for cell in self.cells:
			if temperature >= abs(cell.weight-value)/self.maxvalue:
				cell.stabilizer(value, alpha=alpha)
	def momentum_set(self, value, temperature=1):
		temperature = min(max(temperature, 0), 1)
		for cell in self.cells:
			if temperature >= abs(cell.momentum-value)/self.maxvalue:
				cell.momentum = value
	def weight_set(self, value, temperature=1):
		temperature = min(max(temperature, 0), 1)
		for cell in self.cells:
			if temperature >= abs(cell.weight-value)/self.maxvalue:
				cell.weight = value
	def learning_set(self, value, temperature=1):
		temperature = min(max(temperature, 0), 1)
		for cell in self.cells:
			if temperature >= abs(cell.learning-value)/self.maxvalue:
				cell.learning = value
	def process(self, value, target_value=0, train=True, activation_method="sigmoid", temperature=1, ignore_errors=False, decision_method="all"):
		if target_value == None:
			target_value = value
		temperature = min(max(temperature, 0), 1)
		processes = []
		for cell in self.cells:
			if temperature >= abs(cell.process(0, train=False)-value)/self.maxvalue:
				if ignore_errors:
					try:
						processes.append(cell.process(value, target_value=target_value, train=train, activation_method=activation_method))
					except:
						pass
				else:
					processes.append(cell.process(value, target_value=target_value, train=train, activation_method=activation_method))
		if len(processes) == 0:
			processes = [value]
		if decision_method == "sum":
			value = sum(processes)
		elif decision_method == "average":
			value = sum(processes)/len(processes)
		elif decision_method == "all":
			value = (sum(processes)/len(processes)+sum(processes))/2
		if ignore_errors:
			try:
				value = self.king.process(value, target_value=target_value, train=train, activation_method=activation_method)
			except:
				if decision_method == "sum":
					value = value/len(processes)
		else:
			value = self.king.process(value, target_value=target_value, train=train, activation_method=activation_method)
		return value
	def process_noob(self, value, train=True):
		return self.process(value, target_value=value, train=train, activation_method="sigmoid", temperature=1/5, ignore_errors=True, decision_method="all")
	def limited_weights(self):
		for cell in self.cells:
			cell.weight = min(max(cell.weight, self.minvalue), self.maxvalue)
	def backpropagation(self, value, temperature=1):
		temperature = min(max(temperature, 0), 1)
		for cell in self.cells:
			v = cell.process(value, train=False)
			if temperature >= abs(v-value)/self.maxvalue:
				error = value-v
				cell.weight += error*cell.learning
class AdvancedBrain:
	def __init__(self, layers=[128, 64, 32], auto_layers=False, auto_layers_main=128, auto_layers_count=3, auto_layers_cutter=2, auto_layers_randomize=False, auto_layers_randomize_minlayercount=1, auto_layers_randomize_maxlayercount=5, auto_layers_randomize_minmain=64, auto_layers_randomize_maxmain=512, auto_layers_randomize_mincutter=1, auto_layers_randomize_maxcutter=3, randomize=False, truely=10, minvalue=-1, maxvalue=1, truely_cutter=1.5, momentum_default=0, kingvalue=None, hostkingvalue=None):
		if kingvalue == None:
			kingvalue = random.uniform(minvalue, maxvalue)
		if hostkingvalue == None:
			hostkingvalue = random.uniform(minvalue, maxvalue)
		if auto_layers:
			if auto_layers_randomize:
				auto_layers_count = random.randint(auto_layers_randomize_minlayercount, auto_layers_randomize_maxlayercount)
				auto_layers_main = random.randint(auto_layers_randomize_minmain, auto_layers_randomize_maxmain)
				auto_layers_cutter = random.uniform(auto_layers_randomize_mincutter, auto_layers_randomize_maxcutter)
			layers = []
			main = auto_layers_main
			for _ in range(auto_layers_count):
				layers.append(main)
				main = int(main/auto_layers_cutter)
				if main <= 0:
					break
		self.layers = []
		self.truely = truely
		for cells in layers:
			self.layers.append(Brain(cells=cells, minvalue=minvalue, maxvalue=maxvalue, randomize=randomize, truely=truely, momentum_default=momentum_default, kingvalue=kingvalue))
			truely = int(truely/truely_cutter)
			if truely <= 0:
				truely = 1
		self.minvalue = minvalue
		self.maxvalue = maxvalue
		self.momentum_default = momentum_default
		if randomize:
			self.king = Cell(random.uniform(minvalue, maxvalue), truely, momentum_default)
		else:
			self.king = Cell(hostkingvalue, truely, momentum_default)
	def activation(self, value, method):
		return General_Activation(self, value, method)
	def save(self):
		layers = []
		for layer in self.layers:
			layers.append(layer.save())
		return [layers, self.minvalue, self.maxvalue, self.truely, self.momentum_default, self.king.save()]
	def load(self, data):
		self.layers = []
		for layer in data[0]:
			brain = Brain()
			brain.load(layer)
			self.layers.append(brain)
		self.minvalue = data[1]
		self.maxvalue = data[2]
		self.truely = data[3]
		self.momentun_default = data[4]
		self.king.load(data[5])
	def clusters_to_temperature(self, clusters):
		return 1/clusters
	def stabilizer(self, value, alpha=0.1, temperature=1):
		for layer in self.layers:
			layer.stabilizer(value, alpha=alpha, temperature=temperature)
	def momentum_set(self, value, temperature=1):
		for layer in self.layers:
			layer.momentum_set(value, temperature=temperature)
	def weight_set(self, value, temperature=1):
		for layer in self.layers:
			layer.weight_set(value, temperature=temperature)
	def learning_set(self, value, temperature=1):
		for layer in self.layers:
			layer.learning_set(value, temperature=temperature)
	def process(self, value, target_value=None, train=True, activation_method="sigmoid", temperature=1, ignore_errors=False, decision_method="all", finally_decision_method="all"):
		if target_value == None:
			target_value = value
		processes = []
		for layer in self.layers:
			processes.append(layer.process(value, target_value=target_value, train=train, activation_method=activation_method, temperature=temperature, ignore_errors=ignore_errors, decision_method=decision_method))
		if len(processes) == 0:
			processes = [value]
		if finally_decision_method == "sum":
			value = sum(processes)
		elif finally_decision_method == "average":
			value = sum(processes)/len(processes)
		elif finally_decision_method == "all":
			value = (sum(processes)/len(processes)+sum(processes))/2
		if ignore_errors:
			try:
				value = self.king.process(value, target_value=target_value, train=train, activation_method=activation_method)
			except:
				if decision_method == "sum":
					value = value/len(processes)
		else:
			value = self.king.process(value, target_value=target_value, train=train, activation_method=activation_method)
		return value
	def process_noob(self, value, train=True):
		return self.process(value, target_value=value, train=train, activation_method="sigmoid", temperature=1/7, ignore_errors=True, decision_method="all", finally_decision_method="all")
	def limited_weights(self):
		for layer in self.layers:
			layer.limited_weights()
	def backpropagation(self, value, temperature=1):
		for layer in self.layers:
			layer.backpropagation(value, temperature=temperature)
class Tokenizer:
	def __init__(self, type, custom=None):
		self.type = type
		self.custom = custom
	def save(self):
		return self.type
	def load(self, data):
		self.type = data
	def tokenize(self, text):
		if self.custom != None:
			return self.custom(text)
		elif self.type == "words":
			charbaseds = """.,!?;:'\"()[]{}-_/\\|*&^%$#@~`´<>†††—–·…‰⁰¹²³⁴⁵⁶⁷⁸⁹°∑√≠≡≈∞∈⊂⊆∩∪∧∨⊕⊗√⊥∠∂⊤∇∧∧∑∏
⟨⟩⟪⟫⟬⟭⟮⟯⊗⊗⌘≀∝∧∩⌉⌋☀✌✖✗✚★☆✪✧✩⚡⚠⟧⟦⟮⟯⠀ⁱʰʲʳⴻ⁰ⁿ⁶⁶✓✔✖✂☐☑◉⚪⚫"""
			for char in charbaseds:
				text = text.replace(char, " "+char+" ")
			newtext = []
			for h in text.split(" "):
				if len(h) != 0:
					newtext.append(h)
			return newtext
		elif self.type == "bottombasedwords":
			charbaseds = """.,!?;:'\"()[]{}-_/\\|*&^%$#@~`´<>†††—–·…‰⁰¹²³⁴⁵⁶⁷⁸⁹°∑√≠≡≈∞∈⊂⊆∩∪∧∨⊕⊗√⊥∠∂⊤∇∧∧∑∏
⟨⟩⟪⟫⟬⟭⟮⟯⊗⊗⌘≀∝∧∩⌉⌋☀✌✖✗✚★☆✪✧✩⚡⚠⟧⟦⟮⟯⠀ⁱʰʲʳⴻ⁰ⁿ⁶⁶✓✔✖✂☐☑◉⚪⚫"""
			for char in charbaseds:
				text = text.replace(char, " "+char+" ")
			newtext = []
			for h in text.split(" "):
				if len(h) != 0:
					newtext.append(h)
			newtext2 = []
			for h in newtext:
				base1 = ""
				base2 = ""
				base1full = False
				maxer = int(len(h)/2)
				for h2 in h:
					if not base1full:
						base1 += h2
						if len(base1) >= maxer and h2 in "qwrtypsdfghjklmnbvcxzQWRTYPSDFGHJKLZXCVBNM":
							base1full = True
					else:
						base2 += h2
				if len(base1) >= 1:
						newtext2.append(base1)
				if len(base2) >= 1:
						newtext2.append(base2)
			return newtext2
		elif self.type == "chars":
			return list(text)
		elif self.type == "sentence":
			return text.split(".")
		elif self.type == "basicwords":
			return text.split(" ")
		elif self.type == "directive":
			return text.split()
					
class Chatbot:
	def __init__(self, cells=128, minvalue=-1, maxvalue=1, randomize=False, truely=10, momentum_default=0, kingvalue=None, type="words", tokenizer=None):
		self.brain = Brain(cells=cells, minvalue=minvalue, maxvalue=maxvalue, randomize=randomize, truely=truely, momentum_default=momentum_default, kingvalue=kingvalue)
		self.type = type
		self.grammar = {}
		if tokenizer == None:
			tokenizer = Tokenizer(type)
		self.tokenizer = tokenizer
	def save(self):
		return [self.brain.save(), self.type, self.grammar, self.tokenizer.save()]
	def load(self, data):
		self.brain.load(data[0])
		self.type = data[1]
		self.grammar = data[2]
		self.tokenizer.load(data[3])
	def complation(self, text, train=True, activation_method="sigmoid", temperature=None, ignore_errors=False, decision_method="all", vector="all", vector_temperature=None, vector_randomizer=92840198, vector_hash_truely=5, response_randomizer=17482736, response_temperature=None, maxlength=15):
		if temperature == None:
			temperature = 1/self.brain.truely
		if vector_temperature == None:
			vector_temperature = 1/self.brain.truely
		if response_temperature == None:
			response_temperature = 1/self.brain.truely
		text = self.tokenizer.tokenize(text)
		otext = text
		if vector == "grammarbased":
			newtext = []
			for h in text:
				if h in self.grammar:
					newtext.append(self.grammar[h])
				else:
					canbe = []
					most = 0
					mostv = float("inf")
					for hh, v in self.grammar.items():
						diff = 1-difflib.SequenceMatcher(None, hh, h).ratio()
						if vector_temperature >= diff:
							canbe.append(v)
						if diff <= mostv:
							most = v
							mostv = diff
					if len(canbe) >= 1:
						newtext.append(canbe[int(len(canbe)*vector_randomizer)%len(canbe)])
					else:
						newtext.append(most)
			return newtext
		elif vector == "hashvaluebased":
			newtext = []
			for h in text:
				hash = 0
				for hh in h:
					hash += ord(hh)/(1114112/vector_hash_truely)
					hash += (1/(vector_hash_truely*10))
				newtext.append(h)
		elif vector == "all":
			newtext = []
			for h in text:
				hash = 0
				for hh in h:
					hash += ord(hh)/(1114112/vector_hash_truely)
					hash += (1/(vector_hash_truely*10))
				if h in self.grammar:
					newtext.append((self.grammar[h]+hash)/2)
				else:
					canbe = []
					most = 0
					mostv = float("inf")
					for hh, v in self.grammar.items():
						diff = 1-difflib.SequenceMatcher(None, hh, h).ratio()
						if vector_temperature >= diff:
							canbe.append(v)
						if diff <= mostv:
							most = v
							mostv = diff
					if len(canbe) >= 1:
						newtext.append(((canbe[int(len(canbe)*hash*vector_randomizer)%len(canbe)])+hash)/2)
					else:
						newtext.append((most+hash)/2)
		response = []
		vvh = sum(newtext)/len(newtext)
		vvh = self.brain.process(vvh, train=train, activation_method=activation_method, temperature=temperature, ignore_errors=ignore_errors, decision_method=decision_method)
		targetlength = int((vvh*len(newtext)*response_randomizer)%maxlength)+1
		for h, hhh in zip(newtext, otext):
			if len(response) >= targetlength:
				break
			vh = (self.brain.process(h, train=train, activation_method=activation_method, temperature=temperature, ignore_errors=ignore_errors, decision_method=decision_method)+vvh)/2
			if train:
				self.grammar[hhh] = h
			most = ""
			mostv = float("inf")
			canbe = []
			for hh, v in self.grammar.items():
				diff = abs(vh-v)/self.brain.truely
				if diff <= mostv:
					most = hh
					mostv = diff
				if response_temperature >= diff:
					canbe.append(hh)
			if len(canbe) >= 1:
				response.append(canbe[int(len(canbe)*vh*response_randomizer)%len(canbe)])
			else:
				response.append(most)
		if len(response) < targetlength:
			for _ in range(targetlength-len(response)):
				vh = self.brain.process(vh, train=train, activation_method=activation_method, temperature=temperature, ignore_errors=ignore_errors, decision_method=decision_method)
				most = ""
				mostv = float("inf")
				canbe = []
				for hh, v in self.grammar.items():
					diff = abs(vh-v)/self.brain.truely
					if diff <= mostv:
						most = hh
						mostv = diff
					if response_temperature >= diff:
						canbe.append(hh)
				if len(canbe) >= 1:
					response.append(canbe[int(len(canbe)*vh*response_randomizer)%len(canbe)])
				else:
					response.append(most)
		if self.type == "chars":
			return "".join(response)
		else:
			return " ".join(response)
class AdvancedChatbot:
	def __init__(self, layers=[128, 64, 32], auto_layers=False, auto_layers_main=128, auto_layers_count=3, auto_layers_cutter=2, auto_layers_randomize=False, auto_layers_randomize_minlayercount=1, auto_layers_randomize_maxlayercount=5, auto_layers_randomize_minmain=64, auto_layers_randomize_maxmain=512, auto_layers_randomize_mincutter=1, auto_layers_randomize_maxcutter=3, randomize=False, truely=10, minvalue=-1, maxvalue=1, truely_cutter=1.5, momentum_default=0, kingvalue=None, hostkingvalue=None, type="words", tokenizer=None):
		self.brain = AdvancedBrain(layers=layers, auto_layers=auto_layers, auto_layers_main=auto_layers_main, auto_layers_count=auto_layers_count, auto_layers_cutter=auto_layers_cutter, auto_layers_randomize=auto_layers_randomize, auto_layers_randomize_minlayercount=auto_layers_randomize_minlayercount, auto_layers_randomize_maxlayercount=auto_layers_randomize_maxlayercount, auto_layers_randomize_minmain=auto_layers_randomize_minmain, auto_layers_randomize_maxmain=auto_layers_randomize_maxmain, auto_layers_randomize_mincutter=auto_layers_randomize_mincutter, auto_layers_randomize_maxcutter=auto_layers_randomize_maxcutter, randomize=randomize, truely=truely, minvalue=minvalue, maxvalue=maxvalue, truely_cutter=truely_cutter, momentum_default=momentum_default, kingvalue=kingvalue, hostkingvalue=hostkingvalue)
		self.type = type
		self.grammar = {}
		if tokenizer == None:
			tokenizer = Tokenizer(type)
		self.tokenizer = tokenizer
	def save(self):
		return [self.brain.save(), self.type, self.grammar, self.tokenizer.save()]
	def load(self, data):
		self.brain.load(data[0])
		self.type = data[1]
		self.grammar = data[2]
		self.tokenizer.load(data[3])
	def complation(self, text, train=True, activation_method="sigmoid", temperature=None, ignore_errors=False, decision_method="all", vector="all", vector_temperature=None, vector_randomizer=92840198, vector_hash_truely=5, response_randomizer=17482736, response_temperature=None, maxlength=15, finally_decision_method="all"):
		if temperature == None:
			temperature = 1/self.brain.truely
		if vector_temperature == None:
			vector_temperature = 1/self.brain.truely
		if response_temperature == None:
			response_temperature = 1/self.brain.truely
		text = self.tokenizer.tokenize(text)
		otext = text
		if vector == "grammarbased":
			newtext = []
			for h in text:
				if h in self.grammar:
					newtext.append(self.grammar[h])
				else:
					canbe = []
					most = 0
					mostv = float("inf")
					for hh, v in self.grammar.items():
						diff = 1-difflib.SequenceMatcher(None, hh, h).ratio()
						if vector_temperature >= diff:
							canbe.append(v)
						if diff <= mostv:
							most = v
							mostv = diff
					if len(canbe) >= 1:
						newtext.append(canbe[int(len(canbe)*vector_randomizer)%len(canbe)])
					else:
						newtext.append(most)
			return newtext
		elif vector == "hashvaluebased":
			newtext = []
			for h in text:
				hash = 0
				for hh in h:
					hash += ord(hh)/(1114112/vector_hash_truely)
					hash += (1/(vector_hash_truely*10))
				newtext.append(h)
		elif vector == "all":
			newtext = []
			for h in text:
				hash = 0
				for hh in h:
					hash += ord(hh)/(1114112/vector_hash_truely)
					hash += (1/(vector_hash_truely*10))
				if h in self.grammar:
					newtext.append((self.grammar[h]+hash)/2)
				else:
					canbe = []
					most = 0
					mostv = float("inf")
					for hh, v in self.grammar.items():
						diff = 1-difflib.SequenceMatcher(None, hh, h).ratio()
						if vector_temperature >= diff:
							canbe.append(v)
						if diff <= mostv:
							most = v
							mostv = diff
					if len(canbe) >= 1:
						newtext.append(((canbe[int(len(canbe)*hash*vector_randomizer)%len(canbe)])+hash)/2)
					else:
						newtext.append((most+hash)/2)
		response = []
		vvh = sum(newtext)/len(newtext)
		vvh = self.brain.process(vvh, train=train, activation_method=activation_method, temperature=temperature, ignore_errors=ignore_errors, decision_method=decision_method, finally_decision_method=finally_decision_method)
		targetlength = int((vvh*len(newtext)*response_randomizer)%maxlength)+1
		for h, hhh in zip(newtext, otext):
			if len(response) >= targetlength:
				break
			vh = (self.brain.process(h, train=train, activation_method=activation_method, temperature=temperature, ignore_errors=ignore_errors, decision_method=decision_method, finally_decision_method=finally_decision_method)+vvh)/2
			if train:
				self.grammar[hhh] = h
			most = ""
			mostv = float("inf")
			canbe = []
			for hh, v in self.grammar.items():
				diff = abs(vh-v)/self.brain.truely
				if diff <= mostv:
					most = hh
					mostv = diff
				if response_temperature >= diff:
					canbe.append(hh)
			if len(canbe) >= 1:
				response.append(canbe[int(len(canbe)*vh*response_randomizer)%len(canbe)])
			else:
				response.append(most)
		if len(response) < targetlength:
			for _ in range(targetlength-len(response)):
				vh = self.brain.process(vh, train=train, activation_method=activation_method, temperature=temperature, ignore_errors=ignore_errors, decision_method=decision_method, finally_decision_method=finally_decision_method)
				most = ""
				mostv = float("inf")
				canbe = []
				for hh, v in self.grammar.items():
					diff = abs(vh-v)/self.brain.truely
					if diff <= mostv:
						most = hh
						mostv = diff
					if response_temperature >= diff:
						canbe.append(hh)
				if len(canbe) >= 1:
					response.append(canbe[int(len(canbe)*vh*response_randomizer)%len(canbe)])
				else:
					response.append(most)
		if self.type == "chars":
			return "".join(response)
		else:
			return " ".join(response)