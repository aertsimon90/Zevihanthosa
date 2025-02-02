# Zevihanthosa - AI Model
"""
Zevianthosa, developed solely by aertsimon90, is an advanced artificial intelligence model currently under continuous development. The ZevihaNut/1.0 model is the very first iteration of Zevianthosa, designed with the goal of creating an AI system that can adapt, learn, and make decisions in complex environments. This model incorporates several advanced features aimed at improving performance, flexibility, and scalability.

At the core of Zevianthosa, the model’s neurons (referred to as cells) are designed to process input data through dynamic weight adjustments, which are fine-tuned during training. Each cell’s weight and learning parameters can be adjusted based on feedback, ensuring the AI can learn in diverse scenarios. The Zevianthosa architecture also allows for a variety of activation functions like sigmoid, tanh, ReLU, and custom functions like "zevianthosa," which provides a unique non-linear transformation based on a specific weight formula, leading to more intricate decision-making capabilities.

A critical aspect of Zevianthosa's learning process is the incorporation of momentum. Each neuron has a momentum value that helps smooth out weight updates during training, ensuring the model learns faster and avoids overshooting during optimization. The learning rate and momentum are adaptively updated, making the model flexible enough to handle various types of data and tasks. The weight stabilization mechanism allows for fine-tuning the network’s parameters over time, enhancing its performance even further.

Zevianthosa supports single-layer and multi-layer configurations, with a particular focus on scalability. The AI can be customized to have multiple layers, each with a specific number of neurons. The architecture can be adjusted manually or automatically, allowing for efficient creation of models based on the problem at hand. For automatic layer generation, the system can randomly adjust parameters, creating an optimal network that balances performance and efficiency.

The decision-making process is another standout feature of Zevianthosa. The model offers multiple decision methods, including sum, average, and "all," where the final output is influenced by a combination of results from multiple neurons or layers. This flexibility allows Zevianthosa to be applied to a wide range of tasks, from classification to regression, adapting to various problem complexities.

During training, Zevianthosa continuously adjusts its weights based on the error between predicted and expected outputs, enabling the network to refine its parameters over time. A unique “ignore errors” mode allows the model to skip certain errors during learning, providing the flexibility to train on noisy or inconsistent data.

An essential component of Zevianthosa is the “king” cell, a specialized neuron that influences the learning behavior of the entire network. The king cell's weight and momentum parameters are treated as high-level configurations that shape the network’s overall output, ensuring that the learning process remains coherent and aligned with the desired objective.

As the first version (ZevihaNut/1.0) of Zevianthosa, this model serves as the foundation for future iterations. Over time, new versions will introduce faster training algorithms, more sophisticated decision-making strategies, and increased adaptability to even more complex data structures. These updates will aim to improve Zevianthosa's capabilities, allowing it to tackle a broader range of tasks, from machine learning to AI-driven solutions in various fields.

In summary, Zevianthosa's ZevihaNut/1.0 model, developed by aertsimon90, represents the first step towards a powerful and adaptable artificial intelligence system. It combines dynamic weight adjustments, innovative activation functions, adaptive learning rates, and advanced decision-making strategies to deliver an AI model that can continuously learn and evolve. Future updates will only enhance its functionality, making it a significant player in the artificial intelligence space.

"""

import random
import math
import numpy

class Cell:
	def __init__(self, weight=None, truely=10, momentum_default=0):
		if weight == None:
			weight = random.uniform(-1, 1)
		self.weight = weight
		self.learning = random.random()/truely
		self.momentum = momentum_default
		self.truely = truely
	def save(self):
		return [self.weight, self.learning, self.momentum, self.truely]
	def load(self, data):
		self.weight = float(data[0])
		self.learning = float(data[1])
		self.momentum = float(data[2])
		self.truely = float(data[3])
	def stabilizer(self, value, alpha=0.1):
		self.weight = self.weight*(1-alpha)+value*alpha
	def activation(self, value, method):
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
		return value
	def process(self, value, target_value=0, train=True, activation_method="sigmoid"):
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
	def save(self):
		cells = []
		for cell in self.cells:
			cells.append(cell.save())
		return [cells, self.king.save(), self.minvalue, self.maxvalue, self.momentum_default]
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
	def clusters_to_temperature(self, clusters):
		return 1/clusters
	def stabilizer(self, value, alpha=0.1, temperature=1):
		temperature = min(max(temperature, 0), 1)
		for cell in self.cells:
			if 1-temperature <= abs(cell.weight-value)/self.maxvalue:
				cell.stabilizer(value, alpha=alpha)
	def activation(self, value, method):
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
		return value
	def momentum_set(self, value, temperature=1):
		temperature = min(max(temperature, 0), 1)
		for cell in self.cells:
			if 1-temperature <= abs(cell.momentum-value)/self.maxvalue:
				cell.momentum = value
	def weight_set(self, value, temperature=1):
		temperature = min(max(temperature, 0), 1)
		for cell in self.cells:
			if 1-temperature <= abs(cell.weight-value)/self.maxvalue:
				cell.weight = value
	def learning_set(self, value, temperature=1):
		temperature = min(max(temperature, 0), 1)
		for cell in self.cells:
			if 1-temperature <= abs(cell.learninh-value)/self.maxvalue:
				cell.learning = value
	def process(self, value, target_value=0, train=True, activation_method="sigmoid", temperature=1, ignore_errors=False, decision_method="all"):
		temperature = min(max(temperature, 0), 1)
		processes = []
		for cell in self.cells:
			if 1-temperature <= abs(cell.process(0, train=False)-value)/self.maxvalue:
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
		for cells in layers:
			self.layers.append(Brain(cells=cells, minvalue=minvalue, maxvalue=maxvalue, randomize=randomize, truely=truely, momentum_default=momentum_default, kingvalue=kingvalue))
			truely = int(truely/truely_cutter)
			if truely <= 0:
				truely = 1
		self.minvalue = minvalue
		self.maxvalue = maxvalue
		self.truely = truely
		self.momentum_default = momentum_default
		if randomize:
			self.king = Cell(random.uniform(minvalue, maxvalue), truely, momentum_default)
		else:
			self.king = Cell(hostkingvalue, truely, momentum_default)
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
	def activation(self, value, method):
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
		return value
	def momentum_set(self, value, temperature=1):
		for layer in self.layers:
			layer.momentum_set(value, temperature=temperature)
	def weight_set(self, value, temperature=1):
		for layer in self.layers:
			layer.weight_set(value, temperature=temperature)
	def learning_set(self, value, temperature=1):
		for layer in self.layers:
			layer.learning_set(value, temperature=temperature)
	def process(self, value, target_value=0, train=True, activation_method="sigmoid", temperature=1, ignore_errors=False, decision_method="all", finally_decision_method="all"):
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