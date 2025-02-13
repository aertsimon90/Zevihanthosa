# Zevihanthosa - AI Model
"""
ZevihaNut/1.5 Model Overview

ZevihaNut/1.5 represents a significant evolution of the Zevihanthosa AI framework, integrating a variety of advanced features that enhance its functionality, flexibility, and ease of use. This version introduces an array of upgrades, including the ONCEbende chatbot AI, improvements in language processing, and more robust architectural features that align with current AI development standards. Below is a comprehensive breakdown of the ZevihaNut/1.5 model, its new capabilities, and the exciting new directions it could lead to.


---

Key Features and Enhancements in ZevihaNut/1.5

1. Bug Fixes and Stability Improvements

One of the most notable improvements in ZevihaNut/1.5 is the fixing of numerous bugs that existed in previous versions. These issues ranged from processing errors to minor glitches in data handling. With these bugs addressed, ZevihaNut/1.5 is now far more stable and reliable for real-world applications.



2. Hidden Layer Support

To bring Zevihanthosa closer to advanced AI standards, ZevihaNut/1.5 introduces the concept of hidden layers in neural networks. This feature allows for deeper learning and more complex decision-making capabilities, making it suitable for more intricate tasks, such as advanced text classification, sentiment analysis, and more.

Hidden layers are an essential component of deep learning models, and their integration significantly improves the depth of understanding and pattern recognition of the model. This is a crucial step towards aligning ZevihaNut with state-of-the-art AI systems.



3. Multi-Language Support via Language Recognition and Translation

Language-specific inputs: In ZevihaNut/1.5, all chatbot inputs are now saved with a specific language tag, enabling seamless handling of multiple languages. This format change ensures that many languages can now be integrated into the same chatbot model without conflict.

Automatic Language Detection: A language detection function has been added to all chatbots, allowing the system to automatically detect the language of the input text, making the model more robust and adaptable to a global user base. This means that, even if users speak different languages, the AI can still operate effectively without manual intervention.

GoogleTranslator Class: To handle text translation and language-related tasks, ZevihaNut/1.5 includes the GoogleTranslator class, which allows for automatic translation between multiple languages. This is especially useful when integrating a chatbot into a global environment, ensuring it can communicate fluently in different languages. It also enhances the accuracy and effectiveness of text interpretation across various languages.



4. ONCEbende Chatbot AI

One of the most exciting additions in ZevihaNut/1.5 is the introduction of the ONCEbende chatbot AI. This is a pre-trained chatbot model that has been trained with a 1MB dataset, making it ready to deploy right out of the box. It uses advanced conversational techniques to simulate human-like interactions and can be easily integrated into Zevihanthosa-powered systems.

To implement the ONCEbende AI, users need to download the oncebende chatbot brain (a JSON file) from the GitHub repository and load it using the load_file method. Once integrated, ONCEbende can handle complex conversational tasks, making it an excellent choice for a wide range of chatbot applications.



5. Unicode Character-Based AI Models

As a result of the newly added features, Unicode character-based AI models are now considered to be low-level AI models. These models are capable of handling basic tasks but fall short in more advanced scenarios compared to models that utilize the new features in ZevihaNut/1.5, such as hidden layers and language recognition.



6. Increased Accessibility and Usability

ZevihaNut/1.5 makes Zevihanthosa much more user-friendly and accessible. The model is now easier to use and better suited for individuals and developers with less experience in AI. The improvements in architecture, language processing, and integration make it a more versatile solution for both beginners and experienced developers alike.





---

Upcoming Features and Potential Directions for ZevihaNut

ZevihaNut/1.5 paves the way for further innovations. Some of the possible future features include:

1. Simplified HuggingFace Client Support

The introduction of a HuggingFace client would provide a simplified way for users to interact with popular NLP models. This integration would open up access to a wide range of state-of-the-art pre-trained models, making it even easier to deploy powerful AI systems using Zevihanthosa.

HuggingFace is a widely used platform for transformer-based models, and adding client support would allow ZevihaNut to tap into the massive library of NLP models available on the platform.



2. ONCEbende-2

The development of ONCEbende-2 is highly anticipated. This would be an improved version of the ONCEbende chatbot model, likely featuring better conversational capabilities, more advanced natural language understanding, and greater adaptability to different contexts.



3. Logistic Regression Models for Industrial Applications

The introduction of Logistic Regression Models would make Zevihanthosa applicable in industrial environments, such as decision-making systems in human resources. For example, a logistic regression model could be used to predict whether a candidate should be hired based on various input factors. These models could also be adapted to other fields, like finance, healthcare, and more.



4. Lightweight Chatbot Models for Simple Tasks

ZevihaNut/1.5 may soon feature lightweight chatbot models that have an extremely simplified architecture. These models would be fast and capable of performing specific, less complex tasks like answering basic questions, managing simple interactions, and handling rudimentary queries. While they wouldn't include deep learning or neural network processing, they would be ideal for quick deployment and non-complex tasks.



5. Simple Text Translation Based on Similarity

A new function could be added to chatbots in ZevihaNut to provide a basic text translation service. This service would focus on finding the closest meaning in another language, simplifying text translation by focusing on semantic similarity. It would be useful for scenarios where precision is not as critical but basic translation is required.





---

Conclusion

ZevihaNut/1.5 is a revolutionary step forward for Zevihanthosa, adding a slew of new features and improving the overall performance and accessibility of the platform. By integrating powerful language processing, conversational AI (ONCEbende), and enhanced model architecture, this version is ready to tackle a wide array of applications ranging from chatbots to industrial decision-making.

Looking to the future, ZevihaNut/1.5 sets the stage for even greater advancements, including deeper NLP integration, more sophisticated AI models, and broader compatibility with external platforms like HuggingFace. With its increasing flexibility, ease of use, and growing capabilities, ZevihaNut/1.5 is undoubtedly a game-changer in the world of artificial intelligence.
"""

import random
import math
import numpy
import difflib
import json

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
		self.cells = cells
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
	def __init__(self, layers=[128, 64, 32], auto_layers=False, auto_layers_main=128, auto_layers_count=3, auto_layers_cutter=2, auto_layers_randomize=False, auto_layers_randomize_minlayercount=1, auto_layers_randomize_maxlayercount=5, auto_layers_randomize_minmain=64, auto_layers_randomize_maxmain=512, auto_layers_randomize_mincutter=1, auto_layers_randomize_maxcutter=3, randomize=False, truely=10, minvalue=-1, maxvalue=1, truely_cutter=1.5, momentum_default=0, kingvalue=None, hostkingvalue=None, hidden_layer=128):
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
		self.hidden_layer = Brain(cells=hidden_layer, minvalue=minvalue, maxvalue=maxvalue, randomize=randomize, truely=truely, momentum_default=momentum_default, kingvalue=kingvalue)
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
		return [layers, self.minvalue, self.maxvalue, self.truely, self.momentum_default, self.king.save(), self.hidden_layer.save()]
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
		self.hidden_layer.load(data[6])
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
	def process(self, value, target_value=0, train=True, activation_method="sigmoid", temperature=1, ignore_errors=False, decision_method="all", finally_decision_method="all"):
		processes = []
		for layer in self.layers:
			processes.append(layer.process(value, target_value=target_value, train=train, activation_method=activation_method, temperature=temperature, ignore_errors=ignore_errors, decision_method=decision_method))
		processes.append(self.hidden_layer.process(value, target_value=target_value, train=train, activation_method=activation_method, temperature=temperature, ignore_errors=ignore_errors, decision_method=decision_method))
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
	def __init__(self, cells=128, minvalue=-1, maxvalue=1, randomize=False, truely=10, momentum_default=0, kingvalue=None, type="basicwords", tokenizer=None):
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
	def complation(self, text, train=True, activation_method="sigmoid", temperature=None, ignore_errors=False, decision_method="all", vector="all", vector_temperature=None, vector_randomizer=92840198, vector_hash_truely=5, response_randomizer=17482736, response_temperature=None, maxlength=15, language=None):
		if language == None:
			language = self.detect_language(text)
		if language not in self.grammar:
			self.grammar[language] = {}
		truelytemp = 1/self.brain.truely
		if temperature == None:
			temperature = truelytemp
		if vector_temperature == None:
			vector_temperature = truelytemp
		if response_temperature == None:
			response_temperature = truelytemp
		text = self.tokenizer.tokenize(text)
		otext = text
		if vector == "grammarbased":
			newtext = []
			for h in text:
				if h in self.grammar[language]:
					newtext.append(self.grammar[language][h])
				else:
					canbe = []
					most = 0
					mostv = float("inf")
					for hh, v in self.grammar[language].items():
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
					hash += h.count(hh)
					hash += (1/(vector_hash_truely*10))
				newtext.append(h)
		elif vector == "all":
			newtext = []
			for h in text:
				hash = 0
				for hh in h:
					hash += ord(hh)/(1114112/vector_hash_truely)
					hash += h.count(hh)
					hash += (1/(vector_hash_truely*10))
				if h in self.grammar[language]:
					newtext.append((self.grammar[language][h]+hash)/2)
				else:
					canbe = []
					most = 0
					mostv = float("inf")
					for hh, v in self.grammar[language].items():
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
				self.grammar[language][hhh] = h
			most = ""
			mostv = float("inf")
			canbe = []
			for hh, v in self.grammar[language].items():
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
				for hh, v in self.grammar[grammar].items():
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
			return "".join(response[:maxlength])
		else:
			return " ".join(response[:maxlength])
	def auto_trainer(self, data, activation_method="sigmoid", temperature=None, ignore_errors=False, decision_method="all", vector="all", vector_temperature=None, vector_randomizer=92840198, vector_hash_truely=5, response_randomizer=17482736, response_temperature=None, maxlength=15, epoch=5, joiner=" . ", language=None):
		responses = []
		for _ in range(epoch):
			for message, response in data.items():
				text = message+joiner+response
				responses.append(self.complation(text, train=True, activation_method=activation_method, temperature=temperature, ignore_errors=ignore_errors, decision_method=decision_method, vector=vector, vector_temperature=vector_temperature, vector_randomizer=vector_randomizer, vector_hash_truely=vector_hash_truely, response_randomizer=response_randomizer, response_temperature=response_temperature, maxlength=maxlength, language=language))
		return responses
	def detect_language(self, text, truepower=50):
		best = "main"
		bestv = 0
		o2tok = self.tokenizer.tokenize(text)
		for lang, data in self.grammar.items():
			sameing = 0
			for o, v in data.items():
				for o2 in o2tok:
					for oo in o.split():
						for oo2 in o2.split():
							if difflib.SequenceMatcher(None, oo, oo2).ratio() >= 1/truepower:
								sameing += 1
			if sameing >= bestv:
				best = lang
				bestv = sameing
		return best
class AdvancedChatbot:
	def __init__(self, layers=[128, 64, 32], auto_layers=False, auto_layers_main=128, auto_layers_count=3, auto_layers_cutter=2, auto_layers_randomize=False, auto_layers_randomize_minlayercount=1, auto_layers_randomize_maxlayercount=5, auto_layers_randomize_minmain=64, auto_layers_randomize_maxmain=512, auto_layers_randomize_mincutter=1, auto_layers_randomize_maxcutter=3, randomize=False, truely=10, minvalue=-1, maxvalue=1, truely_cutter=1.5, momentum_default=0, kingvalue=None, hostkingvalue=None, type="basicwords", tokenizer=None, hidden_layer=128):
		self.brain = AdvancedBrain(layers=layers, auto_layers=auto_layers, auto_layers_main=auto_layers_main, auto_layers_count=auto_layers_count, auto_layers_cutter=auto_layers_cutter, auto_layers_randomize=auto_layers_randomize, auto_layers_randomize_minlayercount=auto_layers_randomize_minlayercount, auto_layers_randomize_maxlayercount=auto_layers_randomize_maxlayercount, auto_layers_randomize_minmain=auto_layers_randomize_minmain, auto_layers_randomize_maxmain=auto_layers_randomize_maxmain, auto_layers_randomize_mincutter=auto_layers_randomize_mincutter, auto_layers_randomize_maxcutter=auto_layers_randomize_maxcutter, randomize=randomize, truely=truely, minvalue=minvalue, maxvalue=maxvalue, truely_cutter=truely_cutter, momentum_default=momentum_default, kingvalue=kingvalue, hostkingvalue=hostkingvalue, hidden_layer=hidden_layer)
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
	def complation(self, text, train=True, activation_method="sigmoid", temperature=None, ignore_errors=False, decision_method="all", vector="all", vector_temperature=None, vector_randomizer=92840198, vector_hash_truely=5, response_randomizer=17482736, response_temperature=None, maxlength=15, finally_decision_method="all", language=None):
		if language == None:
			language = self.detect_language(text)
		if language not in self.grammar:
			self.grammar[language] = {}
		truelytemp = 1/self.brain.truely
		if temperature == None:
			temperature = truelytemp
		if vector_temperature == None:
			vector_temperature = truelytemp
		if response_temperature == None:
			response_temperature = truelytemp
		text = self.tokenizer.tokenize(text)
		otext = text
		if vector == "grammarbased":
			newtext = []
			for h in text:
				if h in self.grammar[language]:
					newtext.append(self.grammar[language][h])
				else:
					canbe = []
					most = 0
					mostv = float("inf")
					for hh, v in self.grammar[language].items():
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
					hash += h.count(hh)
					hash += (1/(vector_hash_truely*10))
				newtext.append(h)
		elif vector == "all":
			newtext = []
			for h in text:
				hash = 0
				for hh in h:
					hash += ord(hh)/(1114112/vector_hash_truely)
					hash += h.count(hh)
					hash += (1/(vector_hash_truely*10))
				if h in self.grammar[language]:
					newtext.append((self.grammar[language][h]+hash)/2)
				else:
					canbe = []
					most = 0
					mostv = float("inf")
					for hh, v in self.grammar[language].items():
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
				self.grammar[language][hhh] = h
			most = ""
			mostv = float("inf")
			canbe = []
			for hh, v in self.grammar[language].items():
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
				for hh, v in self.grammar[language].items():
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
			return "".join(response[:maxlength])
		else:
			return " ".join(response[:maxlength])
	def auto_trainer(self, data, activation_method="sigmoid", temperature=None, ignore_errors=False, decision_method="all", vector="all", vector_temperature=None, vector_randomizer=92840198, vector_hash_truely=5, response_randomizer=17482736, response_temperature=None, maxlength=15, finally_decision_method="all", epoch=5, joiner=" . ", language=None):
		responses = []
		for _ in range(epoch):
			for message, response in data.items():
				text = message+joiner+response
				responses.append(self.complation(text, activation_method=activation_method, temperature=temperature, ignore_errors=ignore_errors, decision_method=decision_method, vector=vector, vector_temperature=vector_temperature, vector_randomizer=vector_randomizer, vector_hash_truely=vector_hash_truely, response_randomizer=response_randomizer, response_temperature=response_temperature, maxlength=maxlength, finally_decision_method=finally_decision_method, language=language))
		return responses
	def detect_language(self, text, truepower=100):
		best = "main"
		bestv = 0
		o2tok = self.tokenizer.tokenize(text)
		for lang, data in self.grammar.items():
			sameing = 0
			for o, v in data.items():
				for o2 in o2tok:
					for oo in o.split():
						for oo2 in o2.split():
							if difflib.SequenceMatcher(None, oo, oo2).ratio() >= 1/truepower:
								sameing += 1
			if sameing >= bestv:
				best = lang
				bestv = sameing
		return best
class ContextManager:
	def __init__(self, maxcontext=100):
		self.context = []
		self.maxcontext = maxcontext
	def save(self):
		return [self.context, self.maxcontext]
	def load(self, data):
		self.context = data[0]
		self.maxcontext = data[1]
	def restore(self):
		self.context = self.context[:self.maxcontext]
	def add_context(self, data, value=0):
		self.context = [[data, value]]+self.context
		self.restore()
	def remove_context(self, data):
		context = []
		for h in self.context:
			if h[0] != data:
				context.append(h)
		self.context = context
		self.restore()
	def remove_last_context(self):
		self.context = self.context[1:]
		self.restore()
	def find_context(self, data, value=0, usevalue=False, indata=True, basicscan=True, temperature=0.5, valuepower=3):
		contexts = []
		for t_data, t_value in self.context:
			if indata:
				if data in t_data:
					contexts.append(t_data)
			seq = 0
			seqmax = 0
			if basicscan:
				for h, h2 in zip(t_data, data):
					if h.lower() == h2.lower():
						seq += 1
					seqmax += 1
				for h, h2 in zip(t_data.split(), data.split()):
					if h.lower() == h2.lower():
						seq += 1
					seqmax += 1
			if usevalue:
				seqmax += abs(value-t_value)*valuepower
			if seq/seqmax >= 1-temperature:
				contexts.append(t_data)
		return contexts
class GoogleTranslator:
	def __init__(self):
		# pip install googletrans==4.0.0-rc1
		from googletrans import Translator
		self.engine = Translator()
		self.saver = True
		self.saver_data = {}
	def detect_language(self, text):
		if self.saver:
			if text in self.saver_data:
				lang = self.saver_data[text]
			else:
				lang = self.engine.detect(text).lang.lower()
		else:
			lang = self.engine.detect(text).lang.lower()
		if self.saver:
			self.saver_data[text] = lang
		return lang
	def translate(self, text, dest):
		idd = (text, dest)
		if self.saver:
			if idd in self.saver_data:
				new = self.saver_data[idd]
			else:
				new = self.engine.translate(text, dest=dest).text
			self.saver_data[idd] = new
		else:
			new = self.engine.translate(text, dest=dest).text
		return new
def save_file(object, filename):
	with open(filename, "w") as f:
		f.write(json.dumps({"data": object.save()}))
		f.flush()
def load_file(object, filename):
	with open(filename, "r") as f:
		object.load(json.loads(f.read())["data"])
