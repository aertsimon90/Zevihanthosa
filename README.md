# Zevihanthosa

## What is Zevihanthosa?

Zevihanthosa is an advanced artificial intelligence framework developed by **aertsimon90**. It is designed to deliver a flexible, beginner-friendly, and highly efficient AI architecture. Unlike heavyweight frameworks like TensorFlow or PyTorch, Zevihanthosa provides a streamlined yet powerful environment for AI development, enabling fine-tuned control without overwhelming complexity.

At its core, Zevihanthosa features a dynamic, weight-based learning process enhanced by a momentum-driven training mechanism, which accelerates learning and boosts accuracy. The framework supports custom activation functions, multi-layer neural architectures, and diverse decision-making strategies, making it adaptable to a wide range of applications, from chatbots to experimental AI systems.

With a focus on scalability, usability, and efficiency, Zevihanthosa continues to evolve, incorporating cutting-edge improvements with each release.

---

## Current Models and Versions

### Latest Release: ZevihaNut/1.6

ZevihaNut/1.6 builds on its predecessors, enhancing Zevihanthosa’s capabilities and accessibility. This version refines the framework’s neural network components, chatbot functionality, and text-processing tools. Below are the key features introduced or improved in ZevihaNut/1.6:

### Key Features in ZevihaNut/1.6

**1. Enhanced Chatbot AI Models**  
ZevihaNut/1.6 includes robust `Chatbot` and `AdvancedChatbot` classes, leveraging the `Brain` and `AdvancedBrain` neural architectures. These models support dynamic text generation with customizable tokenization, vectorization, and response strategies.

**2. Multi-Layer Neural Network Support**  
The `AdvancedBrain` class introduces multi-layer neural networks with configurable layers (e.g., `[128, 64, 32]`) and an additional hidden layer. This enables deeper abstraction and improved learning efficiency compared to the single-layer `Brain`.

**3. Custom Activation Functions**  
A standout feature is the proprietary "zevianthosa" activation function:  
`(1/(1 + ((value)/weight)^(-(value + (momentum/weight)))))`  
Alongside standard options like sigmoid, tanh, and ReLU, this function enhances the model’s adaptability and learning dynamics.

**4. Improved Tokenization System**  
The `Tokenizer` class supports multiple tokenization methods (e.g., `words`, `chars`, `bottombasedwords`), allowing flexible text preprocessing tailored to specific tasks.

**5. Language Detection and Multilingual Support**  
Built-in language detection in chatbot classes uses word similarity metrics. The `GoogleTranslator` class provides optional integration with Google Translate for language detection and translation, supporting multilingual applications.

**6. Dynamic Neuron Selection with Temperature**  
The "temperature" parameter in `Brain` and `AdvancedBrain` dynamically controls which neurons process input, acting as an attention-like mechanism to improve efficiency and context sensitivity.

**7. Context Management**  
The `ContextManager` class has been refined to store and retrieve conversation history effectively, with customizable similarity thresholds (`temperature`) for context matching.

**8. Automatic Data Persistence**  
AI models support seamless saving and loading of their states (e.g., weights, grammar) via `save` and `load` methods, serialized in JSON format using `save_file` and `load_file`.

**9. Beginner-Friendly Design**  
ZevihaNut/1.6 retains its focus on usability, offering simplified methods like `process_noob` and `complation` for quick experimentation while preserving advanced customization options.

---

### Upcoming Features (Possible Future Additions)

While ZevihaNut/1.6 is a robust release, potential future enhancements could include:  
1. **Integration with External APIs**: Such as Hugging Face for pre-trained model compatibility.  
2. **Optimized Lightweight Models**: Chatbots without neural networks for faster, resource-efficient applications.  
3. **Advanced Training Algorithms**: Incorporation of techniques like gradient clipping or adaptive learning rates.  
4. **Enhanced Translation**: Improved nearest-meaning translation for more natural chatbot responses.  
5. **Industrial Use Cases**: Specialized models for tasks like logistic regression or decision-making.

---

## Is Zevihanthosa Open Source?

Yes! Zevihanthosa is fully open-source, encouraging developers to contribute, modify, and integrate it into their projects. The aim is to democratize cutting-edge AI development and foster a collaborative community.

---

## Installation & Usage

To get started with Zevihanthosa, follow these steps:

### Step 1: Install Git  
If Git isn’t installed, download it by searching "GIT Download" online.  
For Linux users:  
```bash
apk install git
```

### Step 2: Clone the Repository  
Open a terminal and run:  
```bash
git clone https://github.com/aertsimon90/Zevihanthosa
```

### Step 3: Navigate to the Repository  
By default, the terminal opens in your working directory.  
For Windows users, navigate to the desktop:  
```bash
cd Desktop
```  
Enter the Zevihanthosa folder:  
```bash
cd Zevihanthosa
```

### Step 4: Importing Zevihanthosa in Python  
1. Check the `README.md` file for the latest version.  
2. Locate the model file in the repository.  
3. **File Naming Convention**:  
   Model files follow this format:  
   **zevihanthosa_ModelName_Version-VersionDetail.py**  
   To import, replace `-` with `_`:  
   ```python
   import zevihanthosa_ModelName_Version_VersionDetail
   ```  
   Alternatively, rename the file (e.g., `custom.py`) and import:  
   ```python
   import custom
   ```

### Step 5: Using Zevihanthosa  
- Explore the source code to understand its structure.  
- Test functions like `Chatbot.complation` or `AdvancedBrain.process` to experiment with its capabilities.  
- Example usage:  
  ```python
  from zevihanthosa import Chatbot, save_file, load_file
  chatbot = Chatbot(cells=128, type="basicwords")
  data = {"Hello": "Hi", "How are you": "Good"}
  chatbot.auto_trainer(data, epoch=5)
  print(chatbot.complation("Hello"))  # Outputs "Hi" or similar
  save_file(chatbot, "chatbot_model.json")
  ```

### Step 6: Running AI Models  
Train, test, and customize Zevihanthosa for your needs. Refer to the GitHub documentation for troubleshooting.

---

### Contact Information

For inquiries, contributions, or support, contact:  
**Email**: simon.scap090@gmail.com
