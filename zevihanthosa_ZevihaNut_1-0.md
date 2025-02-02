# ZevihaNut/1.0 Model - Class and Function Arguments

## Classes

### Cell Class
The `Cell` class represents a single unit (neuron) in the Zevianthosa model. It contains properties for weights, learning parameters, and momentum, among others. It also defines various functions to manage the learning process.

#### Arguments:
- `weight`: The initial weight of the cell, defaults to a random value between -1 and 1.
- `truely`: The divisor for adjusting the learning rate, defaults to 10.
- `momentum_default`: The default value for momentum, defaults to 0.

#### Functions:
- `save()`: Saves the current state of the cell (weight, learning rate, momentum, truely).
- `load(data)`: Loads the saved state of the cell from a list of data.
- `stabilizer(value, alpha=0.1)`: Stabilizes the weight by adjusting it based on a given value and a smoothing factor (`alpha`).
- `activation(value, method)`: Applies an activation function to the value. Supported methods include:
  - `"sigmoid"`
  - `"tanh"`
  - `"relu"`
  - `"abs"`
  - `"sin"`
  - `"cos"`
  - `"zevianthosa"`
- `process(value, target_value=0, train=True, activation_method="sigmoid")`: Processes the value by multiplying with weight, applying activation, and adjusting weight based on the error (during training). It uses:
  - `value`: The input value to be processed.
  - `target_value`: The expected target value (used in training).
  - `train`: Whether the cell should adjust its parameters during training.
  - `activation_method`: The activation function to apply.

### Brain Class
The `Brain` class represents a collection of cells that work together to form the basic neural network unit in Zevianthosa.

#### Arguments:
- `cells`: The number of cells in the brain, defaults to 128.
- `minvalue`: The minimum value for cell weights, defaults to -1.
- `maxvalue`: The maximum value for cell weights, defaults to 1.
- `randomize`: If set to `True`, cells will be initialized with random values.
- `truely`: Adjusts the learning rate for the cells, defaults to 10.
- `momentum_default`: The default momentum value for the cells.
- `kingvalue`: The weight of the "king" cell, which controls the learning behavior.

#### Functions:
- `save()`: Saves the state of the brain, including cells and king cell.
- `load(data)`: Loads the saved state of the brain.
- `clusters_to_temperature(clusters)`: Converts cluster count into a temperature value.
- `stabilizer(value, alpha=0.1, temperature=1)`: Stabilizes the weights of the brain by adjusting them towards a given value.
- `activation(value, method)`: Applies an activation function to the value. Same as `Cell` class.
- `momentum_set(value, temperature=1)`: Sets the momentum for all cells in the brain.
- `weight_set(value, temperature=1)`: Sets the weight for all cells in the brain.
- `learning_set(value, temperature=1)`: Sets the learning rate for all cells.
- `process(value, target_value=0, train=True, activation_method="sigmoid", temperature=1, ignore_errors=False, decision_method="all")`: Processes the input value, optionally training the cells and making decisions based on the error. It uses:
  - `value`: The input value.
  - `target_value`: The expected output.
  - `train`: Whether to train the model or not.
  - `activation_method`: The method for activation.
  - `temperature`: Adjusts the "warmth" of learning.
  - `ignore_errors`: Whether to ignore errors during processing.
  - `decision_method`: Defines how the outputs of the cells are aggregated, can be `"sum"`, `"average"`, or `"all"`.

### AdvancedBrain Class
The `AdvancedBrain` class extends `Brain` by introducing layers of brains, each with a different configuration and size. This class is designed for more complex neural network architectures.

#### Arguments:
- `layers`: A list of integers, each representing the number of cells in a layer.
- `auto_layers`: Whether to automatically generate layers.
- `auto_layers_main`: The base number of cells in the main layer when generating layers automatically.
- `auto_layers_count`: The number of layers to generate automatically.
- `auto_layers_cutter`: The factor by which the number of cells is reduced for each layer.
- `randomize`: If `True`, cells are randomized at the start.
- `truely`: Adjusts the learning rate, defaults to 10.
- `momentum_default`: The default momentum value for the layers.
- `kingvalue`: The weight for the "king" cell of the first layer.
- `hostkingvalue`: The weight for the "king" cell of the host layer.
```Extra: auto_layers_randomize=False, auto_layers_randomize_minlayercount=1, auto_layers_randomize_maxlayercount=5, auto_layers_randomize_minmain=64, auto_layers_randomize_maxmain=512, auto_layers_randomize_mincutter=1, auto_layers_randomize_maxcutter=3```
#### Functions:
- `save()`: Saves the state of the advanced brain, including all layers and king cell.
- `load(data)`: Loads the saved state of the advanced brain.
- `clusters_to_temperature(clusters)`: Converts cluster count to temperature.
- `stabilizer(value, alpha=0.1, temperature=1)`: Stabilizes weights for all layers.
- `activation(value, method)`: Applies an activation function to the value.
- `momentum_set(value, temperature=1)`: Sets momentum for all layers.
- `weight_set(value, temperature=1)`: Sets weight for all layers.
- `learning_set(value, temperature=1)`: Sets learning rate for all layers.
- `process(value, target_value=0, train=True, activation_method="sigmoid", temperature=1, ignore_errors=False, decision_method="all", finally_decision_method="all")`: Processes the input value through all layers, making a final decision.
  - `value`: The input value.
  - `target_value`: The expected output.
  - `train`: Whether to train the network.
  - `activation_method`: The activation function to use.
  - `temperature`: Adjusts the learning process.
  - `ignore_errors`: Whether to ignore errors during processing.
  - `decision_method`: How intermediate layer outputs are combined.
  - `finally_decision_method`: How the final output is determined.

## Summary of Function Arguments

- **`value`**: The input value to process or train.
- **`target_value`**: The expected output value (used during training).
- **`train`**: Boolean flag indicating whether to perform training (weight updates).
- **`activation_method`**: Defines the activation function to use during processing (e.g., `sigmoid`, `relu`, `tanh`).
- **`temperature`**: A parameter to adjust the learning or stability behavior.
- **`ignore_errors`**: If `True`, errors during processing are skipped.
- **`decision_method`**: Determines how multiple neuron outputs are combined (e.g., `sum`, `average`, `all`).
- **`finally_decision_method`**: Determines how the final output is combined after all layers are processed.

These arguments allow for flexible manipulation of the neural network's behavior and performance, enabling Zevianthosa to adapt to a wide variety of tasks and scenarios.
