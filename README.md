# gonn — Neural Networks in Go

A general-purpose feedforward neural network framework built from scratch in Go. No ML libraries, no external dependencies — just pure Go.

> Wanna Know why? because I was bored.

---

## Features

- Fully configurable feedforward neural network (any depth, any width)
- Mini-batch gradient descent with correct simultaneous weight & bias updates
- Activation functions: **ReLU**, **Sigmoid**, **Tanh**, **Softmax**
- Loss functions: **Mean Squared Error**, **Categorical Cross-Entropy**
- Weight initializers: **Xavier Uniform**, **Xavier Normal**, **Kaiming Uniform**, **Kaiming Normal**
- Per-epoch validation loss reporting
- Dataset shuffling per epoch
- Built-in MNIST CSV loader (optional — bring your own data)

---

## Installation

```bash
go get github.com/ThakurMayank5/gonn
```

---

## Quick Start

```go
import (
    activ "github.com/ThakurMayank5/gonn/activation"
    nn    "github.com/ThakurMayank5/gonn/neuralnetwork"
)
```

### 1. Define a Model

```go
model := nn.Model{
    NeuralNetwork: nn.NeuralNetwork{
        InputLayer: nn.InputLayer{
            Neurons: 4, // number of input features
        },
        OutputLayer: nn.OutputLayer{
            Neurons:            3,              // number of output classes
            ActivationFunction: activ.Softmax,
            Initialization:     nn.XavierNormalInitializer,
        },
    },
    TrainingConfig: nn.TrainingConfig{
        Epochs:          50,
        LearningRate:    0.01,
        Optimizer:       "sgd",
        LossFunction:    "categorical_crossentropy", // or "mse"
        BatchSize:       32,
        ValidationSplit: 0.2,
    },
}
```

### 2. Add Hidden Layers

Add as many hidden layers as you want, in order from input to output:

```go
model.NeuralNetwork.AddLayer(nn.Layer{
    Neurons:            64,
    ActivationFunction: activ.ReLU,
    Initialization:     nn.KaimingNormalInitializer,
})

model.NeuralNetwork.AddLayer(nn.Layer{
    Neurons:            32,
    ActivationFunction: activ.Sigmoid,
    Initialization:     nn.XavierNormalInitializer,
})
```

**Activation function options:**

| Constant        | Description                         |
| --------------- | ----------------------------------- |
| `activ.ReLU`    | Rectified Linear Unit               |
| `activ.Sigmoid` | Sigmoid (good for hidden layers)    |
| `activ.Tanh`    | Hyperbolic tangent                  |
| `activ.Softmax` | Softmax (output layer, multi-class) |

**Initializer options:**

| Constant                       | Use with       |
| ------------------------------ | -------------- |
| `nn.XavierUniformInitializer`  | Sigmoid / Tanh |
| `nn.XavierNormalInitializer`   | Sigmoid / Tanh |
| `nn.KaimingUniformInitializer` | ReLU           |
| `nn.KaimingNormalInitializer`  | ReLU           |

### 3. Initialize Weights

```go
err := model.InitializeWeights()
if err != nil {
    log.Fatal(err)
}
```

### 4. Prepare Your Dataset

```go
dataset := nn.Dataset{
    Inputs:  [][]float64{ /* your input vectors */ },
    Outputs: [][]float64{ /* your target vectors (one-hot for classification) */ },
}
```

- Each element of `Inputs` is a `[]float64` of length equal to `InputLayer.Neurons`.
- Each element of `Outputs` is a `[]float64` of length equal to `OutputLayer.Neurons`.
- For classification, use one-hot encoded targets.
- For regression, use raw float targets and `"mse"` as the loss function.

### 5. Train

```go
err = model.Fit(dataset, dataset) // pass a separate validation set as the second arg
if err != nil {
    log.Fatal(err)
}
```

`Fit` will print epoch and validation loss to stdout.

### 6. Predict

```go
output, err := model.NeuralNetwork.Predict(inputVector)
// output is a []float64 of length OutputLayer.Neurons
```

### 7. Print Architecture Summary

```go
model.NeuralNetwork.Summary()
```

---

## Project Structure

```
gonn/
├── main.go                        # Example — MNIST digit classification
├── data.csv                       # Example dataset (MNIST CSV)
├── activation/
│   └── activations.go             # ReLU, Sigmoid, Tanh, Softmax
├── losses/
│   └── compute.go                 # MSE, Categorical Cross-Entropy
├── vectors/
│   └── vectors.go                 # Dot product and vector utilities
├── gpuprocessing/
│   └── vectors.go                 # GPU vector ops (experimental)
├── cuda/
│   └── dotproduct.cu              # CUDA kernel for dot product
└── neuralnetwork/
    ├── model.go                   # Core types: Model, Layer, NeuralNetwork, Dataset
    ├── initializers.go            # Weight initialization strategies
    ├── datasetloader.go           # MNIST CSV loader (optional utility)
    ├── training.go                # Fit loop, epoch management, shuffling
    ├── batch.go                   # PredictBatch — forward pass, captures z and a
    ├── backpropogation.go         # Backpropagation, mini-batch gradient descent
    ├── predict.go                 # Single-sample inference
    ├── evaluation.go              # ForwardPassBatch — loss computation
    ├── validation.go              # Validation loss
    └── shuffler.go                # Dataset shuffling utilities
```

---

## Example: MNIST Digit Classification

The included `main.go` trains a network on the MNIST handwritten digit dataset (60,000 samples, 28×28 grayscale images flattened to 784 inputs, 10 output classes).

| Layer  | Neurons | Activation | Initializer    |
| ------ | ------- | ---------- | -------------- |
| Input  | 784     | —          | —              |
| Hidden | 128     | Sigmoid    | Xavier Normal  |
| Hidden | 64      | Sigmoid    | Xavier Normal  |
| Output | 10      | Softmax    | Kaiming Normal |

**Training results (20 epochs, batch size 64, lr=0.01):**

| Epoch | Validation Loss |
| ----- | --------------- |
| 1     | 2.1211          |
| 5     | 0.9027          |
| 10    | 0.5283          |
| 15    | 0.4171          |
| 20    | 0.3653          |

Loss decreased consistently across all 20 epochs with no divergence. Most high-confidence predictions (>0.90) were correct, and the two misclassifications both had noticeably lower confidence scores — the model's uncertainty correlates with its errors.

To run the MNIST example, place a `data.csv` in the root (each row: `label, pixel0, ..., pixel783`) and run:

```bash
go run .
```

The loader normalizes pixel values to `[0, 1]` automatically.

---

## License

MIT
