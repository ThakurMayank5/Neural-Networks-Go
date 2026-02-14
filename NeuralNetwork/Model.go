package neuralnetwork

import (
	"fmt"

	"github.com/ThakurMayank5/Neural-Networks-Go/vectors"
)

// ActivationFunction represents the type of activation function
type ActivationFunction string

const (
	ReLU    ActivationFunction = "relu"
	Sigmoid ActivationFunction = "sigmoid"
	Tanh    ActivationFunction = "tanh"
)

// Optimizer represents the optimization algorithm
type Optimizer string

// LossFunction represents the loss function
type LossFunction string

// Initialization represents weight initialization strategy
type Initialization string

const (
	XavierUniformInitializer  Initialization = "xavier_uniform"
	XavierNormalInitializer   Initialization = "xavier_normal"
	KaimingUniformInitializer Initialization = "kaiming_uniform"
	KaimingNormalInitializer  Initialization = "kaiming_normal"
)

// InputLayer represents the input layer configuration
type InputLayer struct {
	Neurons            int
	ActivationFunction ActivationFunction
}

// OutputLayer represents the output layer configuration
type OutputLayer struct {
	Neurons            int
	ActivationFunction ActivationFunction
	Initialization     Initialization
}

// Dataset represents training data
type Dataset struct {
	Inputs  [][]float64
	Outputs [][]float64
}

// TrainingConfig represents training hyperparameters
type TrainingConfig struct {
	Epochs          int
	LearningRate    float64
	Optimizer       Optimizer
	LossFunction    LossFunction
	BatchSize       int
	ValidationSplit float64
}

// ModelWeightsAndBiases stores the model parameters
type ModelWeightsAndBiases struct {
	Weights [][]float64
	Biases  [][]float64
}

// Layer represents a hidden layer
type Layer struct {
	Neurons            int
	ActivationFunction ActivationFunction
	Initialization     Initialization
}

// NeuralNetwork represents the neural network architecture
type NeuralNetwork struct {
	InputLayer       InputLayer
	Layers           []Layer
	OutputLayer      OutputLayer
	WeightsAndBiases ModelWeightsAndBiases
}

// Model represents the complete model with network and training config
type Model struct {
	NeuralNetwork  NeuralNetwork
	TrainingConfig TrainingConfig
}

// AddLayer adds a hidden layer to the neural network
func (nn *NeuralNetwork) AddLayer(layer Layer) {
	nn.Layers = append(nn.Layers, layer)
}

// SetOutputLayer sets the output layer configuration
func (nn *NeuralNetwork) SetOutputLayer(layer OutputLayer) {
	nn.OutputLayer = layer
}

// SetInputLayer sets the input layer configuration
func (nn *NeuralNetwork) SetInputLayer(layer InputLayer) {
	nn.InputLayer = layer
}

// Predict performs forward propagation
func (nn *NeuralNetwork) Predict(input []float64, weightsAndBiases ModelWeightsAndBiases) ([]float64, error) {

	println("Predicting...")

	weights := weightsAndBiases.Weights
	biases := weightsAndBiases.Biases

	x := input

	println(len(weights))
	println(len(biases))

	for i := range nn.Layers {

		newX := make([]float64, nn.Layers[i].Neurons) // Initialized elements as 0

		// Processing Layer 1
		if i == 0 {
			println("New X", newX)

			println("Processing Input Layer")

			println("Input length:", len(x))
			println("Layer neurons", nn.Layers[i].Neurons)

			println("Input Layer Neurons", nn.InputLayer.Neurons)

			println("Weights length:", len(weights[i]), "Expected:", nn.InputLayer.Neurons*nn.Layers[i].Neurons)

			if len(weights) == 0 || len(biases) == 0 {
				fmt.Println("Weights and biases are not initialized.")
				return nil, fmt.Errorf("weights and biases are not initialized")
			}

			for j := 0; j < nn.Layers[i].Neurons; j++ {
				currWeights := weights[i][j*len(x) : (j+1)*len(x)]
				dotProduct, err := vectors.DotProduct(x, currWeights)

				println(dotProduct)

				// testing weights
				fmt.Printf("Layer %d, Neuron %d, Weights: %v\n", i+1, j+1, currWeights)

				if err != nil {
					fmt.Println("Error computing dot product:", err)
					return nil, err
				}

				continue
			}

			println("Processing Layer", i+1)

			println("Input length:", len(x))
			println("Layer neurons", nn.Layers[i].Neurons)

			println(len(weights[i]), nn.Layers[i].Neurons*len(x))

			if len(weights) == 0 || len(biases) == 0 {
				fmt.Println("Weights and biases are not initialized.")
				return nil, fmt.Errorf("weights and biases are not initialized")
			}

			if len(weights[i]) != len(x)*nn.Layers[i].Neurons && i != len(nn.Layers) {

				println("Weights:", len(weights[i]))
				println("Input length:", len(x))

				fmt.Println("Mismatch in weights length.")
				return nil, fmt.Errorf("mismatch in weights length")
			}
		}
	}
	return x, nil
}

// Summary prints the neural network architecture
func (nn *NeuralNetwork) Summary() {

	TotalLayers := len(nn.Layers) + 2 // Input and Output layers

	fmt.Println("Neural Network Summary:")
	fmt.Println("Total Layers:", TotalLayers)
	fmt.Println("Input Layer Neurons:", nn.InputLayer.Neurons)
	fmt.Println("Input Layer Activation Function:", nn.InputLayer.ActivationFunction)

	for i, layer := range nn.Layers {
		fmt.Println("Layer", i+1, "Neurons:", layer.Neurons)
		fmt.Println("Layer", i+1, "Activation Function:", layer.ActivationFunction)
	}

	fmt.Println("Output Layer Neurons:", nn.OutputLayer.Neurons)
	fmt.Println("Output Layer Activation Function:", nn.OutputLayer.ActivationFunction)

}

// Fit trains the model on the provided dataset
func (model *Model) Fit(dataset Dataset) error {

	// Dataset validation

	if len(dataset.Inputs) == 0 || len(dataset.Outputs) == 0 {
		return fmt.Errorf("dataset is empty")
	}

	if len(dataset.Inputs) != len(dataset.Outputs) {
		return fmt.Errorf("number of inputs and outputs must be the same")
	}

	if len(dataset.Inputs[0]) != model.NeuralNetwork.InputLayer.Neurons {
		return fmt.Errorf("input data does not match the number of neurons in the input layer")
	}

	totalLayers := len(model.NeuralNetwork.Layers) + 2 // Input and Output layers

	totalTrainableLayers := totalLayers - 1 // Exclude input layer

	fmt.Printf("Total Layers: %d\n", totalLayers)
	fmt.Printf("Total Trainable Layers: %d\n", totalTrainableLayers)

	weights := make([][]float64, totalTrainableLayers)

	println(len(weights))

	for i := range weights {

		if i == 0 {

			weights[i] = make([]float64, model.NeuralNetwork.InputLayer.Neurons*model.NeuralNetwork.Layers[i].Neurons)

			continue
		}

		if i == totalTrainableLayers-1 {

			weights[i] = make([]float64, model.NeuralNetwork.Layers[i-1].Neurons*model.NeuralNetwork.OutputLayer.Neurons)

			continue
		}

		weights[i] = make([]float64, model.NeuralNetwork.Layers[i-1].Neurons*model.NeuralNetwork.Layers[i].Neurons)

	}

	biases := make([][]float64, totalTrainableLayers)

	for i := range biases {
		if i == totalTrainableLayers-1 {

			biases[i] = make([]float64, model.NeuralNetwork.OutputLayer.Neurons)
			continue
		}
		biases[i] = make([]float64, model.NeuralNetwork.Layers[i].Neurons)
	}

	// Bias initialization
	for i := range biases {
		for j := range biases[i] {
			biases[i][j] = 0.0
		}
	}

	for i := range biases {
		for j := range biases[i] {
			fmt.Printf("Bias[%d][%d]: %f\n", i, j, biases[i][j])
		}
	}

	// Testing the weights and biases initialization
	for i := range weights {
		for j := range weights[i] {
			weights[i][j] = 0.01 * float64(i+j) // Example initialization
		}
	}

	// testing the weights and biases initialization
	for i := range weights {
		for j := range weights[i] {
			fmt.Printf("Weight[%d][%d]: %f\n", i, j, weights[i][j])
		}
	}

	for epoch := 1; epoch <= model.TrainingConfig.Epochs; epoch++ {
		fmt.Printf("Epoch %d/%d\n", epoch, model.TrainingConfig.Epochs)
	}

	// Forward pass, loss calculation, backward pass, and weights update

	// Forward pass

	for i := range dataset.Inputs {
		input := dataset.Inputs[i]
		output, err := model.NeuralNetwork.Predict(input, ModelWeightsAndBiases{Weights: weights, Biases: biases})
		if err != nil {
			fmt.Printf("Error predicting output for input %v: %v\n", input, err)

			return err

		}
		fmt.Printf("Input: %v, Predicted Output: %v\n", input, output)
	}

	return nil

}

// InitializeWeights initializes the model weights and biases
func (model *Model) InitializeWeights() error {

	totalLayers := len(model.NeuralNetwork.Layers) + 2 // Input and Output layers

	totalTrainableLayers := totalLayers - 1 // Exclude input layer

	biases := make([][]float64, totalTrainableLayers)

	for i := range biases {
		if i == totalTrainableLayers-1 {

			biases[i] = make([]float64, model.NeuralNetwork.OutputLayer.Neurons)
			continue
		}
		biases[i] = make([]float64, model.NeuralNetwork.Layers[i].Neurons)
	}

	// Bias initialization
	for i := range biases {
		for j := range biases[i] {
			biases[i][j] = 0.0
		}
	}

	model.NeuralNetwork.WeightsAndBiases.Biases = biases

	// Initialize weights using Kaiming Normal initialization
	err := KaimingNormal(&model.NeuralNetwork)
	if err != nil {
		return err
	}

	println("Weights:", model.NeuralNetwork.WeightsAndBiases.Weights)

	// print shape of weights
	for i := range model.NeuralNetwork.WeightsAndBiases.Weights {
		fmt.Printf("Weights for Layer %d: %d\n", i+1, len(model.NeuralNetwork.WeightsAndBiases.Weights[i]))
	}

	// print shape of biases
	for i := range model.NeuralNetwork.WeightsAndBiases.Biases {
		fmt.Printf("Biases for Layer %d: %d\n", i+1, len(model.NeuralNetwork.WeightsAndBiases.Biases[i]))
	}

	return nil
}

func println(args ...interface{}) {
	fmt.Println(args...)
}
