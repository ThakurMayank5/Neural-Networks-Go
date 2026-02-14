package NeuralNetwork

import (
	"fmt"
	// "github.com/ThakurMayank5/Neural-Networks-Go/Activation"
	"github.com/ThakurMayank5/Neural-Networks-Go/Vectors"
)

type ActivationFunction string

const (
	ReLU    ActivationFunction = "relu"
	Sigmoid ActivationFunction = "sigmoid"
	Tanh    ActivationFunction = "tanh"
)

type Optimizer string

type LossFunction string

type InputLayer struct {
	Neurons            int
	ActivationFunction ActivationFunction
}

type OutputLayer struct {
	Neurons            int
	ActivationFunction ActivationFunction
}

type Dataset struct {
	Inputs  [][]float64
	Outputs [][]float64
}

type TrainingConfig struct {
	Epochs          int
	LearningRate    float64
	Optimizer       Optimizer
	LossFunction    LossFunction
	BatchSize       int
	ValidationSplit float64
}

type Model struct {
	NeuralNetwork  NeuralNetwork
	TrainingConfig TrainingConfig
}

type NeuralNetwork struct {
	InputLayer InputLayer

	Layers []Layer

	OutputLayer OutputLayer
}

type ModelWeightsAndBiases struct {
	Weights [][]float64
	Biases  [][]float64
}

type Layer struct {
	Neurons            int
	ActivationFunction ActivationFunction
}

func (nn *NeuralNetwork) AddLayer(layer Layer) {
	nn.Layers = append(nn.Layers, layer)
}

func (nn *NeuralNetwork) SetOutputLayer(layer OutputLayer) {
	nn.OutputLayer = layer
}

func (nn *NeuralNetwork) SetInputLayer(layer InputLayer) {
	nn.InputLayer = layer
}

func (nn *NeuralNetwork) Predict(input []float64, weightsAndBiases ModelWeightsAndBiases) ([]float64, error) {

	println("Predicting...")

	weights := weightsAndBiases.Weights
	biases := weightsAndBiases.Biases

	x := input

	println(len(weights))
	println(len(biases))

	for i := range nn.Layers {

		newX := make([]float64, nn.Layers[i].Neurons) // Initialized elements as 0

		/**
		Processing Layer 1

		*/
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
				dotProduct, err := Vectors.DotProduct(x, currWeights)

				// testting weights
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

	return x, nil
}

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
			weights[i][j] = 0.01*float64(i+j) // Example initialization
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

func println(args ...interface{}) {
	fmt.Println(args...)
}
