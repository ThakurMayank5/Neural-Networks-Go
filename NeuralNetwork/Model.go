package NeuralNetwork

import "fmt"

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
	ModelWeights   ModelWeights
}

type ModelWeights struct {
	Weights [][]float64
	Biases  []float64
}

type NeuralNetwork struct {
	InputLayer InputLayer

	Layers []Layer

	OutputLayer OutputLayer
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

func (nn *NeuralNetwork) Predict(input []float64) []float64 {
	// Prediction logic goes here
	return []float64{}
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

}

func (model *Model) Train(dataset Dataset) {

	totalSamples := len(dataset.Inputs)
	batchSize := model.TrainingConfig.BatchSize
	epochs := model.TrainingConfig.Epochs

	for epoch := 1; epoch <= epochs; epoch++ {
		fmt.Printf("Epoch %d/%d\n", epoch, epochs)
	}

}
