package main

import (
	"fmt"

	activ "github.com/ThakurMayank5/Neural-Networks-Go/activation"
	nn "github.com/ThakurMayank5/Neural-Networks-Go/neuralnetwork"
)

func main() {

	model := nn.Model{
		NeuralNetwork: nn.NeuralNetwork{
			InputLayer: nn.InputLayer{
				Neurons:            3,
				ActivationFunction: activ.ReLU,
			},
			OutputLayer: nn.OutputLayer{
				Neurons:            1,
				ActivationFunction: activ.Sigmoid,
				Initialization:     nn.KaimingNormalInitializer,
			},
		},
		TrainingConfig: nn.TrainingConfig{
			Epochs:          100,
			LearningRate:    0.01,
			Optimizer:       "adam",
			LossFunction:    "binary_crossentropy",
			BatchSize:       32,
			ValidationSplit: 0.2,
		},
	}

	model.NeuralNetwork.AddLayer(nn.Layer{
		Neurons:            5,
		ActivationFunction: activ.ReLU,
		Initialization:     nn.KaimingNormalInitializer,
	})

	model.NeuralNetwork.AddLayer(nn.Layer{
		Neurons:            4,
		ActivationFunction: activ.ReLU,
		Initialization:     nn.KaimingNormalInitializer,
	})

	model.NeuralNetwork.Summary()

	model.TrainingConfig.Epochs = 200

	/*

		Neural Network Summary:
		Total Layers: 4
		Input Layer Neurons: 3
		Input Layer Activation Function: relu
		Layer 1 Neurons: 5
		Layer 1 Activation Function: relu
		Layer 2 Neurons: 4
		Layer 2 Activation Function: relu
		Output Layer Neurons: 1
		Output Layer Activation Function: sigmoid

	*/

	err := model.InitializeWeights()

	if err != nil {
		fmt.Println("Error initializing weights:", err)
	}

	err = model.SGDFit(nn.Dataset{
		Inputs: [][]float64{
			{0.1, 0.2, 0.3},
			{0.9, 0.1, 0.2},
			{0.8, 0.7, 0.6},
			{0.2, 0.3, 0.1},
			{0.5, 0.4, 0.9},
			{0.7, 0.8, 0.2},
			{0.3, 0.9, 0.7},
			{0.6, 0.1, 0.8},
			{0.4, 0.5, 0.6},
			{0.9, 0.9, 0.9},
			{0.2, 0.8, 0.4},
			{0.6, 0.6, 0.3},
			{0.05, 0.1, 0.2},
			{0.3, 0.2, 0.9},
			{0.8, 0.4, 0.1},
			{0.9, 0.3, 0.7},
			{0.45, 0.55, 0.65},
			{0.15, 0.25, 0.35},
			{0.75, 0.85, 0.95},
			{0.25, 0.35, 0.45},
		},
		Outputs: [][]float64{
			{0},
			{0},
			{1},
			{0},
			{1},
			{1},
			{1},
			{1},
			{1},
			{1},
			{0},
			{1},
			{0},
			{1},
			{0},
			{1},
			{1},
			{0},
			{1},
			{0},
		},
	})

	model.Evaluate(nn.Dataset{
		Inputs: [][]float64{
			{0.12, 0.22, 0.32},
			{0.88, 0.18, 0.28},
			{0.52, 0.42, 0.92},
			{0.22, 0.82, 0.42},
			{0.95, 0.95, 0.95},
			{0.35, 0.15, 0.85},
			{0.65, 0.75, 0.25},
			{0.18, 0.28, 0.38},
		},
		Outputs: [][]float64{
			{0},
			{0},
			{1},
			{0},
			{1},
			{1},
			{1},
			{0},
		},
	})

	if err != nil {
		fmt.Println("Training completed with error:", err)
	} else {
		fmt.Println("Training completed successfully.")
	}
}
