package main

import (
	"fmt"

	"github.com/ThakurMayank5/Neural-Networks-Go/NeuralNetwork"
)

func main() {

	model := NeuralNetwork.Model{
		NeuralNetwork: NeuralNetwork.NeuralNetwork{
			InputLayer: NeuralNetwork.InputLayer{
				Neurons:            3,
				ActivationFunction: "relu",
			},
			OutputLayer: NeuralNetwork.OutputLayer{
				Neurons:            1,
				ActivationFunction: "sigmoid",
			},
		},
		TrainingConfig: NeuralNetwork.TrainingConfig{
			Epochs:          100,
			LearningRate:    0.01,
			Optimizer:       "adam",
			LossFunction:    "binary_crossentropy",
			BatchSize:       32,
			ValidationSplit: 0.2,
		},
	}

	model.NeuralNetwork.AddLayer(NeuralNetwork.Layer{
		Neurons:            5,
		ActivationFunction: NeuralNetwork.ReLU,
	})

	model.NeuralNetwork.AddLayer(NeuralNetwork.Layer{
		Neurons:            4,
		ActivationFunction: NeuralNetwork.ReLU,
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

	err := model.Fit(NeuralNetwork.Dataset{
		Inputs:  [][]float64{{1, 0, 0}, {0, 0, 1}, {0, 1, 0}, {1, 0, 0}},
		Outputs: [][]float64{{0}, {1}, {1}, {1}},
	})
	if err != nil {
		fmt.Println("Training completed with error:", err)
	} else {
		fmt.Println("Training completed successfully.")
	}
}
