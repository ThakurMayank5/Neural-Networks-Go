package main

import "github.com/ThakurMayank5/Neural-Networks-Go/NeuralNetwork"

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

}
