package main

import (
	"fmt"

	activ "github.com/ThakurMayank5/Neural-Networks-Go/activation"
	"github.com/ThakurMayank5/Neural-Networks-Go/neuralnetwork"
	nn "github.com/ThakurMayank5/Neural-Networks-Go/neuralnetwork"
)

func main() {

	model := nn.Model{
		NeuralNetwork: nn.NeuralNetwork{
			InputLayer: nn.InputLayer{
				Neurons:            784, // MNIST images are 28x28 = 784 pixels
				ActivationFunction: activ.ReLU,
			},
			OutputLayer: nn.OutputLayer{
				Neurons:            10, // 10 digit classes (0-9)
				ActivationFunction: activ.Softmax,
				Initialization:     nn.KaimingNormalInitializer,
			},
		},
		TrainingConfig: nn.TrainingConfig{
			Epochs:          20,   // More epochs for gradual learning
			LearningRate:    0.01, // Lower rate to prevent divergence
			Optimizer:       "sgd",
			LossFunction:    "categorical_crossentropy",
			BatchSize:       64,
			ValidationSplit: 0.2,
		},
	}

	// Neural network architecture for MNIST
	model.NeuralNetwork.AddLayer(nn.Layer{
		Neurons:            128,
		ActivationFunction: activ.Sigmoid,
		Initialization:     nn.XavierNormalInitializer,
	})

	model.NeuralNetwork.AddLayer(nn.Layer{
		Neurons:            64,
		ActivationFunction: activ.Sigmoid,
		Initialization:     nn.XavierNormalInitializer,
	})

	model.NeuralNetwork.Summary()

	err := model.InitializeWeights()

	if err != nil {
		fmt.Println("Error initializing weights:", err)
	}

	dataset, err := neuralnetwork.LoadMNISTCSV("data.csv")
	if err != nil {
		panic(err)
	}

	// Training on full MNIST dataset (60,000 samples)
	// Previously used subset for testing: subsetSize := 5000

	fmt.Printf("MNIST dataset loaded: %d samples with 784 features (28x28 pixels)!\n", len(dataset.Inputs))
	fmt.Println("Pixel values already normalized to [0, 1]")

	err = model.Fit(dataset,dataset)

	if err != nil {
		fmt.Println("Training completed with error:", err)
	} else {
		fmt.Println("\nTraining completed successfully!")
	}

	// Show predictions from different parts of dataset
	fmt.Println("\n--- Sample Predictions (First 3 of each digit) ---")
	classCounts := make(map[int]int)
	samplesPerClass := 3

	for i := 0; i < len(dataset.Inputs) && len(classCounts) < 10; i++ {
		// Find actual class
		actualIdx := 0
		for j := 1; j < len(dataset.Outputs[i]); j++ {
			if dataset.Outputs[i][j] > dataset.Outputs[i][actualIdx] {
				actualIdx = j
			}
		}

		if classCounts[actualIdx] < samplesPerClass {
			prediction, err := model.NeuralNetwork.Predict(dataset.Inputs[i])
			if err != nil {
				continue
			}

			// Find predicted class
			maxIdx := 0
			for j := 1; j < len(prediction); j++ {
				if prediction[j] > prediction[maxIdx] {
					maxIdx = j
				}
			}

			match := "✓"
			if maxIdx != actualIdx {
				match = "✗"
			}

			fmt.Printf("Sample %d: Predicted Digit %d (%.4f) | Actual Digit %d %s\n",
				i+1, maxIdx, prediction[maxIdx], actualIdx, match)
			classCounts[actualIdx]++
		}
	}
}
