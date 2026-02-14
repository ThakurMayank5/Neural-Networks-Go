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
				Neurons:            13, // Wine dataset has 13 features
				ActivationFunction: activ.ReLU,
			},
			OutputLayer: nn.OutputLayer{
				Neurons:            3, // 3 wine classes
				ActivationFunction: activ.Softmax,
				Initialization:     nn.KaimingNormalInitializer,
			},
		},
		TrainingConfig: nn.TrainingConfig{
			Epochs:          100,
			LearningRate:    0.01,
			Optimizer:       "sgd",
			LossFunction:    "categorical_crossentropy",
			BatchSize:       32,
			ValidationSplit: 0.2,
		},
	}

	// Optimal architecture for Wine dataset (13 features)
	model.NeuralNetwork.AddLayer(nn.Layer{
		Neurons:            128, // Larger first layer for more features
		ActivationFunction: activ.ReLU,
		Initialization:     nn.KaimingNormalInitializer,
	})

	model.NeuralNetwork.AddLayer(nn.Layer{
		Neurons:            64,
		ActivationFunction: activ.ReLU,
		Initialization:     nn.KaimingNormalInitializer,
	})

	model.NeuralNetwork.AddLayer(nn.Layer{
		Neurons:            32,
		ActivationFunction: activ.ReLU,
		Initialization:     nn.KaimingNormalInitializer,
	})

	model.NeuralNetwork.Summary()

	model.TrainingConfig.Epochs = 200

	err := model.InitializeWeights()

	if err != nil {
		fmt.Println("Error initializing weights:", err)
	}

	dataset, err := neuralnetwork.LoadCSVWithOneHot("data.csv", 13, 3)
	if err != nil {
		panic(err)
	}

	// Normalize the dataset for better training
	dataset.NormalizeInputs()
	fmt.Printf("Wine dataset loaded: %d samples with 13 features, normalized!\n", len(dataset.Inputs))

	err = model.SGDFitWithEpochs(dataset)

	if err != nil {
		fmt.Println("Training completed with error:", err)
	} else {
		fmt.Println("\nTraining completed successfully!")
	}

	// Show predictions from different parts of dataset
	fmt.Println("\n--- Sample Predictions (First 5 of each class) ---")
	classCounts := make(map[int]int)
	samplesPerClass := 5

	for i := 0; i < len(dataset.Inputs) && len(classCounts) < 3; i++ {
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

			fmt.Printf("Sample %d: Predicted Class %d (%.4f) | Actual Class %d %s\n",
				i+1, maxIdx, prediction[maxIdx], actualIdx, match)
			classCounts[actualIdx]++
		}
	}
}
