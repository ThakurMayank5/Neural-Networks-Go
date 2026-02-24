package main

import (
	"fmt"

	activ "github.com/ThakurMayank5/gonn/activation"
	"github.com/ThakurMayank5/gonn/dataloader"
	"github.com/ThakurMayank5/gonn/dataset"
	nn "github.com/ThakurMayank5/gonn/neuralnetwork"
)

func main() {

	// MNIST Handwritten Digit Classification
	//
	// CSV format: label, pixel0, pixel1, ..., pixel783
	//   Column 0   : digit label (0-9)
	//   Columns 1-784 : pixel values (0-255), normalized to [0,1]
	//
	// Architecture:
	//   Input  : 784  (28×28 pixels)
	//   Hidden : 128  → Sigmoid  (Xavier Normal)
	//   Hidden :  64  → Sigmoid  (Xavier Normal)
	//   Output :  10  → Softmax  (Kaiming Normal)

	model := nn.Model{
		NeuralNetwork: nn.NeuralNetwork{
			InputLayer: nn.InputLayer{
				Neurons: 784,
			},
			Layers: []nn.Layer{
				{
					Neurons:            128,
					ActivationFunction: activ.Sigmoid,
					Initialization:     nn.XavierNormalInitializer,
				},
				{
					Neurons:            64,
					ActivationFunction: activ.Sigmoid,
					Initialization:     nn.XavierNormalInitializer,
				},
			},
			OutputLayer: nn.OutputLayer{
				Neurons:            10,
				ActivationFunction: activ.Softmax,
				Initialization:     nn.KaimingNormalInitializer,
			},
		},
		TrainingConfig: nn.TrainingConfig{
			Epochs:       20,
			LearningRate: 0.01,
			Optimizer:    "sgd",
			LossFunction: "categorical_crossentropy",
			BatchSize:    64,
		},
	}

	model.NeuralNetwork.Summary()

	// Load pre-trained weights from file instead of training
	err := model.LoadWeights("mnist.weights")
	if err != nil {
		fmt.Println("Error loading weights:", err)
		return
	}

	fmt.Println("Weights loaded from mnist.weights")

	// Build column index slice for 784 pixel columns (columns 1-784)
	pixelCols := make([]int, 784)
	for i := range pixelCols {
		pixelCols[i] = i + 1
	}

	// Load MNIST CSV
	// Column 0 is the integer label (0-9); columns 1-784 are pixel values.
	d, err := dataloader.FromCSV(
		"data.csv",
		dataset.CSVConfig{
			HasHeader:      true,
			InputColumns:   pixelCols,
			HasLabelColumn: true,
			LabelColumn:    0,
			NumClasses:     10,
			Delimiter:      ',',
		},
	)
	if err != nil {
		fmt.Println("Error loading dataset:", err)
		return
	}

	fmt.Printf("MNIST dataset loaded: %d samples, %d features, %d output classes\n",
		d.NumSamples, d.NumFeatures, d.NumOutputs)

	// Normalize pixel values from [0, 255] to [0, 1]
	for i := range d.Inputs {
		for j := range d.Inputs[i] {
			d.Inputs[i][j] /= 255.0
		}
	}

	fmt.Println("Pixel values normalized to [0, 1]")

	// Split into 80% train / 20% test (shuffled) — using test set for evaluation
	_, test, err := dataset.SplitWithShuffle(d, 0.8)
	if err != nil {
		fmt.Println("Error splitting dataset:", err)
		return
	}

	fmt.Printf("Test set: %d samples\n", test.NumSamples)

	fmt.Println("\nEvaluating on test set...")

	// Sample predictions — first 3 of each digit class from the test set
	fmt.Println("\n--- Sample Predictions from Test Set (first 3 of each digit) ---")
	classCounts := make(map[int]int)
	samplesPerClass := 3

	for i := 0; i < len(test.Inputs) && len(classCounts) < 10; i++ {
		// Actual class
		actualIdx := 0
		for j := 1; j < len(test.Outputs[i]); j++ {
			if test.Outputs[i][j] > test.Outputs[i][actualIdx] {
				actualIdx = j
			}
		}

		if classCounts[actualIdx] >= samplesPerClass {
			continue
		}

		prediction, err := model.NeuralNetwork.Predict(test.Inputs[i])
		if err != nil {
			continue
		}

		// Predicted class
		predIdx := 0
		for j := 1; j < len(prediction); j++ {
			if prediction[j] > prediction[predIdx] {
				predIdx = j
			}
		}

		match := "✓"
		if predIdx != actualIdx {
			match = "✗"
		}

		fmt.Printf("Sample %d: Predicted Digit %d (%.4f) | Actual Digit %d %s\n",
			i+1, predIdx, prediction[predIdx], actualIdx, match)

		classCounts[actualIdx]++
	}

	model.SaveWeights("mnist.weights")
}
