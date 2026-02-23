package neuralnetwork

import (
	"fmt"
	"math/rand"
	"time"
)

func (model *Model) Fit(training Dataset, validation Dataset) error {

	// initialize a random seed for further use
	rand.Seed(time.Now().UnixNano())

	// Dataset validation

	if len(training.Inputs) == 0 || len(training.Outputs) == 0 {
		return fmt.Errorf("training dataset is empty")
	}

	if len(training.Inputs) != len(training.Outputs) {
		return fmt.Errorf("number of inputs and outputs must be the same")
	}

	if len(training.Inputs[0]) != model.NeuralNetwork.InputLayer.Neurons {
		return fmt.Errorf("input data does not match the number of neurons in the input layer")
	}

	total_samples := len(training.Inputs)

	// totalLayers := len(model.NeuralNetwork.Layers) + 2 // Input and Output layers

	epochs := model.TrainingConfig.Epochs
	batchSize := model.TrainingConfig.BatchSize

	batchesPerEpoch := (total_samples + batchSize - 1) / batchSize // Ceiling division

	fmt.Printf("Starting training for %d epochs with batch size %d (%d batches per epoch)\n", epochs, batchSize, batchesPerEpoch)

	for epoch := 1; epoch <= epochs; epoch++ {

		fmt.Printf("Epoch %d/%d\n", epoch, epochs)

		// Shuffle the training data at the beginning of each epoch
		shuffledIndices := rand.Perm(total_samples)

		for batch := 0; batch < batchesPerEpoch; batch++ {

			start := batch * batchSize
			end := start + batchSize
			if end > total_samples {
				end = total_samples
			}

			// Create mini-batch
			batchInputs := make([][]float64, end-start)
			batchTargets := make([][]float64, end-start)

			for i, idx := range shuffledIndices[start:end] {
				batchInputs[i] = training.Inputs[idx]
				batchTargets[i] = training.Outputs[idx]
			}

			// Backward Propagation with weight/bias updates for the entire batch
			err := model.BackpropagateBatch(batchInputs, batchTargets)
			if err != nil {
				return err
			}

		}

	}

	// _ = totalLayers - 1 // totalTrainableLayers (Exclude input layer)

	// // fmt.Printf("Total Layers: %d\n", totalLayers)
	// // fmt.Printf("Total Trainable Layers: %d\n", totalTrainableLayers)

	// // Steps:
	// // 1. Forward Propagation
	// // 2. Compute Loss
	// // 3. Backward Propagation
	// // 4. Compute Gradients
	// // 5. Update Weights and Biases

	// for i := range training.Inputs {
	// 	input := training.Inputs[i]
	// 	target := training.Outputs[i]

	// 	// Forward pass with activation caching
	// 	output, cache, err := model.NeuralNetwork.PredictWithCache(input)
	// 	if err != nil {
	// 		fmt.Printf("Error predicting output for input %v: %v\n", input, err)
	// 		return err
	// 	}

	// 	// fmt.Printf("Input: %v, Predicted Output: %v\n", input, output)

	// 	// Compute loss (use Cross-Entropy for Softmax, MSE for others)
	// 	if model.NeuralNetwork.OutputLayer.ActivationFunction == "softmax" {
	// 		_, err = losses.CategoricalCrossEntropy(output, target)
	// 	} else {
	// 		_, err = losses.MeanSquaredError(target, output)
	// 	}
	// 	if err != nil {
	// 		fmt.Printf("Error computing loss for input %v: %v\n", input, err)
	// 		return err
	// 	}

	// 	// fmt.Printf("Loss: %v\n", loss)

	// 	// Backward Propagation with weight/bias updates
	// 	err = model.Backpropagate(cache, target, output)
	// 	if err != nil {
	// 		fmt.Printf("Error during backpropagation: %v\n", err)
	// 		return err
	// 	}

	// }

	return nil

}
