package neuralnetwork

import (
	"fmt"

	"github.com/ThakurMayank5/Neural-Networks-Go/losses"
)

func (model *Model) SGDFitWithEpochs(dataset Dataset) error {

	for epoch := 1; epoch <= model.TrainingConfig.Epochs; epoch++ {
		fmt.Printf("Epoch %d/%d\n", epoch, model.TrainingConfig.Epochs)

		err := model.SGDFit(dataset)

		if err != nil {
			fmt.Printf("Error during training: %v\n", err)
			return err
		}

		_, evalErr := model.Evaluate(dataset)

		if evalErr != nil {
			fmt.Printf("Error during evaluation: %v\n", evalErr)
			return evalErr
		}
	}

	return nil
}

// Fit trains the model on the provided dataset

// Stochastic Gradient Descent implementation
// Training loop using SGD for training the neural network
func (model *Model) SGDFit(dataset Dataset) error {

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

	_ = totalLayers - 1 // totalTrainableLayers (Exclude input layer)

	// fmt.Printf("Total Layers: %d\n", totalLayers)
	// fmt.Printf("Total Trainable Layers: %d\n", totalTrainableLayers)

	// Steps:
	// 1. Forward Propagation
	// 2. Compute Loss
	// 3. Backward Propagation
	// 4. Compute Gradients
	// 5. Update Weights and Biases

	for i := range dataset.Inputs {
		input := dataset.Inputs[i]
		target := dataset.Outputs[i]

		// Forward pass with activation caching
		output, cache, err := model.NeuralNetwork.PredictWithCache(input)
		if err != nil {
			fmt.Printf("Error predicting output for input %v: %v\n", input, err)
			return err
		}

		// fmt.Printf("Input: %v, Predicted Output: %v\n", input, output)

		// Compute loss (use Cross-Entropy for Softmax, MSE for others)
		if model.NeuralNetwork.OutputLayer.ActivationFunction == "softmax" {
			_, err = losses.CategoricalCrossEntropy(output, target)
		} else {
			_, err = losses.MeanSquaredError(target, output)
		}
		if err != nil {
			fmt.Printf("Error computing loss for input %v: %v\n", input, err)
			return err
		}

		// fmt.Printf("Loss: %v\n", loss)

		// Backward Propagation with weight/bias updates
		err = model.Backpropagate(cache, target, output)
		if err != nil {
			fmt.Printf("Error during backpropagation: %v\n", err)
			return err
		}

	}

	return nil

}
