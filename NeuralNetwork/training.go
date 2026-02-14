package neuralnetwork

import "fmt"

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

	totalTrainableLayers := totalLayers - 1 // Exclude input layer

	fmt.Printf("Total Layers: %d\n", totalLayers)
	fmt.Printf("Total Trainable Layers: %d\n", totalTrainableLayers)

	// Steps:
	// 1. Forward Propagation
	// 2. Compute Loss
	// 3. Backward Propagation
	// 4. Compute Gradients
	// 5. Update Weights and Biases

	for i := range dataset.Inputs {
		input := dataset.Inputs[i]
		output, err := model.NeuralNetwork.Predict(input)
		if err != nil {
			fmt.Printf("Error predicting output for input %v: %v\n", input, err)

			return err

		}
		fmt.Printf("Input: %v, Predicted Output: %v\n", input, output)
	}

	return nil

}
