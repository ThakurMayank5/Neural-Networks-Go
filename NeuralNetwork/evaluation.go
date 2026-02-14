package neuralnetwork

import (
	"fmt"

	"github.com/ThakurMayank5/Neural-Networks-Go/losses"
)

func (model *Model) Evaluate(dataset Dataset) error {

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

	accuracy := 0.0

	for i := range dataset.Inputs {
		input := dataset.Inputs[i]
		output, err := model.NeuralNetwork.Predict(input)
		if err != nil {
			fmt.Printf("Error predicting output for input %v: %v\n", input, err)
			return err
		}

		fmt.Printf("Input: %v, Predicted Output: %v\n", input, output)

		// Compute loss
		loss, err := losses.MeanSquaredError(dataset.Outputs[i], output)
		if err != nil {
			fmt.Printf("Error computing loss for input %v: %v\n", input, err)
			return err
		}

		accuracy += loss

		fmt.Printf("Loss: %v\n", loss)
	}

	accuracy = accuracy / float64(len(dataset.Inputs))

	fmt.Printf("Evaluation Accuracy: %v\n", accuracy)

	return nil
}
