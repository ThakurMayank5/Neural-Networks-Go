package neuralnetwork

import (
	"fmt"

	"github.com/ThakurMayank5/Neural-Networks-Go/activation"
	vectors "github.com/ThakurMayank5/Neural-Networks-Go/vectors"
)

func (nn *NeuralNetwork) Predict(input []float64) ([]float64, error) {

	weights := nn.WeightsAndBiases.Weights
	biases := nn.WeightsAndBiases.Biases

	x := input

	for i := range len(weights) {

		activationFunction := activation.ReLU

		if i == len(weights)-1 {
			activationFunction = nn.OutputLayer.ActivationFunction
		} else {
			activationFunction = nn.Layers[i].ActivationFunction
		}

		newX := make([]float64, len(biases[i])) // Initialized elements as 0

		for j := 0; j < len(biases[i]); j++ {
			currWeights := weights[i][j*len(x) : (j+1)*len(x)]
			dotProduct, err := vectors.DotProduct(x, currWeights)
			if err != nil {
				fmt.Println("Error computing dot product:", err)
				return nil, err
			}

			// For softmax, we'll apply it after computing all z values
			if activationFunction != activation.Softmax {
				activationFunctionToUse := activation.GetActivationFunction(activationFunction)
				newX[j] = activationFunctionToUse(dotProduct + biases[i][j])
			} else {
				newX[j] = dotProduct + biases[i][j] // Store pre-activation for softmax
			}
		}

		// Apply softmax if needed
		if activationFunction == activation.Softmax {
			newX = activation.SoftmaxFunc(newX)
		}

		x = newX

	}
	return x, nil
}
