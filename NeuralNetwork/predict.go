package neuralnetwork

import (
	"fmt"

	"github.com/ThakurMayank5/Neural-Networks-Go/activation"
	vectors "github.com/ThakurMayank5/Neural-Networks-Go/vectors"
)

// PredictWithCache performs forward propagation and returns cached activation data
func (nn *NeuralNetwork) PredictWithCache(input []float64) ([]float64, CacheData, error) {

	weights := nn.WeightsAndBiases.Weights
	biases := nn.WeightsAndBiases.Biases

	cache := CacheData{
		Activations:    make([][]float64, len(weights)+1),
		PreActivations: make([][]float64, len(weights)),
	}

	// Store input as first activation
	cache.Activations[0] = make([]float64, len(input))
	copy(cache.Activations[0], input)

	x := input

	for i := range len(weights) {

		activationFunction := activation.ReLU

		if i == len(weights)-1 {
			activationFunction = nn.OutputLayer.ActivationFunction
		} else {
			activationFunction = nn.Layers[i].ActivationFunction
		}

		newX := make([]float64, len(biases[i]))
		cache.PreActivations[i] = make([]float64, len(biases[i]))

		for j := 0; j < len(biases[i]); j++ {
			currWeights := weights[i][j*len(x) : (j+1)*len(x)]
			dotProduct, err := vectors.DotProduct(x, currWeights)
			if err != nil {
				fmt.Println("Error computing dot product:", err)
				return nil, cache, err
			}

			z := dotProduct + biases[i][j]
			cache.PreActivations[i][j] = z

			// For softmax, we'll apply it after computing all z values
			if activationFunction != activation.Softmax {
				activationFunctionToUse := activation.GetActivationFunction(activationFunction)
				newX[j] = activationFunctionToUse(z)
			} else {
				newX[j] = z // Store pre-activation for softmax
			}
		}

		// Apply softmax if needed
		if activationFunction == activation.Softmax {
			newX = activation.SoftmaxFunc(newX)
		}

		x = newX
		cache.Activations[i+1] = make([]float64, len(newX))
		copy(cache.Activations[i+1], newX)
	}
	return x, cache, nil
}

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
