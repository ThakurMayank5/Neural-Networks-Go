package neuralnetwork

import (
	"math"

	"github.com/ThakurMayank5/Neural-Networks-Go/activation"
)

// CacheData stores forward pass information needed for backpropagation
type CacheData struct {
	Activations    [][]float64 // Activations from each layer (including input)
	PreActivations [][]float64 // Pre-activation values z = wx + b
}

func (model *Model) Backpropagate(cache CacheData, target []float64, prediction []float64) error {

	weights := model.NeuralNetwork.WeightsAndBiases.Weights
	biases := model.NeuralNetwork.WeightsAndBiases.Biases
	learningRate := model.TrainingConfig.LearningRate

	numLayers := len(weights)
	deltas := make([][]float64, numLayers)

	// ---- OUTPUT LAYER DELTA ----
	// delta_L = (prediction - target) * activation'(z_L)
	// Special case: Softmax + Cross-Entropy gradient is simply (prediction - target)
	lastIdx := numLayers - 1
	deltas[lastIdx] = make([]float64, len(prediction))

	outputActivation := model.NeuralNetwork.OutputLayer.ActivationFunction

	// For Softmax + Cross-Entropy, gradient simplifies to (prediction - target)
	if outputActivation == activation.Softmax {
		for j := range prediction {
			deltas[lastIdx][j] = prediction[j] - target[j]
		}
	} else {
		// For other activations, use standard gradient
		outputDerivFunc := getActivationDerivative(outputActivation)
		for j := range prediction {
			// Loss gradient: dL/da = 2(a - target) for MSE, simplified to (a - target)
			lossDelta := prediction[j] - target[j]

			// Multiply by activation derivative
			if outputDerivFunc != nil && lastIdx < len(cache.PreActivations) {
				lossDelta *= outputDerivFunc(cache.PreActivations[lastIdx][j])
			}
			deltas[lastIdx][j] = lossDelta
		}
	}

	// ---- HIDDEN LAYER DELTAS (Backpropagate) ----
	for layerIdx := lastIdx - 1; layerIdx >= 0; layerIdx-- {
		nextLayerIdx := layerIdx + 1

		// deltas[layerIdx] represents error for OUTPUTS of layer layerIdx
		// Size should be len(biases[layerIdx]) or len(cache.Activations[layerIdx+1])
		neuronsInCurrentLayer := len(biases[layerIdx])
		deltas[layerIdx] = make([]float64, neuronsInCurrentLayer)

		hiddenActivation := model.NeuralNetwork.Layers[layerIdx].ActivationFunction
		hiddenDerivFunc := getActivationDerivative(hiddenActivation)

		// For each neuron in current layer (output neurons)
		for j := range deltas[layerIdx] {
			// Backprop: sum over next layer neurons
			var weightedDeltaSum float64
			inputsToCurrentLayer := len(cache.Activations[layerIdx])

			for k := 0; k < len(deltas[nextLayerIdx]); k++ {
				// Weight from neuron j (in layer layerIdx) to neuron k (in next layer)
				// In weights[nextLayerIdx]: for each output neuron k, weights are stored for all inputs
				weightsPerOutputNeuron := inputsToCurrentLayer
				weightIndex := k*weightsPerOutputNeuron + j

				if weightIndex < len(weights[nextLayerIdx]) {
					w := weights[nextLayerIdx][weightIndex]
					weightedDeltaSum += w * deltas[nextLayerIdx][k]
				}
			}

			// Multiply by activation derivative of current layer's pre-activation
			if hiddenDerivFunc != nil && j < len(cache.PreActivations[layerIdx]) {
				weightedDeltaSum *= hiddenDerivFunc(cache.PreActivations[layerIdx][j])
			}
			deltas[layerIdx][j] = weightedDeltaSum
		}
	}

	// ---- UPDATE WEIGHTS AND BIASES ----
	for layerIdx := 0; layerIdx < numLayers; layerIdx++ {
		prevLayerActivations := cache.Activations[layerIdx]
		currentDeltas := deltas[layerIdx]

		// Update biases for this layer
		var biasUpdateSum float64
		for j := range biases[layerIdx] {
			// Bias gradient = delta_j
			gradient := currentDeltas[j]
			biases[layerIdx][j] -= learningRate * gradient
			biasUpdateSum += math.Abs(gradient)
		}

		// Update weights for this layer
		numNeurons := len(currentDeltas)
		var weightUpdateSum float64
		var weightUpdateCount int
		for j := 0; j < numNeurons; j++ {
			weightsPerNeuron := len(prevLayerActivations)
			for i := 0; i < weightsPerNeuron; i++ {
				weightIndex := j*weightsPerNeuron + i
				// Weight gradient = delta_j * a_i (activation from previous layer)
				gradient := currentDeltas[j] * prevLayerActivations[i]
				weights[layerIdx][weightIndex] -= learningRate * gradient
				weightUpdateSum += math.Abs(gradient)
				weightUpdateCount++
			}
		}

		// Debug: Print gradient statistics for first layer occasionally
		// (This will help us see if gradients are non-zero)
		_ = weightUpdateSum // Avoid unused variable warning
		_ = biasUpdateSum
	}

	return nil
}

// getActivationDerivative returns the derivative function for an activation
func getActivationDerivative(activationFunc activation.ActivationFunction) func(float64) float64 {
	switch activationFunc {
	case activation.ReLU:
		return func(z float64) float64 {
			if z > 0 {
				return 1.0
			}
			return 0.0
		}
	case activation.Sigmoid:
		return func(z float64) float64 {
			// sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
			s := 1.0 / (1.0 + math.Exp(-z))
			return s * (1.0 - s)
		}
	case activation.Tanh:
		return func(z float64) float64 {
			// tanh'(z) = 1 - tanh(z)^2
			t := math.Tanh(z)
			return 1.0 - (t * t)
		}
	default:
		return nil
	}
}
