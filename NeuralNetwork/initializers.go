package neuralnetwork

import (
	"math"
	"math/rand"
	"sync"
)

// Kaiming Normal Initialization (He Initialization)
// w = randn() × sqrt(2/fan_in)
// fan_in is the number of input in the neuron from previous layers
func KaimingNormal(nn *NeuralNetwork) error {

	wg := sync.WaitGroup{}

	println(nn.Layers)
	println(len(nn.Layers))

	// Weight[layer][neuron][wgts]

	// Allocate memory for layers + output layer
	nn.WeightsAndBiases.Weights = make([][][]float64, len(nn.Layers)+1) // +1 for output layer

	// Allocate memory for each layer's neurons
	for i := 0; i < len(nn.Layers)+1; i++ {

		if i == len(nn.Layers) {
			nn.WeightsAndBiases.Weights[i] = make([][]float64, nn.OutputLayer.Neurons)

			continue
		}

		nn.WeightsAndBiases.Weights[i] = make([][]float64, nn.Layers[i].Neurons)

	}

	// Output + Hidden Layers
	for i := len(nn.Layers); i >= 0; i-- {
		wg.Add(1)

		// Output Layer
		if i == len(nn.Layers) {

			go func(n int) {
				defer wg.Done()

				// For each neuron in the output layer
				for j := nn.OutputLayer.Neurons - 1; j >= 0; j-- {

					totalWeights := 0

					if len(nn.Layers) == 0 {
						totalWeights = nn.InputLayer.Neurons
					} else {
						totalWeights = nn.Layers[n-1].Neurons
					}

					standard_deviation := math.Sqrt(2.0 / float64(totalWeights))

					currWeights := make([]float64, totalWeights)

					for k := range currWeights {
						currWeights[k] = rand.NormFloat64() * standard_deviation
					}

					nn.WeightsAndBiases.Weights[n][j] = currWeights

				}
			}(i)

			continue
		}

		// First Hidden Layer
		if i == 0 {

			go func(n int) {
				defer wg.Done()

				// For each neuron in the first hidden layer
				for j := nn.Layers[n].Neurons - 1; j >= 0; j-- {

					totalWeights := nn.InputLayer.Neurons

					standard_deviation := math.Sqrt(2.0 / float64(totalWeights))

					currWeights := make([]float64, totalWeights)

					for k := range currWeights {
						currWeights[k] = rand.NormFloat64() * standard_deviation
					}

					nn.WeightsAndBiases.Weights[n][j] = currWeights

				}
			}(i)

			continue

		}

		// Hidden Layers
		go func(layerIndex int) {
			defer wg.Done()

			// For each neuron in the hidden layer
			for j := nn.Layers[layerIndex].Neurons - 1; j >= 0; j-- {

				totalWeights := nn.Layers[layerIndex-1].Neurons

				standard_deviation := math.Sqrt(2.0 / float64(totalWeights))
				currWeights := make([]float64, totalWeights)

				for k := range currWeights {
					currWeights[k] = rand.NormFloat64() * standard_deviation
				}

				nn.WeightsAndBiases.Weights[layerIndex][j] = currWeights

			}

		}(i)

	}

	wg.Wait()

	return nil
}
