package neuralnetwork

import (
	"fmt"
	"math"
	"math/rand"
)

// Kaiming Normal Initialization (He Initialization)
// w = randn() × sqrt(2/fan_in)
// fan_in is the number of input in the neuron from previous layers
func KaimingNormal(nn *NeuralNetwork) error {

	println(nn.Layers)
	println(len(nn.Layers))

	for i := range len(nn.Layers) + 1 { // +1 for output layer

		println("Value of i:", i)

		if i == 0 {
			fmt.Printf("Initializing weights for Layer %d (Input Layer) with Kaiming Normal\n", i+1)

			standard_deviation := math.Sqrt(2.0 / float64(nn.InputLayer.Neurons))

			fmt.Printf("Standard Deviation for Layer %d: %f\n", i+1, standard_deviation)

			currWeights := make([]float64, nn.InputLayer.Neurons*nn.Layers[i].Neurons)

			for j := range currWeights {
				currWeights[j] = rand.NormFloat64() * standard_deviation
			}

			nn.WeightsAndBiases.Weights = append(nn.WeightsAndBiases.Weights, currWeights)

			continue
		}

		if i == len(nn.Layers) {
			fmt.Printf("Initializing weights for Layer %d (Output Layer) with Kaiming Normal\n", i+1)

			standard_deviation := math.Sqrt(2.0 / float64(nn.Layers[i-1].Neurons))

			fmt.Printf("Standard Deviation for Layer %d: %f\n", i+1, standard_deviation)

			currWeights := make([]float64, nn.Layers[i-1].Neurons*nn.OutputLayer.Neurons)

			for j := range currWeights {
				currWeights[j] = rand.NormFloat64() * standard_deviation
			}

			nn.WeightsAndBiases.Weights = append(nn.WeightsAndBiases.Weights, currWeights)

			continue
		}

		fmt.Printf("Initializing weights for Layer %d with Kaiming Normal\n", i+1)

		standard_deviation := math.Sqrt(2.0 / float64(nn.Layers[i-1].Neurons))

		fmt.Printf("Standard Deviation for Layer %d: %f\n", i+1, standard_deviation)

		currWeights := make([]float64, nn.Layers[i-1].Neurons*nn.Layers[i].Neurons)

		for k := range currWeights {
			currWeights[k] = rand.NormFloat64() * standard_deviation
		}

		nn.WeightsAndBiases.Weights = append(nn.WeightsAndBiases.Weights, currWeights)

	}

	return nil
}
