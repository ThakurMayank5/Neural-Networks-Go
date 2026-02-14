package neuralnetwork

import (
	"fmt"

	vectors "github.com/ThakurMayank5/Neural-Networks-Go/Vectors"
)

func (nn *NeuralNetwork) Predict(input []float64) ([]float64, error) {

	println("Predicting...")

	weights := nn.WeightsAndBiases.Weights
	biases := nn.WeightsAndBiases.Biases

	x := input

	println(len(weights))
	println(len(biases))

	for i := range len(weights) {

		newX := make([]float64, len(biases[i])) // Initialized elements as 0

		for j := 0; j < len(biases[i]); j++ {
			currWeights := weights[i][j*len(x) : (j+1)*len(x)]
			dotProduct, err := vectors.DotProduct(x, currWeights)
			if err != nil {
				fmt.Println("Error computing dot product:", err)
				return nil, err
			}
			newX[j] = dotProduct + biases[i][j]
		}

		x = newX

	}
	return x, nil
}
