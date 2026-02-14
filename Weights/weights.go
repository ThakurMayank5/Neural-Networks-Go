package Weights

import (
	"fmt"
)

func KaimingUniform(weights []float64, inputSize, outputSize int, nonLinearity string) ([]float64, error) {

	if nonLinearity != "relu" {
		return nil, fmt.Errorf("unsupported non-linearity: %s", nonLinearity)
	}

	return weights, nil
}
