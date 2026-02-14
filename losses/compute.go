package losses

import (
	"fmt"
	"math"
)

func MeanSquaredError(predictions, targets []float64) (float64, error) {
	if len(predictions) != len(targets) {
		return 0, fmt.Errorf("predictions and targets must be of the same length")
	}
	mse := 0.0
	for i := range predictions {
		diff := predictions[i] - targets[i]
		mse += diff * diff
	}
	return mse / float64(len(predictions)), nil
}

// CategoricalCrossEntropy computes cross-entropy loss for multi-class classification
func CategoricalCrossEntropy(predictions, targets []float64) (float64, error) {
	if len(predictions) != len(targets) {
		return 0, fmt.Errorf("predictions and targets must be of the same length")
	}

	loss := 0.0
	epsilon := 1e-15 // For numerical stability

	for i := range predictions {
		// Clip predictions to avoid log(0)
		p := math.Max(epsilon, math.Min(1.0-epsilon, predictions[i]))
		if targets[i] > 0 { // Only compute for actual class (one-hot encoded)
			loss += -targets[i] * math.Log(p)
		}
	}

	return loss, nil
}
