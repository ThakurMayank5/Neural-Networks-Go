package losses

import "fmt"

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
