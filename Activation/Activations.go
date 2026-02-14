package activation

import "math"

func ReLU(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func Tanh(x float64) float64 {
	return math.Tanh(x)
}
