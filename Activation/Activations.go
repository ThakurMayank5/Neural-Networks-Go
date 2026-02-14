package activation

import (
	"math"
)

// ActivationFunction represents the type of activation function
type ActivationFunction string

const (
	ReLU    ActivationFunction = "relu"
	Sigmoid ActivationFunction = "sigmoid"
	Tanh    ActivationFunction = "tanh"
)

func GetActivationFunction(name ActivationFunction) func(float64) float64 {
	switch name {
	case ReLU:
		return reluFunc
	case Sigmoid:
		return sigmoidFunc
	case Tanh:
		return tanhFunc
	default:
		return nil
	}
}

func reluFunc(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func sigmoidFunc(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func tanhFunc(x float64) float64 {
	return math.Tanh(x)
}
