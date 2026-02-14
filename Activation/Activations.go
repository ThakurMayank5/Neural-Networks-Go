package Activation

import "math"

func GetActivationFunction(name string) func(float64) float64 {
	switch name {
	case "relu":
		return ReLU
	case "sigmoid":
		return Sigmoid
	case "tanh":
		return Tanh
	default:
		return nil
	}
}

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
