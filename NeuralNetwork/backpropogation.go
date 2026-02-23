package neuralnetwork

import (
	"fmt"
	"math"

	"github.com/ThakurMayank5/Neural-Networks-Go/activation"
)

func (model *Model) BackpropagateBatch(batchInputs [][]float64, batchTargets [][]float64) error {

	loss, err := model.ForwardPassBatch(batchInputs, batchTargets)

	if err != nil {
		return err
	}

	fmt.Printf("Batch Loss: %.4f\n", loss)

	// deltas := make([][]float64, len(model.NeuralNetwork.OutputLayer.Neurons))

	// Output Layer Backpropagation





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
