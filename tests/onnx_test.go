package tests

import (
	"fmt"
	"testing"

	"github.com/ThakurMayank5/gonn/onnx"
)

func TestONNXModelLoading(t *testing.T) {

	onnxModelPath := "../onnx-models/mlp.onnx"

	model, err := onnx.LoadONNXMLPModel(onnxModelPath)

	t.Logf("Loaded ONNX model from %s", onnxModelPath)

	if err != nil {
		t.Fatalf("Failed to load ONNX model: %v", err)

	}

	model.NeuralNetwork.Summary()

	model.SetInferenceMode(true)

	output, err := model.Predict([]float64{0.5, 0.2, 0.1, 0.4})
	if err != nil {
		t.Fatalf("Failed to predict: %v", err)
	}
	fmt.Printf("Prediction output: %v\n", output)
}

/*
Input: 1x28x28

Conv(20,5x5)
ReLU
MaxPool(2x2)

Conv(50,5x5)
ReLU

# Flatten

Dense(500)
ReLU

Dense(10)
Softmax
*/

// go test ./tests  -v -run TestONNXMNISTModelLoading

func TestONNXMNISTModelLoading(t *testing.T) {

	onnxModelPath := "../mnist-8.onnx"

	model, err := onnx.LoadONNXModel(onnxModelPath)

	t.Logf("Loaded ONNX model from %s", onnxModelPath)

	if err != nil {
		t.Fatalf("Failed to load ONNX model: %v", err)

	}

	model.Summary()

}
