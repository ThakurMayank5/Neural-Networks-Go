package gonn

import (
	"fmt"
	"testing"

	imageprocessing "github.com/ThakurMayank5/gonn/image-processing"
	"github.com/ThakurMayank5/gonn/tensor"
)

func TestSequentialMLPModel(t *testing.T) {

	// Create a simple sequential model
	model := Sequential(
		Dense(4, WithInitializer(XavierNormalInitializer), WithDropout(0.5)),
		ReLU(),
		Dense(2, WithInitializer(XavierNormalInitializer)),
		SoftMax(),
	)

	fmt.Println("Created a Sequential Model")

	model.Build([]int{1})

	// model.Summary()

	output := model.Predict(tensor.Tensor{
		Data:  []float64{1.0},
		Shape: []int{1},
	})

	fmt.Printf("Model output: %v\n", output.Data)

}

func TestSequentialCNNModel(t *testing.T) {

	fmt.Println("Starting Convolution Neural Network Test")

	model := Sequential(
		Input([]int{3, 28, 28}),
		Conv2D(2, 3, WithStride(1), WithPadding(1), WithInitializer(XavierUniformInitializer)),
		ReLU(),
		MaxPool2D(2, WithStride(2)),
		Conv2D(4, 3, WithStride(1), WithPadding(1), WithInitializer(XavierUniformInitializer)),
		ReLU(),
		Conv2D(2, 3, WithStride(1), WithPadding(1), WithInitializer(XavierUniformInitializer)),
		ReLU(),
		Flatten(),
		Dense(10, WithInitializer(XavierUniformInitializer)),
		SoftMax(),
	)

	model.Build([]int{3, 28, 28})

	model.Summary()

	imageData, err := imageprocessing.LoadImage("test-images\\images.jpg")
	if err != nil {
		t.Fatalf("Error loading image: %v", err)
	}

	// fmt.Printf("Loaded image data with shape: [%d, %d, %d]\n", len(imageData), len(imageData[0]), len(imageData[0][0]))

	// fmt.Printf("Image data: %v\n", imageData)

	pred := model.Predict(
		*tensor.NewTensorFrom3D(imageData),
	)

	fmt.Printf("Model output: %v\n", pred.Data)

}
