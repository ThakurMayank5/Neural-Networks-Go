package onnx

import (
	"fmt"
	"os"

	"github.com/ThakurMayank5/gonn"

	"google.golang.org/protobuf/proto"
)

func LoadONNXModel(filePath string) (*gonn.Model, error) {

	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read ONNX file: %v", err)
	}

	modelONNX := &ModelProto{}

	err = proto.Unmarshal(data, modelONNX)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal ONNX model: %v", err)
	}

	tensorMap := map[string]*TensorProto{}

	for _, t := range modelONNX.Graph.Initializer {
		tensorMap[*t.Name] = t
	}

	model := &gonn.Model{}

	for _, node := range modelONNX.Graph.Node {

		// fmt.Printf("Node: %s, OpType: %s\n", *node.Name, *node.OpType)

		switch *node.OpType {

		case "Conv":
			fmt.Printf(" Conv Node: %s\n", *node.Name)

			weightName := node.Input[1]
			weightTensor := tensorMap[weightName]

			filters := int(weightTensor.Dims[0])
			// channels := int(weightTensor.Dims[1])
			kernel := int(weightTensor.Dims[2])

			layer := gonn.Conv2D(filters, kernel)

			model.Layers = append(model.Layers, layer)

		case "Relu":
			fmt.Printf(" ReLU Node: %s\n", *node.Name)

		case "MaxPool":
			fmt.Printf(" MaxPool Node: %s\n", *node.Name)

		case "Reshape":
			fmt.Printf(" Reshape Node: %s\n", *node.Name)

		case "Gemm":
			fmt.Printf(" Gemm Node: %s\n", *node.Name)
		}

	}

	return model, nil
}
