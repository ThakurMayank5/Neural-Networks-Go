package dataset

import (
	"fmt"
	"math/rand"
)

func SplitWithoutShuffle(dataset Dataset, trainRatio float64) (Dataset, Dataset, error) {

	if trainRatio <= 0 || trainRatio >= 1 {
		return Dataset{}, Dataset{}, fmt.Errorf("trainRatio must be between 0 and 1")
	}

	if len(dataset.Inputs) == 0 {
		return Dataset{}, Dataset{}, fmt.Errorf("dataset is empty")
	}

	totalSamples := len(dataset.Inputs)
	trainSize := int(float64(totalSamples) * trainRatio)

	trainDataset := Dataset{
		Inputs:      dataset.Inputs[:trainSize],
		Outputs:     dataset.Outputs[:trainSize],
		NumSamples:  trainSize,
		NumFeatures: dataset.NumFeatures,
		NumOutputs:  dataset.NumOutputs,
	}

	testDataset := Dataset{
		Inputs:      dataset.Inputs[trainSize:],
		Outputs:     dataset.Outputs[trainSize:],
		NumSamples:  totalSamples - trainSize,
		NumFeatures: dataset.NumFeatures,
		NumOutputs:  dataset.NumOutputs,
	}

	return trainDataset, testDataset, nil
}

func SplitWithShuffle(dataset Dataset, trainRatio float64) (Dataset, Dataset, error) {

	if trainRatio <= 0 || trainRatio >= 1 {
		return Dataset{}, Dataset{}, fmt.Errorf("trainRatio must be between 0 and 1")
	}

	if len(dataset.Inputs) == 0 {
		return Dataset{}, Dataset{}, fmt.Errorf("dataset is empty")
	}

	totalSamples := len(dataset.Inputs)
	trainSize := int(float64(totalSamples) * trainRatio)

	// Create a slice of indices and shuffle it
	indices := make([]int, totalSamples)
	for i := range indices {
		indices[i] = i
	}
	rand.Shuffle(len(indices), func(i, j int) {
		indices[i], indices[j] = indices[j], indices[i]
	})

	trainDataset := Dataset{
		Inputs:      make([][]float64, trainSize),
		Outputs:     make([][]float64, trainSize),
		NumSamples:  trainSize,
		NumFeatures: dataset.NumFeatures,
		NumOutputs:  dataset.NumOutputs,
	}

	testDataset := Dataset{
		Inputs:      make([][]float64, totalSamples-trainSize),
		Outputs:     make([][]float64, totalSamples-trainSize),
		NumSamples:  totalSamples - trainSize,
		NumFeatures: dataset.NumFeatures,
		NumOutputs:  dataset.NumOutputs,
	}

	for i, idx := range indices {
		if i < trainSize {
			trainDataset.Inputs[i] = dataset.Inputs[idx]
			trainDataset.Outputs[i] = dataset.Outputs[idx]
		} else {
			testDataset.Inputs[i-trainSize] = dataset.Inputs[idx]
			testDataset.Outputs[i-trainSize] = dataset.Outputs[idx]
		}
	}

	return trainDataset, testDataset, nil
}
