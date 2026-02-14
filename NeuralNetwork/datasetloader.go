package neuralnetwork

import (
	"encoding/csv"
	"os"
	"strconv"
)

func LoadCSV(path string, inputSize int) (Dataset, error) {

	file, err := os.Open(path)
	if err != nil {
		return Dataset{}, err
	}
	defer file.Close()

	reader := csv.NewReader(file)

	records, err := reader.ReadAll()
	if err != nil {
		return Dataset{}, err
	}

	var inputs [][]float64
	var outputs [][]float64

	// Skip header → start from index 1
	for i := 1; i < len(records); i++ {

		row := records[i]

		var input []float64
		var output []float64

		// Parse inputs
		for j := 0; j < inputSize; j++ {
			val, err := strconv.ParseFloat(row[j], 64)
			if err != nil {
				return Dataset{}, err
			}
			input = append(input, val)
		}

		// Parse output (remaining columns)
		for j := inputSize; j < len(row); j++ {
			val, err := strconv.ParseFloat(row[j], 64)
			if err != nil {
				return Dataset{}, err
			}
			output = append(output, val)
		}

		inputs = append(inputs, input)
		outputs = append(outputs, output)
	}

	return Dataset{
		Inputs:  inputs,
		Outputs: outputs,
	}, nil
}

// LoadCSVWithOneHot loads CSV and converts single output column to one-hot encoding
func LoadCSVWithOneHot(path string, inputSize int, numClasses int) (Dataset, error) {

	file, err := os.Open(path)
	if err != nil {
		return Dataset{}, err
	}
	defer file.Close()

	reader := csv.NewReader(file)

	records, err := reader.ReadAll()
	if err != nil {
		return Dataset{}, err
	}

	var inputs [][]float64
	var outputs [][]float64

	// Skip header → start from index 1
	for i := 1; i < len(records); i++ {

		row := records[i]

		var input []float64
		var output []float64

		// Parse inputs
		for j := 0; j < inputSize; j++ {
			val, err := strconv.ParseFloat(row[j], 64)
			if err != nil {
				return Dataset{}, err
			}
			input = append(input, val)
		}

		// Parse class label (last column) and convert to one-hot
		classLabel, err := strconv.ParseFloat(row[inputSize], 64)
		if err != nil {
			return Dataset{}, err
		}

		// Convert to one-hot encoding
		classIndex := int(classLabel) - 1 // Classes are 1-indexed
		oneHot := make([]float64, numClasses)
		if classIndex >= 0 && classIndex < numClasses {
			oneHot[classIndex] = 1.0
		}
		output = oneHot

		inputs = append(inputs, input)
		outputs = append(outputs, output)
	}

	return Dataset{
		Inputs:  inputs,
		Outputs: outputs,
	}, nil
}

// NormalizeInputs normalizes the input features using min-max scaling to [0, 1]
func (d *Dataset) NormalizeInputs() {
	if len(d.Inputs) == 0 || len(d.Inputs[0]) == 0 {
		return
	}

	numFeatures := len(d.Inputs[0])
	mins := make([]float64, numFeatures)
	maxs := make([]float64, numFeatures)

	// Initialize with first sample
	for j := 0; j < numFeatures; j++ {
		mins[j] = d.Inputs[0][j]
		maxs[j] = d.Inputs[0][j]
	}

	// Find min and max for each feature
	for i := 0; i < len(d.Inputs); i++ {
		for j := 0; j < numFeatures; j++ {
			if d.Inputs[i][j] < mins[j] {
				mins[j] = d.Inputs[i][j]
			}
			if d.Inputs[i][j] > maxs[j] {
				maxs[j] = d.Inputs[i][j]
			}
		}
	}

	// Normalize each feature
	for i := 0; i < len(d.Inputs); i++ {
		for j := 0; j < numFeatures; j++ {
			range_ := maxs[j] - mins[j]
			if range_ > 0 {
				d.Inputs[i][j] = (d.Inputs[i][j] - mins[j]) / range_
			}
		}
	}
}
