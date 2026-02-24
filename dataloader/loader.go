package dataloader

import (
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/ThakurMayank5/gonn/dataset"
)

// FromCSV loads a Dataset from a CSV file using the provided CSVConfig.
//
// Supported modes:
//
//  1. Regression / pre-encoded targets:
//     Set InputColumns and TargetColumns to the relevant column indices.
//     Both must contain numeric values.
//
//  2. Classification with a label column (string or integer class names):
//     Set InputColumns, HasLabelColumn = true, and LabelColumn to the column index.
//     The labels are one-hot encoded into the output vector.
//     Set NumClasses if known; leave it 0 to auto-detect from the data.
//
// Modes can be combined: TargetColumns and LabelColumn can both be active at the
// same time — numeric targets are written first, then the one-hot vector is appended.
func FromCSV(filePath string, config dataset.CSVConfig) (dataset.Dataset, error) {

	file, err := os.Open(filePath)
	if err != nil {
		return dataset.Dataset{}, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	if config.Delimiter != 0 {
		reader.Comma = config.Delimiter
	}
	// Allow rows with different field counts (e.g. trailing comma in header).
	reader.FieldsPerRecord = -1

	records, err := reader.ReadAll()
	if err != nil {
		return dataset.Dataset{}, err
	}

	startRow := 0
	if config.HasHeader {
		startRow = 1
	}

	if len(records) <= startRow {
		return dataset.Dataset{}, fmt.Errorf("no data rows found in %s", filePath)
	}

	// --- Build label → index map (first pass if label column is used) ---
	labelIndex := map[string]int{}

	if config.HasLabelColumn {
		for i := startRow; i < len(records); i++ {
			row := records[i]
			if config.LabelColumn >= len(row) {
				return dataset.Dataset{}, fmt.Errorf("row %d: label column %d is out of range", i, config.LabelColumn)
			}
			label := strings.TrimSpace(row[config.LabelColumn])
			if _, exists := labelIndex[label]; !exists {
				// Try integer first — if it parses, use it directly as an index
				if _, parseErr := strconv.Atoi(label); parseErr != nil {
					// String label — will be sorted and assigned indices below
					labelIndex[label] = -1
				}
			}
		}

		// Assign stable indices to string labels (sorted alphabetically)
		stringLabels := make([]string, 0)
		for k, v := range labelIndex {
			if v == -1 {
				stringLabels = append(stringLabels, k)
			}
		}
		sort.Strings(stringLabels)
		for i, k := range stringLabels {
			labelIndex[k] = i
		}

		if config.NumClasses == 0 {
			if len(stringLabels) > 0 {
				config.NumClasses = len(stringLabels)
			} else {
				// All labels were integers — find max to determine NumClasses
				maxClass := 0
				for i := startRow; i < len(records); i++ {
					label := strings.TrimSpace(records[i][config.LabelColumn])
					idx, _ := strconv.Atoi(label)
					if idx > maxClass {
						maxClass = idx
					}
				}
				config.NumClasses = maxClass + 1
			}
		}
	}

	// --- Second pass: build inputs and outputs ---
	var inputs [][]float64
	var outputs [][]float64

	for i := startRow; i < len(records); i++ {
		row := records[i]

		// Input features
		input := make([]float64, 0, len(config.InputColumns))
		for _, col := range config.InputColumns {
			if col >= len(row) {
				return dataset.Dataset{}, fmt.Errorf("row %d: input column %d is out of range", i, col)
			}
			val, err := strconv.ParseFloat(strings.TrimSpace(row[col]), 64)
			if err != nil {
				return dataset.Dataset{}, fmt.Errorf("row %d, column %d: cannot parse %q as float: %v", i, col, row[col], err)
			}
			input = append(input, val)
		}

		// Numeric target columns
		output := make([]float64, 0)
		for _, col := range config.TargetColumns {
			if col >= len(row) {
				return dataset.Dataset{}, fmt.Errorf("row %d: target column %d is out of range", i, col)
			}
			val, err := strconv.ParseFloat(strings.TrimSpace(row[col]), 64)
			if err != nil {
				return dataset.Dataset{}, fmt.Errorf("row %d, column %d: cannot parse %q as float: %v", i, col, row[col], err)
			}
			output = append(output, val)
		}

		// Label column → one-hot encoding
		if config.HasLabelColumn {
			if config.LabelColumn >= len(row) {
				return dataset.Dataset{}, fmt.Errorf("row %d: label column %d is out of range", i, config.LabelColumn)
			}
			label := strings.TrimSpace(row[config.LabelColumn])

			classIdx, isString := labelIndex[label]
			if !isString {
				// It was a pure-integer label
				idx, err := strconv.Atoi(label)
				if err != nil {
					return dataset.Dataset{}, fmt.Errorf("row %d: unknown label %q", i, label)
				}
				classIdx = idx
			}

			oneHot := make([]float64, config.NumClasses)
			if classIdx >= 0 && classIdx < config.NumClasses {
				oneHot[classIdx] = 1.0
			}
			output = append(output, oneHot...)
		}

		inputs = append(inputs, input)
		outputs = append(outputs, output)
	}

	numOutputs := 0
	if len(outputs) > 0 {
		numOutputs = len(outputs[0])
	}

	// Apply input scaling if requested
	switch config.Scaling {
	case dataset.MinMaxNormalize:
		minMaxNormalize(inputs)
	case dataset.ZScoreStandardize:
		zScoreStandardize(inputs)
	}

	return dataset.Dataset{
		Inputs:      inputs,
		Outputs:     outputs,
		NumSamples:  len(inputs),
		NumFeatures: len(config.InputColumns),
		NumOutputs:  numOutputs,
	}, nil
}

// minMaxNormalize scales each feature column to [0, 1].
// x' = (x - min) / (max - min)
func minMaxNormalize(inputs [][]float64) {
	if len(inputs) == 0 {
		return
	}
	numFeatures := len(inputs[0])
	for f := 0; f < numFeatures; f++ {
		min, max := inputs[0][f], inputs[0][f]
		for _, row := range inputs {
			if row[f] < min {
				min = row[f]
			}
			if row[f] > max {
				max = row[f]
			}
		}
		r := max - min
		for i := range inputs {
			if r > 0 {
				inputs[i][f] = (inputs[i][f] - min) / r
			} else {
				inputs[i][f] = 0
			}
		}
	}
}

// zScoreStandardize scales each feature column to mean=0, std=1.
// x' = (x - mean) / std
func zScoreStandardize(inputs [][]float64) {
	if len(inputs) == 0 {
		return
	}
	n := float64(len(inputs))
	numFeatures := len(inputs[0])
	for f := 0; f < numFeatures; f++ {
		mean := 0.0
		for _, row := range inputs {
			mean += row[f]
		}
		mean /= n

		variance := 0.0
		for _, row := range inputs {
			d := row[f] - mean
			variance += d * d
		}
		variance /= n
		std := math.Sqrt(variance)

		for i := range inputs {
			if std > 0 {
				inputs[i][f] = (inputs[i][f] - mean) / std
			} else {
				inputs[i][f] = 0
			}
		}
	}
}
