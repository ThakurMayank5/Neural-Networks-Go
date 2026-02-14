package neuralnetwork

func (model *Model) Backpropagate(layerCache ModelWeightsAndBiases, target []float64, prediction []float64) (float64, error) {

	deltas := make([][]float64, len(layerCache.Weights))

	// ---- OUTPUT DELTA ----
	last := len(layerCache.Weights) - 1
	deltas[last] = make([]float64, len(prediction))

	for i := range prediction {
		deltas[last][i] = prediction[i] - target[i]
	}

	

	return 0.0, nil
}
