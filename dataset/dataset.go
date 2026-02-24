package dataset

type Dataset struct {
	Inputs      [][]float64
	Outputs     [][]float64
	NumSamples  int
	NumFeatures int
	NumOutputs  int
}
