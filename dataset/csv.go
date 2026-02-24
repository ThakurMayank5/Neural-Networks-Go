package dataset

// ScalingMethod controls how input features are scaled after loading.
type ScalingMethod string

const (
	// NoScaling leaves input values as-is.
	NoScaling ScalingMethod = ""

	// MinMaxNormalize scales each feature to [0, 1] using min-max normalization.
	// x' = (x - min) / (max - min)
	MinMaxNormalize ScalingMethod = "minmax"

	// ZScoreStandardize scales each feature to mean=0, std=1.
	// x' = (x - mean) / std
	ZScoreStandardize ScalingMethod = "zscore"
)

// CSVConfig controls how a CSV file is parsed into a Dataset.
type CSVConfig struct {

	// HasHeader skips the first row of the file if true.
	HasHeader bool

	// InputColumns are the column indices to use as input features.
	// These must contain numeric values.
	InputColumns []int

	// TargetColumns are the column indices to use as numeric targets.
	// Use this for regression or when targets are already numeric / one-hot.
	TargetColumns []int

	// HasLabelColumn enables label-column mode.
	// When true, LabelColumn is read as a class label (string or integer)
	// and one-hot encoded into the output vector.
	HasLabelColumn bool

	// LabelColumn is the column index that contains the class label.
	// Only used when HasLabelColumn is true.
	LabelColumn int

	// NumClasses is the number of classes for one-hot encoding.
	// Set to 0 to let the loader auto-detect unique labels from the data.
	NumClasses int

	// Scaling controls whether and how input features are scaled after loading.
	// Use MinMaxNormalize or ZScoreStandardize. Defaults to NoScaling.
	Scaling ScalingMethod

	// Delimiter is the field separator. Defaults to comma if zero.
	Delimiter rune
}
