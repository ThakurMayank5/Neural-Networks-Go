package imageprocessing

import (
	"fmt"
	"image"
	"os"

	_ "image/jpeg"
	_ "image/png"
)

// LoadImage loads an image from the specified file path and returns a 3D slice of normalized pixel values in RGB format.
func LoadImage(filePath string) ([][][]float64, error) {

	fmt.Printf("Loading image from: %s\n", filePath)

	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// Decode the image
	img, _, err := image.Decode(file)

	// The second return value is the image format in string form

	if err != nil {
		return nil, err
	}

	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y

	fmt.Printf("Image dimensions: %dx%d\n", width, height)

	imageData := make([][][]float64, 3) // Assuming RGB

	// Initialize the 3D slice for RGB channels
	for i := 0; i < 3; i++ {
		imageData[i] = make([][]float64, height)
		for j := 0; j < height; j++ {
			imageData[i][j] = make([]float64, width)
		}
	}

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA()

			// Normalize the pixel values to [0, 1] range
			// 0xFFFF is the maximum value for a color channel in Go's RGBA representation

			imageData[0][y][x] = float64(r) / 0xFFFF
			imageData[1][y][x] = float64(g) / 0xFFFF
			imageData[2][y][x] = float64(b) / 0xFFFF
		}
	}

	return imageData, nil
}
