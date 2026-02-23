package gpuprocessing

// /*
// #cgo LDFLAGS: ../cuda/dotproduct.o -lcudart
// #include "dot.h"
// */
// import "C"

// import (
// 	"fmt"
// 	"unsafe"
// )

// func DotProduct(a, b []float64) (float64, error) {

// 	if len(a) != len(b) {
// 		return 0, fmt.Errorf("vectors must be of the same length")
// 	}

// 	if len(a) == 0 {
// 		return 0, fmt.Errorf("vectors cannot be empty")
// 	}

// 	// Convert to float32 (CUDA expects float)
// 	a32 := make([]float32, len(a))
// 	b32 := make([]float32, len(b))

// 	for i := range a {
// 		a32[i] = float32(a[i])
// 		b32[i] = float32(b[i])
// 	}

// 	n := C.int(len(a32))

// 	dot := C.dotProductCUDA(
// 		(*C.float)(unsafe.Pointer(&a32[0])),
// 		(*C.float)(unsafe.Pointer(&b32[0])),
// 		n,
// 	)

// 	return float64(dot), nil
// }
