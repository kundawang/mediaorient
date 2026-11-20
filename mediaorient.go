package mediaorient

import (
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"

	"github.com/vegidio/go-sak/fs"
	_ "golang.org/x/image/bmp"
	_ "golang.org/x/image/tiff"
	_ "golang.org/x/image/webp"

	"github.com/disintegration/imaging"
	ort "github.com/yalue/onnxruntime_go"
)

const Version = "<version>"

// modelInputSize is the width and height, in pixels, of the tensor fed to the neural network.
const modelInputSize = 384

// classRotations maps the model's output class indices to clockwise rotation angles in degrees.
var classRotations = []int{0, 270, 180, 90}

// rotationAngles lists every possible rotation angle in ascending order, so aggregation iterates deterministically.
var rotationAngles = []int{0, 90, 180, 270}

var session *ort.DynamicAdvancedSession

// runInference executes the ONNX model on a preprocessed input tensor and returns the raw logits.
//
// It is a package-level variable so tests can inject fixed logits without a real model or runtime.
var runInference = func(input []float32) ([]float32, error) {
	inputTensor, err := ort.NewTensor[float32](ort.NewShape(1, 3, modelInputSize, modelInputSize), input)
	if err != nil {
		return nil, err
	}
	defer inputTensor.Destroy()

	outputTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 4))
	if err != nil {
		return nil, err
	}
	defer outputTensor.Destroy()

	if err = session.Run([]ort.Value{inputTensor}, []ort.Value{outputTensor}); err != nil {
		return nil, err
	}

	return slices.Clone(outputTensor.GetData()), nil
}

// CalculateImageOrientation calculates the orientation of one or more images.
//
// When more than one image is provided they are treated as frames of a single video and their predictions are
// aggregated. The returned Media carries no confidence report; use CalculateImageOrientationWithOptions for that.
func CalculateImageOrientation(name string, images []image.Image) (*Media, error) {
	return calculateOrientation(name, images, DefaultOptions(), false)
}

// CalculateImageOrientationWithOptions behaves like CalculateImageOrientation, but also attaches a confidence
// report (per-rotation probabilities, per-frame predictions, divergence and a NeedsReview flag governed by
// Options.MinConfidence) to the returned Media.
func CalculateImageOrientationWithOptions(name string, images []image.Image, opts Options) (*Media, error) {
	return calculateOrientation(name, images, opts, true)
}

// CalculateFileOrientation calculates the orientation of a single image or video file.
//
// The returned Media carries no confidence report; use CalculateFileOrientationWithOptions for that.
func CalculateFileOrientation(filePath string) (*Media, error) {
	return calculateFileOrientation(filePath, DefaultOptions(), false)
}

// CalculateFileOrientationWithOptions behaves like CalculateFileOrientation, but also attaches a confidence
// report to the returned Media.
func CalculateFileOrientationWithOptions(filePath string, opts Options) (*Media, error) {
	return calculateFileOrientation(filePath, opts, true)
}

// CalculateFilesOrientation calculates the orientation of multiple files concurrently.
//
// The returned Media values carry no confidence report; use CalculateFilesOrientationWithOptions for that.
func CalculateFilesOrientation(filePaths []string) <-chan Result[Media] {
	return calculateFilesOrientation(filePaths, DefaultOptions(), false)
}

// CalculateFilesOrientationWithOptions behaves like CalculateFilesOrientation, but also attaches a confidence
// report to each returned Media.
func CalculateFilesOrientationWithOptions(filePaths []string, opts Options) <-chan Result[Media] {
	return calculateFilesOrientation(filePaths, opts, true)
}

// CalculateDirectoryOrientation calculates the orientation of every supported media file in a directory.
//
// The returned Media values carry no confidence report; use CalculateDirectoryOrientationWithOptions for that.
func CalculateDirectoryOrientation(directory string, mediaType string, recursive bool) (<-chan Result[Media], int) {
	return calculateDirectoryOrientation(directory, mediaType, recursive, DefaultOptions(), false)
}

// CalculateDirectoryOrientationWithOptions behaves like CalculateDirectoryOrientation, but also attaches a
// confidence report to each returned Media.
func CalculateDirectoryOrientationWithOptions(
	directory string,
	mediaType string,
	recursive bool,
	opts Options,
) (<-chan Result[Media], int) {
	return calculateDirectoryOrientation(directory, mediaType, recursive, opts, true)
}

// region - Private functions

func calculateOrientation(name string, images []image.Image, opts Options, withConfidence bool) (*Media, error) {
	if err := opts.Validate(); err != nil {
		return nil, err
	}

	if len(images) == 0 {
		return nil, fmt.Errorf("no images provided for '%s'", name)
	}

	frames := make([]FrameConfidence, 0, len(images))
	for i, img := range images {
		if img == nil {
			return nil, fmt.Errorf("image %d of '%s' is nil", i, name)
		}

		frame, err := analyzeFrame(i, img)
		if err != nil {
			return nil, err
		}

		frames = append(frames, frame)
	}

	return buildMedia(name, images, frames, opts, withConfidence), nil
}

func calculateFileOrientation(filePath string, opts Options, withConfidence bool) (*Media, error) {
	if err := opts.Validate(); err != nil {
		return nil, err
	}

	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("error opening file '%s': %w", filePath, err)
	}

	defer file.Close()

	ext := strings.ToLower(filepath.Ext(file.Name()))
	images := make([]image.Image, 0)

	if slices.Contains(validImageTypes, ext) {
		img, _, imgErr := image.Decode(file)
		if imgErr != nil {
			return nil, imgErr
		}

		images = append(images, img)

	} else if slices.Contains(validVideoTypes, ext) {
		videos, vidErr := extractFrames(file.Name())
		if vidErr != nil {
			return nil, vidErr
		}

		images = append(images, videos...)
	}

	if len(images) == 0 {
		return nil, fmt.Errorf("no valid images found in file '%s'", filePath)
	}

	return calculateOrientation(filePath, images, opts, withConfidence)
}

func calculateFilesOrientation(filePaths []string, opts Options, withConfidence bool) <-chan Result[Media] {
	out := make(chan Result[Media])

	go func() {
		defer close(out)

		var wg sync.WaitGroup
		sem := make(chan struct{}, runtime.NumCPU())

		for _, filePath := range filePaths {
			wg.Add(1)
			sem <- struct{}{}

			go func(file string) {
				defer wg.Done()
				defer func() { <-sem }()

				newMedia, err := calculateFileOrientation(file, opts, withConfidence)
				if err == nil {
					out <- Result[Media]{Data: *newMedia}
				} else {
					out <- Result[Media]{Err: err}
				}

			}(filePath)
		}

		wg.Wait()
		close(sem)
	}()

	return out
}

func calculateDirectoryOrientation(
	directory string,
	mediaType string,
	recursive bool,
	opts Options,
	withConfidence bool,
) (<-chan Result[Media], int) {
	mediaTypes := make([]string, 0)
	if mediaType == "image" || mediaType == "all" {
		mediaTypes = append(mediaTypes, validImageTypes...)
	}
	if mediaType == "video" || mediaType == "all" {
		mediaTypes = append(mediaTypes, validVideoTypes...)
	}

	flags := fs.LpFile
	if recursive {
		flags |= fs.LpRecursive
	}

	files, err := fs.ListPath(directory, flags, mediaTypes)
	if err != nil {
		out := make(chan Result[Media], 1)
		defer close(out)

		out <- Result[Media]{Err: err}
		return out, 0
	}

	return calculateFilesOrientation(files, opts, withConfidence), len(files)
}

// analyzeFrame runs the model on a single frame and converts the logits into a FrameConfidence.
//
// Entirely black frames carry no orientation evidence: they are flagged as blank and skipped without calling the
// model, so they cannot drag the aggregate result towards an arbitrary rotation.
func analyzeFrame(index int, img image.Image) (FrameConfidence, error) {
	frame := FrameConfidence{Index: index}

	input, blank := preprocess(img, modelInputSize)
	if blank {
		frame.Blank = true
		return frame, nil
	}

	logits, err := runInference(input)
	if err != nil {
		return frame, fmt.Errorf("error running inference on frame %d: %w", index, err)
	}

	if len(logits) != len(classRotations) {
		return frame, fmt.Errorf("unexpected number of logits %d on frame %d, expected %d",
			len(logits), index, len(classRotations))
	}

	probs := softmax(logits)
	bestIdx, tie := argmax(probs)

	frame.Rotation = classRotations[bestIdx]
	frame.Confidence = float64(probs[bestIdx])
	frame.Tie = tie
	frame.Probabilities = make(map[int]float64, len(probs))
	for i, p := range probs {
		frame.Probabilities[classRotations[i]] = float64(p)
	}

	return frame, nil
}

// buildMedia aggregates the per-frame predictions into a Media, optionally attaching a confidence report.
func buildMedia(
	name string,
	images []image.Image,
	frames []FrameConfidence,
	opts Options,
	withConfidence bool,
) *Media {
	rotation, meanProbs, divergent, tie, usable := aggregateFrames(frames)

	mediaType := "image"
	mediaFrames := images[:1]
	if len(images) > 1 {
		mediaType = "video"
		mediaFrames = nil
	}

	media := &Media{
		Name:     name,
		Type:     mediaType,
		Rotation: rotation,
		Frames:   mediaFrames,
		Width:    images[0].Bounds().Dx(),
		Height:   images[0].Bounds().Dy(),
	}

	if withConfidence {
		report := &ConfidenceReport{
			Rotation:        rotation,
			Threshold:       opts.MinConfidence,
			Probabilities:   meanProbs,
			Frames:          frames,
			DivergentFrames: divergent,
			Tie:             tie,
		}

		switch {
		case usable == 0:
			report.Confidence = 0
			report.NeedsReview = true
			report.Reason = "all frames are blank; no orientation evidence"
		default:
			report.Confidence = meanProbs[rotation]
			report.NeedsReview = report.Confidence < opts.MinConfidence
			if report.NeedsReview {
				report.Reason = fmt.Sprintf("confidence %.4f is below the threshold %.4f",
					report.Confidence, opts.MinConfidence)
			}
		}

		media.Confidence = report
	}

	return media
}

// aggregateFrames combines per-frame predictions into a final rotation using a deterministic rule:
//  1. the rotation with the most frame votes wins;
//  2. vote ties are broken by the highest summed class probability;
//  3. remaining ties are broken by the smallest angle.
//
// It also returns the mean probability per rotation, the indices of frames that disagree with the final rotation,
// whether any tie had to be broken, and the number of usable (non-blank) frames.
func aggregateFrames(frames []FrameConfidence) (
	rotation int,
	meanProbs map[int]float64,
	divergent []int,
	tie bool,
	usable int,
) {
	votes := make(map[int]int, len(rotationAngles))
	probSums := make(map[int]float64, len(rotationAngles))
	usableFrames := make([]FrameConfidence, 0, len(frames))

	for _, frame := range frames {
		if frame.Blank {
			continue
		}

		usableFrames = append(usableFrames, frame)
		votes[frame.Rotation]++
		for _, angle := range rotationAngles {
			probSums[angle] += frame.Probabilities[angle]
		}
		tie = tie || frame.Tie
	}

	usable = len(usableFrames)
	if usable == 0 {
		return 0, nil, nil, false, 0
	}

	rotation = rotationAngles[0]
	for _, angle := range rotationAngles[1:] {
		if votes[angle] > votes[rotation] ||
			(votes[angle] == votes[rotation] && probSums[angle] > probSums[rotation]) {
			rotation = angle
		}
	}

	// Another rotation reached the same vote count as the winner: the frames genuinely disagree.
	for _, angle := range rotationAngles {
		if angle != rotation && votes[angle] == votes[rotation] {
			tie = true
			break
		}
	}

	meanProbs = make(map[int]float64, len(rotationAngles))
	for _, angle := range rotationAngles {
		meanProbs[angle] = probSums[angle] / float64(usable)
	}

	for _, frame := range usableFrames {
		if frame.Rotation != rotation {
			divergent = append(divergent, frame.Index)
		}
	}

	return rotation, meanProbs, divergent, tie, usable
}

// argmax returns the index of the first maximum value (deterministic) and whether the maximum is shared by more
// than one element.
func argmax(probs []float32) (int, bool) {
	bestIdx := 0
	tie := false

	for i := 1; i < len(probs); i++ {
		if probs[i] > probs[bestIdx] {
			bestIdx = i
			tie = false
		} else if probs[i] == probs[bestIdx] {
			tie = true
		}
	}

	return bestIdx, tie
}

func softmax(x []float32) []float32 {
	maxValue := x[0]
	for _, v := range x {
		if v > maxValue {
			maxValue = v
		}
	}
	sum := float64(0)
	out := make([]float32, len(x))
	for i, v := range x {
		e := math.Exp(float64(v - maxValue))
		out[i] = float32(e)
		sum += e
	}
	for i := range out {
		out[i] /= float32(sum)
	}
	return out
}

// preprocess converts an image into a normalized NCHW tensor. It also reports whether the image is entirely
// black, in which case the tensor carries no orientation evidence.
func preprocess(img image.Image, size int) ([]float32, bool) {
	// Center-crop to square and resize (closest to common ImageNet eval)
	square := imaging.Fill(img, size, size, imaging.Center, imaging.Lanczos)

	b := square.Bounds()
	w, h := b.Dx(), b.Dy()
	mean := [3]float32{0.485, 0.456, 0.406} // ImageNet mean
	std := [3]float32{0.229, 0.224, 0.225}  // ImageNet std

	blank := true
	data := make([]float32, 3*w*h) // NCHW
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, bl, _ := square.At(x, y).RGBA()
			r8 := uint8(r >> 8)
			g8 := uint8(g >> 8)
			b8 := uint8(bl >> 8)
			if r8 != 0 || g8 != 0 || b8 != 0 {
				blank = false
			}

			rf := float32(r8) / 255.0
			gf := float32(g8) / 255.0
			bf := float32(b8) / 255.0
			i := y*w + x
			data[0*w*h+i] = (rf - mean[0]) / std[0]
			data[1*w*h+i] = (gf - mean[1]) / std[1]
			data[2*w*h+i] = (bf - mean[2]) / std[2]
		}
	}

	return data, blank
}

// endregion
