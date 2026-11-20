package mediaorient

import (
	"encoding/json"
	"errors"
	"image"
	"image/color"
	"math"
	"strings"
	"testing"
)

// region - Test helpers

func solidImage(c color.NRGBA) image.Image {
	img := image.NewNRGBA(image.Rect(0, 0, 64, 64))
	for y := 0; y < 64; y++ {
		for x := 0; x < 64; x++ {
			img.SetNRGBA(x, y, c)
		}
	}
	return img
}

var (
	testImage  = solidImage(color.NRGBA{R: 120, G: 80, B: 200, A: 255})
	blackImage = solidImage(color.NRGBA{R: 0, G: 0, B: 0, A: 255})
)

// stubInference replaces the ONNX inference with a function that returns fixed logits per call.
func stubInference(t *testing.T, fn func(call int) ([]float32, error)) {
	t.Helper()

	orig := runInference
	call := 0
	runInference = func(_ []float32) ([]float32, error) {
		c := call
		call++
		return fn(c)
	}
	t.Cleanup(func() { runInference = orig })
}

func fixedLogits(t *testing.T, logits ...[]float32) {
	t.Helper()
	stubInference(t, func(call int) ([]float32, error) {
		if call >= len(logits) {
			return nil, errors.New("unexpected extra inference call")
		}
		return logits[call], nil
	})
}

// softmax64 computes the expected probabilities in float64 for assertions.
func softmax64(logits ...float64) []float64 {
	maxValue := logits[0]
	for _, v := range logits {
		if v > maxValue {
			maxValue = v
		}
	}

	sum := 0.0
	out := make([]float64, len(logits))
	for i, v := range logits {
		out[i] = math.Exp(v - maxValue)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}

	return out
}

func almostEqual(t *testing.T, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-5 {
		t.Errorf("got %v, want %v", got, want)
	}
}

// endregion

func TestSingleImageConfidence(t *testing.T) {
	fixedLogits(t, []float32{0, 1, 2, 3})
	probs := softmax64(0, 1, 2, 3)

	media, err := CalculateImageOrientationWithOptions("test.png", []image.Image{testImage}, Options{MinConfidence: 0.5})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if media.Confidence == nil {
		t.Fatal("expected a confidence report")
	}

	report := media.Confidence

	// Class 3 maps to a 90-degree clockwise rotation
	if media.Rotation != 90 || report.Rotation != 90 {
		t.Errorf("got rotation %d (report %d), want 90", media.Rotation, report.Rotation)
	}

	almostEqual(t, report.Confidence, probs[3])
	almostEqual(t, report.Probabilities[0], probs[0])
	almostEqual(t, report.Probabilities[90], probs[3])
	almostEqual(t, report.Probabilities[180], probs[2])
	almostEqual(t, report.Probabilities[270], probs[1])

	if report.NeedsReview {
		t.Errorf("confidence %.4f should pass the 0.5 threshold", report.Confidence)
	}
	if len(report.DivergentFrames) != 0 {
		t.Errorf("got divergent frames %v, want none", report.DivergentFrames)
	}
	if len(report.Frames) != 1 || report.Frames[0].Rotation != 90 {
		t.Errorf("unexpected per-frame report: %+v", report.Frames)
	}
}

func TestThresholdFlagsLowConfidence(t *testing.T) {
	// Probabilities: ~[0.4404, 0.4404, 0.0596, 0.0596] -> top confidence ~0.4404
	fixedLogits(t, []float32{2, 2, 0, 0}, []float32{2, 2, 0, 0})
	probs := softmax64(2, 2, 0, 0)

	below, err := CalculateImageOrientationWithOptions("a.png", []image.Image{testImage}, Options{MinConfidence: 0.5})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !below.Confidence.NeedsReview {
		t.Errorf("confidence %.4f should be flagged below the 0.5 threshold", below.Confidence.Confidence)
	}
	if below.Confidence.Reason == "" {
		t.Error("expected a reason explaining the review flag")
	}
	// The rotation is still reported, but must not be trusted automatically
	if below.Rotation != 0 {
		t.Errorf("got rotation %d, want 0", below.Rotation)
	}

	above, err := CalculateImageOrientationWithOptions("b.png", []image.Image{testImage}, Options{MinConfidence: 0.4})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	almostEqual(t, above.Confidence.Confidence, probs[0])
	if above.Confidence.NeedsReview {
		t.Errorf("confidence %.4f should pass the 0.4 threshold", above.Confidence.Confidence)
	}
}

func TestMultiFrameAggregationAndDivergence(t *testing.T) {
	fixedLogits(t,
		[]float32{5, 0, 0, 0}, // frame 0 -> 0 degrees
		[]float32{0, 5, 0, 0}, // frame 1 -> 270 degrees (divergent)
		[]float32{5, 0, 0, 0}, // frame 2 -> 0 degrees
	)

	media, err := CalculateImageOrientationWithOptions("video.mp4",
		[]image.Image{testImage, testImage, testImage}, DefaultOptions())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if media.Type != "video" {
		t.Errorf("got type %q, want video", media.Type)
	}
	if media.Rotation != 0 {
		t.Errorf("got rotation %d, want 0 (majority vote)", media.Rotation)
	}

	report := media.Confidence
	if len(report.DivergentFrames) != 1 || report.DivergentFrames[0] != 1 {
		t.Errorf("got divergent frames %v, want [1]", report.DivergentFrames)
	}

	p := softmax64(5, 0, 0, 0)[0]
	q := softmax64(0, 5, 0, 0)[0]
	almostEqual(t, report.Confidence, (p+q+p)/3)
	almostEqual(t, report.Probabilities[270], (q+p+q)/3)
}

func TestVoteTieBrokenBySummedProbability(t *testing.T) {
	fixedLogits(t,
		[]float32{5, 0, 0, 0}, // frame 0 -> 0 degrees, strong
		[]float32{0, 4, 0, 0}, // frame 1 -> 270 degrees, weaker class evidence overall
	)

	media, err := CalculateImageOrientationWithOptions("video.mp4",
		[]image.Image{testImage, testImage}, DefaultOptions())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Votes tie 1-1; the summed probability of class 0 (rotation 0) is higher
	if media.Rotation != 0 {
		t.Errorf("got rotation %d, want 0 (probability tie-break)", media.Rotation)
	}
	if !media.Confidence.Tie {
		t.Error("expected the tie to be reported")
	}
	if len(media.Confidence.DivergentFrames) != 1 || media.Confidence.DivergentFrames[0] != 1 {
		t.Errorf("got divergent frames %v, want [1]", media.Confidence.DivergentFrames)
	}
}

func TestProbabilityTieHasDeterministicResult(t *testing.T) {
	fixedLogits(t, []float32{1, 1, 1, 1})

	media, err := CalculateImageOrientationWithOptions("tie.png", []image.Image{testImage}, DefaultOptions())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// All probabilities are equal: the smallest angle wins deterministically
	if media.Rotation != 0 {
		t.Errorf("got rotation %d, want 0 (deterministic tie-break)", media.Rotation)
	}
	if !media.Confidence.Tie {
		t.Error("expected the probability tie to be reported")
	}
	almostEqual(t, media.Confidence.Confidence, 0.25)
	for _, angle := range rotationAngles {
		almostEqual(t, media.Confidence.Probabilities[angle], 0.25)
	}
}

func TestBlankFrames(t *testing.T) {
	t.Run("all frames blank", func(t *testing.T) {
		stubInference(t, func(call int) ([]float32, error) {
			return nil, errors.New("inference must not run on blank frames")
		})

		media, err := CalculateImageOrientationWithOptions("black.png",
			[]image.Image{blackImage, blackImage}, DefaultOptions())
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if media.Rotation != 0 {
			t.Errorf("got rotation %d, want 0", media.Rotation)
		}
		if !media.Confidence.NeedsReview {
			t.Error("expected an all-black video to be flagged for manual review")
		}
		if media.Confidence.Confidence != 0 {
			t.Errorf("got confidence %v, want 0", media.Confidence.Confidence)
		}
		if media.Confidence.Reason == "" {
			t.Error("expected a reason explaining the review flag")
		}
		if !media.Confidence.Frames[0].Blank || !media.Confidence.Frames[1].Blank {
			t.Error("expected both frames to be marked blank")
		}
	})

	t.Run("blank frames are excluded from aggregation", func(t *testing.T) {
		fixedLogits(t, []float32{0, 0, 5, 0}) // only the non-blank frame reaches the model
		probs := softmax64(0, 0, 5, 0)

		media, err := CalculateImageOrientationWithOptions("mixed.mp4",
			[]image.Image{blackImage, testImage}, DefaultOptions())
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if media.Rotation != 180 {
			t.Errorf("got rotation %d, want 180", media.Rotation)
		}
		almostEqual(t, media.Confidence.Confidence, probs[2])
		if !media.Confidence.Frames[0].Blank {
			t.Error("expected frame 0 to be marked blank")
		}
		if len(media.Confidence.DivergentFrames) != 0 {
			t.Errorf("blank frames must not count as divergent, got %v", media.Confidence.DivergentFrames)
		}
	})
}

func TestEmptyImages(t *testing.T) {
	if _, err := CalculateImageOrientation("none.png", nil); err == nil {
		t.Error("expected an error for an empty image list (legacy API)")
	}
	if _, err := CalculateImageOrientationWithOptions("none.png", []image.Image{}, DefaultOptions()); err == nil {
		t.Error("expected an error for an empty image list")
	}
}

func TestInvalidThresholds(t *testing.T) {
	thresholds := []float64{math.NaN(), math.Inf(1), math.Inf(-1), -0.1, 1.1}

	for _, threshold := range thresholds {
		_, err := CalculateImageOrientationWithOptions("test.png", []image.Image{testImage},
			Options{MinConfidence: threshold})
		if err == nil {
			t.Errorf("expected an error for threshold %v", threshold)
		}
	}
}

func TestInferenceFailure(t *testing.T) {
	stubInference(t, func(call int) ([]float32, error) {
		return nil, errors.New("onnx runtime exploded")
	})

	_, err := CalculateImageOrientationWithOptions("test.png", []image.Image{testImage}, DefaultOptions())
	if err == nil {
		t.Fatal("expected the inference error to be propagated")
	}
	if !strings.Contains(err.Error(), "onnx runtime exploded") {
		t.Errorf("error should wrap the inference failure, got: %v", err)
	}
}

func TestLegacyAPIBehaviorUnchanged(t *testing.T) {
	fixedLogits(t, []float32{0, 0, 0, 5})

	media, err := CalculateImageOrientation("test.png", []image.Image{testImage})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if media.Rotation != 90 {
		t.Errorf("got rotation %d, want 90", media.Rotation)
	}
	if media.Confidence != nil {
		t.Error("legacy API must not attach a confidence report")
	}

	// The JSON shape of the legacy API is unchanged
	data, err := json.Marshal(media)
	if err != nil {
		t.Fatalf("unexpected marshal error: %v", err)
	}
	if strings.Contains(string(data), "confidence") {
		t.Errorf("legacy JSON output must not contain a confidence field, got: %s", data)
	}
}

func TestConfidenceReportJSON(t *testing.T) {
	fixedLogits(t, []float32{0, 1, 2, 3})

	media, err := CalculateImageOrientationWithOptions("test.png", []image.Image{testImage},
		Options{MinConfidence: 0.9})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	data, err := json.Marshal(media)
	if err != nil {
		t.Fatalf("unexpected marshal error: %v", err)
	}

	for _, field := range []string{`"confidence"`, `"needsReview":true`, `"probabilities"`, `"frames"`, `"threshold"`} {
		if !strings.Contains(string(data), field) {
			t.Errorf("JSON output is missing %s, got: %s", field, data)
		}
	}
}
