package main

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"strings"
	"testing"

	"github.com/vegidio/mediaorient"
)

func captureStdout(t *testing.T, fn func() error) string {
	t.Helper()

	orig := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("failed to create pipe: %v", err)
	}
	os.Stdout = w

	fnErr := fn()

	_ = w.Close()
	os.Stdout = orig

	var buf bytes.Buffer
	if _, err := io.Copy(&buf, r); err != nil {
		t.Fatalf("failed to read captured output: %v", err)
	}

	if fnErr != nil {
		t.Fatalf("unexpected error: %v", fnErr)
	}

	return buf.String()
}

func testMedia() []mediaorient.Media {
	return []mediaorient.Media{
		{
			Name:     "photo.png",
			Type:     "image",
			Rotation: 90,
			Width:    200,
			Height:   100,
			Confidence: &mediaorient.ConfidenceReport{
				Rotation:      90,
				Confidence:    0.6439,
				Threshold:     0.5,
				NeedsReview:   false,
				Probabilities: map[int]float64{0: 0.03, 90: 0.6439, 180: 0.23, 270: 0.0961},
				Frames:        []mediaorient.FrameConfidence{{Index: 0, Rotation: 90, Confidence: 0.6439}},
			},
		},
		{
			Name:     "clip.mp4",
			Type:     "video",
			Rotation: 270,
			Width:    1920,
			Height:   1080,
			Confidence: &mediaorient.ConfidenceReport{
				Rotation:        270,
				Confidence:      0.31,
				Threshold:       0.5,
				NeedsReview:     true,
				Reason:          "confidence 0.3100 is below the threshold 0.5000",
				Probabilities:   map[int]float64{0: 0.2, 90: 0.29, 180: 0.2, 270: 0.31},
				Frames:          []mediaorient.FrameConfidence{{Index: 0, Rotation: 270, Confidence: 0.31}},
				DivergentFrames: []int{1, 4},
			},
		},
	}
}

func TestPrintJSON(t *testing.T) {
	out := captureStdout(t, func() error { return printJSON(testMedia()) })

	var decoded []map[string]any
	if err := json.Unmarshal([]byte(out), &decoded); err != nil {
		t.Fatalf("output is not valid JSON: %v\n%s", err, out)
	}

	if len(decoded) != 2 {
		t.Fatalf("got %d entries, want 2", len(decoded))
	}

	confidence, ok := decoded[1]["confidence"].(map[string]any)
	if !ok {
		t.Fatalf("expected a confidence object in: %s", out)
	}
	if confidence["needsReview"] != true {
		t.Errorf("expected needsReview to be true in: %s", out)
	}
	if confidence["threshold"] != 0.5 {
		t.Errorf("expected threshold 0.5 in: %s", out)
	}
}

func TestPrintCSV(t *testing.T) {
	out := captureStdout(t, func() error { return printCSV(testMedia()) })

	lines := strings.Split(strings.TrimSpace(out), "\n")
	if len(lines) != 3 {
		t.Fatalf("got %d lines, want 3 (header + 2 rows):\n%s", len(lines), out)
	}

	if lines[0] != "name,type,rotation,confidence,needs_review,divergent_frames,width,height" {
		t.Errorf("unexpected header: %s", lines[0])
	}
	if !strings.Contains(lines[1], "photo.png,image,90,0.6439,false") {
		t.Errorf("unexpected first row: %s", lines[1])
	}
	if !strings.Contains(lines[2], "clip.mp4,video,270,0.3100,true,1;4") {
		t.Errorf("unexpected second row: %s", lines[2])
	}
}
