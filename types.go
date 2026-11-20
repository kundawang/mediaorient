package mediaorient

import (
	"fmt"
	"image"
	"math"
)

// Result is a generic struct that represents the result of an operation.
//
// Parameters:
//   - Data is data of type T.
//   - Err is an error that indicates if the operation failed.
type Result[T any] struct {
	Data T
	Err  error
}

// IsSuccess returns true if the operation was successful (no error occurred), false otherwise.
func (r *Result[T]) IsSuccess() bool {
	return r.Err == nil
}

// Options configures how the orientation is calculated and reported.
type Options struct {
	// MinConfidence is the minimum confidence (a probability between 0 and 1) required to trust the calculated
	// rotation. Results whose confidence falls below this threshold are flagged with NeedsReview ("pending manual
	// confirmation") instead of being trusted blindly. A value of 0 disables the threshold, which is the default
	// and matches the historical behavior.
	MinConfidence float64
}

// DefaultOptions returns the default calculation options: no minimum confidence threshold.
func DefaultOptions() Options {
	return Options{MinConfidence: 0}
}

// Validate checks that the options hold sane values, returning an error otherwise.
func (o Options) Validate() error {
	switch {
	case math.IsNaN(o.MinConfidence):
		return fmt.Errorf("min confidence must not be NaN")
	case math.IsInf(o.MinConfidence, 0):
		return fmt.Errorf("min confidence must be finite, got %v", o.MinConfidence)
	case o.MinConfidence < 0 || o.MinConfidence > 1:
		return fmt.Errorf("min confidence must be between 0 and 1, got %v", o.MinConfidence)
	}

	return nil
}

// FrameConfidence describes the model's prediction for a single frame of media.
type FrameConfidence struct {
	// Index is the zero-based position of the frame in the media.
	Index int `json:"index"`
	// Rotation is the predicted clockwise rotation of this frame, in degrees (0, 90, 180 or 270).
	Rotation int `json:"rotation"`
	// Probabilities maps each rotation angle to the probability the model assigned to it for this frame.
	Probabilities map[int]float64 `json:"probabilities,omitempty"`
	// Confidence is the probability of the predicted rotation for this frame.
	Confidence float64 `json:"confidence"`
	// Blank is true when the frame is entirely black and was excluded from the aggregation.
	Blank bool `json:"blank,omitempty"`
	// Tie is true when the model assigned the top probability to more than one rotation for this frame.
	Tie bool `json:"tie,omitempty"`
}

// ConfidenceReport describes how much trust can be placed in a calculated orientation.
type ConfidenceReport struct {
	// Rotation is the final aggregated clockwise rotation, in degrees (0, 90, 180 or 270).
	Rotation int `json:"rotation"`
	// Confidence is the mean probability of the final rotation across all usable frames.
	Confidence float64 `json:"confidence"`
	// Threshold is the minimum confidence that was required (see Options.MinConfidence).
	Threshold float64 `json:"threshold"`
	// NeedsReview is true when the confidence is below the threshold (or there was no usable evidence),
	// meaning the rotation is pending manual confirmation and should not be applied automatically.
	NeedsReview bool `json:"needsReview"`
	// Reason explains, in human-readable form, why the result was flagged for review.
	Reason string `json:"reason,omitempty"`
	// Probabilities maps each rotation angle to its mean probability across all usable frames.
	Probabilities map[int]float64 `json:"probabilities,omitempty"`
	// Frames holds the per-frame predictions used in the aggregation.
	Frames []FrameConfidence `json:"frames"`
	// DivergentFrames lists the indices of usable frames whose prediction disagrees with the final rotation.
	DivergentFrames []int `json:"divergentFrames,omitempty"`
	// Tie is true when frames or probabilities tied and the deterministic tie-break rule decided the result.
	Tie bool `json:"tie,omitempty"`
}

// Media represents a media object.
type Media struct {
	// Name of the media.
	Name string `json:"name"`
	// Type of the media (e.g., image, video).
	Type string `json:"type"`
	// Rotation represents the rotation of the media.
	Rotation int `json:"rotation"`
	// Frames contain the image data of the media.
	Frames []image.Image `json:"-"`
	// Width represents the width of the media in pixels.
	Width int `json:"width"`
	// Height represents the height of the media in pixels.
	Height int `json:"height"`
	// Confidence holds the confidence report for the rotation. It is only populated when the media was
	// calculated with one of the "WithOptions" functions; the legacy functions leave it nil.
	Confidence *ConfidenceReport `json:"confidence,omitempty"`
}

func (m *Media) String() string {
	return fmt.Sprintf(`{Name: %s, Type: %s, Rotation: %d, Width: %d, Height: %d}`,
		m.Name, m.Type, m.Rotation, m.Width, m.Height)
}
