package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/samber/lo"
	"github.com/vegidio/mediaorient"
)

func printJSON(media []mediaorient.Media) error {
	if media == nil {
		media = []mediaorient.Media{}
	}

	data, err := json.MarshalIndent(media, "", "  ")
	if err != nil {
		return err
	}

	fmt.Println(string(data))
	return nil
}

func printCSV(media []mediaorient.Media) error {
	w := csv.NewWriter(os.Stdout)
	defer w.Flush()

	header := []string{"name", "type", "rotation", "confidence", "needs_review", "divergent_frames", "width", "height"}
	if err := w.Write(header); err != nil {
		return err
	}

	for _, m := range media {
		confidence := ""
		needsReview := ""
		divergent := ""

		if m.Confidence != nil {
			confidence = strconv.FormatFloat(m.Confidence.Confidence, 'f', 4, 64)
			needsReview = strconv.FormatBool(m.Confidence.NeedsReview)
			divergent = strings.Join(lo.Map(m.Confidence.DivergentFrames, func(index, _ int) string {
				return strconv.Itoa(index)
			}), ";")
		}

		row := []string{
			m.Name,
			m.Type,
			strconv.Itoa(m.Rotation),
			confidence,
			needsReview,
			divergent,
			strconv.Itoa(m.Width),
			strconv.Itoa(m.Height),
		}

		if err := w.Write(row); err != nil {
			return err
		}
	}

	return w.Error()
}
