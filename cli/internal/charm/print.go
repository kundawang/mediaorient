package charm

import (
	"fmt"
	"sort"
	"strconv"

	"github.com/samber/lo"
	"github.com/vegidio/mediaorient"
)

func PrintDownloadModel() {
	fmt.Printf("\n🌟 %s\n", yellow.Render("Downloading the neural network model..."))
}

func PrintCalculateFiles(amount int) {
	fmt.Printf("⏳ Calculating orientation in %s files\n", green.Render(strconv.Itoa(amount)))
}

func PrintCalculateDirectory(dir string) {
	fmt.Printf("⏳ Calculating orientation in the directory %s\n", green.Render(dir))
}

func PrintError(message string, a ...interface{}) {
	format := fmt.Sprintf(message, a...)
	fmt.Printf("🧨 %s\n", red.Render(format))
}

func PrintReport(media []mediaorient.Media, includeZero bool) {
	// Media flagged for manual confirmation is listed separately, never under a rotation group
	review, trusted := lo.FilterReject(media, func(m mediaorient.Media, _ int) bool {
		return m.Confidence != nil && m.Confidence.NeedsReview
	})

	groups := lo.GroupBy(trusted, func(m mediaorient.Media) int {
		return m.Rotation
	})

	if !includeZero {
		delete(groups, 0)
	}

	angles := lo.Keys(groups)
	sort.Ints(angles)

	for _, k := range angles {
		fmt.Printf("\nMedia rotated %s clockwise:\n", getRotationColor(k))

		for _, m := range groups[k] {
			fmt.Printf("  -> %s%s\n", bold.Render(m.Name), formatConfidence(m))
		}
	}

	if len(review) > 0 {
		fmt.Printf("\n%s\n", orange.Render("⚠ Needs manual confirmation (confidence below threshold):"))

		for _, m := range review {
			fmt.Printf("  -> %s%s\n", bold.Render(m.Name), formatConfidence(m))
		}
	}
}

// region - Private functions

func formatConfidence(m mediaorient.Media) string {
	if m.Confidence == nil {
		return ""
	}

	text := fmt.Sprintf(" (confidence %.1f%%", m.Confidence.Confidence*100)

	if len(m.Confidence.Frames) > 1 && len(m.Confidence.DivergentFrames) > 0 {
		text += fmt.Sprintf(", %d/%d frames disagree", len(m.Confidence.DivergentFrames), len(m.Confidence.Frames))
	}

	return gray.Render(text + ")")
}

func getRotationColor(rotation int) string {
	switch rotation {
	case 90:
		return yellow.Render("90º")
	case 180:
		return blue.Render("180º")
	case 270:
		return red.Render("270º")
	default:
		return bold.Render("0º")
	}
}

// endregion
