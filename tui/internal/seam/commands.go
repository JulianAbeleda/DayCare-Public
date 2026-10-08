package seam

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os/exec"
	"path/filepath"
	"strings"
)

// TrainArgv is the command `start` runs: the loop, pointed at the run folder, with the predeclared recipe
// verbatim. The recipe already carries --states, --envelope and --protocol; the model and the calculator
// runner come from DAYCARE_BASE_GGUF and DAYCARE_CALCULATE_RUNNER in the environment.
func (c Client) TrainArgv(runDir string, recipe []string) []string {
	return append([]string{c.Python, "-m", "daycare.nursery.rloo_posttool", "train", "--root", runDir}, recipe...)
}

// Generated is what a generate command reports.
type Generated struct {
	Kind   string `json:"kind"`
	Thing  string `json:"thing"`
	Path   string `json:"path"`
	Output string `json:"output"`
}

// FreezeTasks runs the task generator without the unpublished suites and without the network, so it works
// on a fresh machine. The GPU harvest and `states` that follow are printed by the setup check's fix text.
func (c Client) FreezeTasks(name string) ([]byte, error) {
	dir, err := c.RunDir(name)
	if err != nil {
		return nil, err
	}
	cmd := exec.Command(c.Python, "-m", "daycare.nursery.posttool_tasks", "freeze", "--root", dir,
		"--no-private-suites", "--countdown-per-size", "0")
	cmd.Dir = c.Repo
	var out bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &out
	if err := cmd.Run(); err != nil {
		return nil, &Error{Message: fmt.Sprintf("posttool_tasks freeze failed: %v", err), Stderr: tail(out.String(), 20), Code: 1}
	}
	return json.Marshal(Generated{Kind: "generated", Thing: "tasks", Path: filepath.Join(dir, "tasks.xml"),
		Output: tail(out.String(), 5)})
}

func tail(text string, n int) string {
	lines := strings.Split(strings.TrimRight(text, "\n"), "\n")
	if len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	return strings.Join(lines, "\n")
}

// NextName is the first `<prefix>-NNN` not already a run folder, for a generated predeclaration or task set.
func NextName(runs *Runs, prefix string) string {
	taken := map[string]bool{}
	if runs != nil {
		for _, r := range runs.Runs {
			taken[r.ID] = true
		}
	}
	for i := 1; ; i++ {
		name := fmt.Sprintf("%s-%03d", prefix, i)
		if !taken[name] {
			return name
		}
	}
}
