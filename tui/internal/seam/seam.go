// Package seam is the Go side of DayCare's one machine-readable boundary: it runs
// `python -m daycare.harness.runs <action> --json-shaped` and decodes the JSON it prints.
//
// Python owns compute and data; this package never reads a run record itself and never
// writes one. The contract is pinned on both sides by tui/testdata (the fixture run
// folders and the expected JSON), checked by seam_test.go here and tests/test_harness_runs.py
// in Python.
package seam

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

// Client names the interpreter, the DayCare checkout (the cwd the module runs in) and the runs folder.
type Client struct {
	Python string
	Repo   string
	Root   string
}

// Error is what the seam reports when Python exits non-zero. Message is Python's own `error` field when it
// printed one, else the process error with its stderr.
type Error struct {
	Message string
	Stderr  string
	Code    int
}

func (e *Error) Error() string { return e.Message }

// Call runs one seam action and returns the raw JSON bytes Python printed, so agent mode can print the
// same bytes a human screen was built from.
func (c Client) Call(args ...string) ([]byte, error) {
	cmd := exec.Command(c.Python, append([]string{"-m", "daycare.harness.runs"}, args...)...)
	cmd.Dir = c.Repo
	var out, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &stderr
	err := cmd.Run()
	if err == nil {
		return bytes.TrimSpace(out.Bytes()), nil
	}
	code := -1
	var exit *exec.ExitError
	if errors.As(err, &exit) {
		code = exit.ExitCode()
	}
	var reported struct {
		Error string `json:"error"`
	}
	if json.Unmarshal(out.Bytes(), &reported) == nil && reported.Error != "" {
		return nil, &Error{Message: reported.Error, Stderr: stderr.String(), Code: code}
	}
	return nil, &Error{Message: fmt.Sprintf("%s -m daycare.harness.runs %s: %v", c.Python, strings.Join(args, " "), err),
		Stderr: stderr.String(), Code: code}
}

func (c Client) decode(v any, args ...string) ([]byte, error) {
	raw, err := c.Call(args...)
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(raw, v); err != nil {
		return nil, fmt.Errorf("seam output is not the expected shape: %w", err)
	}
	return raw, nil
}

// RunDir resolves a run id inside the runs folder and refuses path tricks.
func (c Client) RunDir(id string) (string, error) {
	if id == "" || id != filepath.Base(id) || strings.HasPrefix(id, ".") {
		return "", fmt.Errorf("run id must be a folder name under the runs folder: %q", id)
	}
	return filepath.Join(c.Root, id), nil
}

func (c Client) Setup() (*Setup, []byte, error) {
	var s Setup
	raw, err := c.decode(&s, "setup", "--root", c.Root, "--repo", c.Repo)
	return &s, raw, err
}

func (c Client) List() (*Runs, []byte, error) {
	var r Runs
	raw, err := c.decode(&r, "list", "--root", c.Root)
	return &r, raw, err
}

func (c Client) Show(id string) (*Run, []byte, error) {
	dir, err := c.RunDir(id)
	if err != nil {
		return nil, nil, err
	}
	var r Run
	raw, err := c.decode(&r, "show", "--root", dir)
	return &r, raw, err
}

// Predeclaration is what `predeclare` needs; Recipe empty means the template's recipe.
type Predeclaration struct {
	ID, Protocol, Template, Hypothesis, States, Envelope string
	Gates                                                []string
	Recipe                                               []string
}

func (c Client) Predeclare(p Predeclaration) ([]byte, error) {
	dir, err := c.RunDir(p.ID)
	if err != nil {
		return nil, err
	}
	args := []string{"predeclare", "--root", dir, "--protocol", p.Protocol}
	for flag, value := range map[string]string{"--template": p.Template, "--hypothesis": p.Hypothesis,
		"--states": p.States, "--envelope": p.Envelope} {
		if value != "" {
			args = append(args, flag, value)
		}
	}
	for _, g := range p.Gates {
		args = append(args, "--gate", g)
	}
	if len(p.Recipe) > 0 {
		args = append(append(args, "--recipe"), p.Recipe...)
	}
	return c.Call(args...)
}

func (c Client) Score(id, gate string, diff float64, lo, hi *float64, note string) ([]byte, error) {
	dir, err := c.RunDir(id)
	if err != nil {
		return nil, err
	}
	args := []string{"score", "--root", dir, "--gate", gate, "--diff", strconv.FormatFloat(diff, 'g', -1, 64)}
	if lo != nil && hi != nil {
		args = append(args, "--lo", strconv.FormatFloat(*lo, 'g', -1, 64), "--hi", strconv.FormatFloat(*hi, 'g', -1, 64))
	}
	if note != "" {
		args = append(args, "--note", note)
	}
	return c.Call(args...)
}

func (c Client) Adopt(id, by, exception string) ([]byte, error) {
	dir, err := c.RunDir(id)
	if err != nil {
		return nil, err
	}
	return c.Call("adopt", "--root", dir, "--by", by, "--exception", exception)
}
