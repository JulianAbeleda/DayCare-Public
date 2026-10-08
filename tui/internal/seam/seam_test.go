package seam

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"testing"
)

// repoRoot is the DayCare checkout: this file sits at tui/internal/seam.
func repoRoot(t *testing.T) string {
	t.Helper()
	root, err := filepath.Abs(filepath.Join("..", "..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	return root
}

func expected(t *testing.T, name string) []byte {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(repoRoot(t), "tui", "testdata", "expected", name+".json"))
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

// The pinned JSON decodes into the typed contract with every load-bearing field present.
func TestPinnedContractDecodes(t *testing.T) {
	var runs Runs
	if err := json.Unmarshal(expected(t, "list"), &runs); err != nil {
		t.Fatal(err)
	}
	if len(runs.Runs) != 3 || runs.Runs[0].State != "stopped" || runs.Runs[0].Stopped.Update != 6 ||
		runs.Runs[0].Verdict != "fail" || runs.Runs[1].State != "predeclared" || runs.Runs[2].State != "in_progress" {
		t.Fatalf("list decoded wrong: %+v", runs.Runs)
	}
	var run Run
	if err := json.Unmarshal(expected(t, "show-001"), &run); err != nil {
		t.Fatal(err)
	}
	if run.ID != "posttool-fixture-001" || len(run.GateTable) != 9 || run.GateTable[0].Measured == nil ||
		run.GateTable[0].Measured.Diff != 15.5 || run.GateTable[8].Result != "fail" {
		t.Fatalf("gates decoded wrong: %+v", run.GateTable)
	}
	if !run.Window.Enabled || run.Window.Tripped == nil || run.Window.Tripped.Reason != run.Stopped.Reason {
		t.Fatalf("window decoded wrong: %+v", run.Window)
	}
	crossed := []string{}
	for _, r := range run.Window.Readings {
		if r.Crossed {
			crossed = append(crossed, r.Metric)
		}
	}
	if !reflect.DeepEqual(crossed, []string{"entropy"}) || run.Verify == nil || !run.Verify.BitExact {
		t.Fatalf("readings decoded wrong: %v verify %+v", crossed, run.Verify)
	}
	var waiting Run
	if err := json.Unmarshal(expected(t, "show-002"), &waiting); err != nil {
		t.Fatal(err)
	}
	if waiting.State != "predeclared" || waiting.Window.Enabled || len(waiting.Recipe) == 0 ||
		waiting.Recipe[len(waiting.Recipe)-2] != "--protocol" {
		t.Fatalf("predeclared run decoded wrong: %+v", waiting)
	}
}

func TestRunDirRefusesPaths(t *testing.T) {
	c := Client{Root: "/runs"}
	for _, bad := range []string{"", "../x", "a/b", ".hidden"} {
		if _, err := c.RunDir(bad); err == nil {
			t.Errorf("%q accepted", bad)
		}
	}
	if dir, err := c.RunDir("posttool-fixture-001"); err != nil || dir != "/runs/posttool-fixture-001" {
		t.Fatalf("got %q %v", dir, err)
	}
}

// livePython returns an interpreter that can import the seam's dependencies, or "" (then the live test skips).
func livePython(t *testing.T, repo string) string {
	t.Helper()
	python := os.Getenv("DAYCARE_PYTHON")
	if python == "" {
		python = "python3"
	}
	cmd := exec.Command(python, "-c", "import numpy, daycare.nursery.rl_triggers, daycare.artifact.record_xml")
	cmd.Dir = repo
	if err := cmd.Run(); err != nil {
		return ""
	}
	return python
}

// The live seam on the fixture prints exactly the pinned JSON; Python's own test pins the same files.
func TestLiveSeamMatchesPinnedContract(t *testing.T) {
	repo := repoRoot(t)
	python := livePython(t, repo)
	if python == "" {
		t.Skip("no interpreter with numpy + the RL modules (set DAYCARE_PYTHON); the pinned contract still holds")
	}
	c := Client{Python: python, Repo: repo, Root: filepath.Join("tui", "testdata", "fixture")}
	for name, call := range map[string]func() ([]byte, error){
		"list":     func() ([]byte, error) { _, raw, err := c.List(); return raw, err },
		"show-001": func() ([]byte, error) { _, raw, err := c.Show("posttool-fixture-001"); return raw, err },
		"show-003": func() ([]byte, error) { _, raw, err := c.Show("posttool-fixture-003"); return raw, err },
	} {
		raw, err := call()
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		var got, want any
		if err := json.Unmarshal(raw, &got); err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(expected(t, name), &want); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("%s: live seam differs from tui/testdata/expected/%s.json", name, name)
		}
	}
	if _, _, err := c.Show("missing-run"); err == nil {
		t.Fatal("a missing run must be an error")
	} else if e, ok := err.(*Error); !ok || e.Code != 1 {
		t.Fatalf("want a seam Error with code 1, got %#v", err)
	}
}

func TestCallNamesAMissingModule(t *testing.T) {
	fake := filepath.Join(t.TempDir(), "python")
	script := "#!/bin/sh\necho \"ModuleNotFoundError: No module named 'numpy.core'\" >&2\nexit 1\n"
	if err := os.WriteFile(fake, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	_, err := Client{Python: fake, Repo: t.TempDir()}.Call("list")
	want := fake + " has no numpy. Run `" + fake + " -m pip install numpy`, or pass -python with an interpreter that has it."
	if err == nil || err.Error() != want {
		t.Fatalf("got %v, want %q", err, want)
	}
}
