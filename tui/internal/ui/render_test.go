package ui

import (
	"encoding/json"
	"flag"
	"os"
	"path/filepath"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/muesli/termenv"

	"github.com/JulianAbeleda/DayCare/tui/internal/jobs"
	"github.com/JulianAbeleda/DayCare/tui/internal/seam"
)

var update = flag.Bool("update", false, "rewrite the golden screens under testdata/screens")

func load(t *testing.T, name string, v any) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "expected", name+".json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, v); err != nil {
		t.Fatal(err)
	}
}

func golden(t *testing.T, name, got string) {
	t.Helper()
	path := filepath.Join("..", "..", "testdata", "screens", name)
	if *update {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(got), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("%v (run with -update to write it)", err)
	}
	if string(want) != got {
		t.Errorf("%s differs from the golden screen:\n%s", name, got)
	}
}

// setupSample is a setup result as the seam reports it on a machine with nothing configured, trimmed to the
// checks a screen must show in each shape (ok, missing with a fix, missing with a generator).
func setupSample() *seam.Setup {
	gen := "predeclaration"
	return &seam.Setup{Kind: "setup", Ready: false, Python: "/usr/bin/python3", Repo: "/home/u/DayCare", Root: "/home/u/runs",
		Checks: []seam.Check{
			{ID: "numpy", Label: "numpy importable", OK: true, Detail: "/usr/bin/python3"},
			{ID: "model", Label: "DAYCARE_BASE_GGUF is a file", OK: false, Detail: "unset",
				Fix: "convert Nemotron 3 Nano 4B to a BF16 GGUF (docs/rl-training.md); export DAYCARE_BASE_GGUF=<file>"},
			{ID: "envelope", Label: "DAYCARE_ENVELOPE is the captured GameTerm request record", OK: false, Detail: "unset",
				Fix: "the envelope is captured from GameTerm (docs/rl-training.md); it cannot be generated"},
			{ID: "predeclaration", Label: "a predeclared run is waiting to start", OK: false, Detail: "none",
				Fix:      "python -m daycare.harness.runs predeclare --root <runs>/<run> --protocol <research file> --template run5",
				Generate: &gen},
		}}
}

type sample struct {
	runs                      seam.Runs
	stopped, waiting, running seam.Run
	job                       *jobs.Job
	tail                      []string
}

func loadSample(t *testing.T) sample {
	var s sample
	load(t, "list", &s.runs)
	load(t, "show-001", &s.stopped)
	load(t, "show-002", &s.waiting)
	load(t, "show-003", &s.running)
	s.job = &jobs.Job{ID: "posttool-fixture-003", PID: 4242, Alive: true, LogPath: "/home/u/.local/state/daycare-tui/posttool-fixture-003.log",
		StartedAt: "2026-10-08T10:00:00Z", Argv: []string{"python3", "-m", "daycare.nursery.rloo_posttool", "train"}}
	s.tail = []string{"load 5.1s setup 2.0s states 427 shared 1024",
		"update 1: reward 0.700 mixed 1 loss 0.01000 max_gap 0.0004 kl 0.00400 entropy 0.550",
		"update 2: reward 0.720 mixed 1 loss 0.02000 max_gap 0.0004 kl 0.00500 entropy 0.560"}
	return s
}

// program renders the whole frame: runs screen, open the second row, toggle the mode, resize.
func program(s sample, technical bool, width, height int) string {
	m := New(seam.Client{}, jobs.Store{})
	next, _ := m.Update(runsMsg{&s.runs, nil})
	next, _ = next.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("2")})
	next, _ = next.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("j")})
	next, _ = next.Update(runMsg{&s.stopped, nil})
	if technical {
		next, _ = next.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("t")})
	}
	next, _ = next.Update(tea.WindowSizeMsg{Width: width, Height: height})
	return next.View()
}

// The plain goldens are what NO_COLOR shows: lipgloss strips every colour under the Ascii profile.
func TestScreensPlain(t *testing.T) {
	lipgloss.SetColorProfile(termenv.Ascii)
	s := loadSample(t)
	for name, got := range map[string]string{
		"status-plain.txt":          StatusView(setupSample(), 1, false, 100),
		"status-technical.txt":      StatusView(setupSample(), 3, true, 100),
		"runs-plain.txt":            RunsView(&s.runs, 0, false, 100),
		"runs-technical.txt":        RunsView(&s.runs, 2, true, 140),
		"runs-empty.txt":            RunsView(&seam.Runs{Root: "/home/u/runs"}, 0, false, 60),
		"run-stopped-plain.txt":     RunView(&s.stopped, false, 120),
		"run-stopped-technical.txt": RunView(&s.stopped, true, 120),
		"run-predeclared.txt":       RunView(&s.waiting, false, 100),
		"run-in-progress.txt":       RunView(&s.running, false, 100),
		"adapters-plain.txt":        AdaptersView(&s.runs, false, 100),
		"adapters-technical.txt":    AdaptersView(&s.runs, true, 120),
		"job-plain.txt":             JobsView("posttool-fixture-003", s.job, s.tail, false, 100),
		"job-technical.txt":         JobsView("posttool-fixture-003", s.job, s.tail, true, 140),
		"job-none.txt":              JobsView("", nil, nil, false, 80),
		"program-run-technical.txt": program(s, true, 120, 60),
		"program-run-80x24.txt":     program(s, false, 80, 24),
	} {
		golden(t, name, got)
	}
}

// One styled capture per shape, with true colour on a dark background: what a terminal shows.
func TestScreensStyled(t *testing.T) {
	lipgloss.SetColorProfile(termenv.TrueColor)
	lipgloss.SetHasDarkBackground(true)
	defer lipgloss.SetColorProfile(termenv.Ascii)
	s := loadSample(t)
	golden(t, "styled/run-stopped-plain.ansi", RunView(&s.stopped, false, 100))
	golden(t, "styled/program-run-80x24.ansi", program(s, false, 80, 24))
	golden(t, "styled/status-plain.ansi", StatusView(setupSample(), 1, false, 100))
}

func TestModelNavigation(t *testing.T) {
	lipgloss.SetColorProfile(termenv.Ascii)
	s := loadSample(t)
	m := New(seam.Client{}, jobs.Store{})
	next, _ := m.Update(runsMsg{&s.runs, nil})
	next, _ = next.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("2")})
	next, _ = next.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("j")})
	next, _ = next.Update(runMsg{&s.stopped, nil})
	next, _ = next.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("t")})
	got := next.(Model)
	if got.screen != screenRun || !got.technical || got.jobID != "posttool-fixture-001" || got.mood() != faceWorried {
		t.Fatalf("model state wrong: screen %d technical %t job %q mood %q", got.screen, got.technical, got.jobID, got.mood())
	}
	next, _ = next.Update(tea.KeyMsg{Type: tea.KeyEsc})
	if next.(Model).screen != screenRuns {
		t.Fatal("esc must return to the runs screen")
	}
	if m.mood() != faceSleep {
		t.Fatalf("an empty nursery sleeps, got %q", m.mood())
	}
}
