package ui

import (
	"encoding/json"
	"flag"
	"github.com/charmbracelet/bubbles/viewport"
	"os"
	"path/filepath"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"
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

func press(m tea.Model, k string) tea.Model {
	msg := tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(k)}
	switch k {
	case "enter":
		msg = tea.KeyMsg{Type: tea.KeyEnter}
	case "esc":
		msg = tea.KeyMsg{Type: tea.KeyEsc}
	}
	next, _ := m.Update(msg)
	return next
}

// program feeds the model the seam's answers in the order they arrive on a real start, then sizes it.
func program(s sample, setup *seam.Setup, run *seam.Run, job *jobs.Job, tail []string) tea.Model {
	var m tea.Model = New(seam.Client{}, jobs.Store{})
	for _, msg := range []tea.Msg{setupMsg{setup, nil}, runsMsg{&s.runs, nil}, tea.WindowSizeMsg{Width: 80, Height: 24}} {
		m, _ = m.Update(msg)
	}
	if run != nil {
		m, _ = m.Update(runMsg{run, nil})
	}
	if job != nil {
		m, _ = m.Update(jobMsg{job, tail})
	}
	return m
}

func ready() *seam.Setup {
	return &seam.Setup{Kind: "setup", Ready: true, Checks: []seam.Check{{ID: "numpy", Label: "numpy importable", OK: true, Detail: "/usr/bin/python3"}}}
}

func facts(m tea.Model) Facts { return m.(Model).f }

// The plain goldens are what NO_COLOR shows: lipgloss strips every colour under the Ascii profile.
func TestScreensPlain(t *testing.T) {
	lipgloss.SetColorProfile(termenv.Ascii)
	s := loadSample(t)
	stopped := program(s, ready(), &s.stopped, nil, nil)
	waiting := program(s, ready(), &s.waiting, nil, nil)
	running := program(s, ready(), &s.running, s.job, s.tail)
	notReady := program(s, setupSample(), nil, nil, nil)
	scoring := press(press(press(press(stopped, "j"), "enter"), "enter"), "-1")
	for name, got := range map[string]string{
		"checklist-stopped.txt":     stopped.View(),
		"checklist-predeclared.txt": waiting.View(),
		"checklist-training.txt":    running.View(),
		"checklist-not-ready.txt":   notReady.View(),
		"score-form-80x24.txt":      scoring.View(),
		"detail-ready.txt":          detail(facts(notReady), 0),
		"detail-predeclare.txt":     detail(facts(stopped), 1),
		"detail-train.txt":          detail(facts(stopped), 2),
		"detail-score.txt":          detail(facts(stopped), 3),
		"detail-verdict.txt":        detail(facts(stopped), 4),
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
	stopped := program(s, ready(), &s.stopped, nil, nil)
	golden(t, "styled/checklist-stopped.ansi", stopped.View())
	golden(t, "styled/checklist-not-ready.ansi", program(s, setupSample(), nil, nil, nil).View())
	golden(t, "styled/detail-score.ansi", detail(facts(stopped), 3))
}

// Every frame fits 80x24: 24 lines, none wider than 80 cells.
func TestFramesFit80x24(t *testing.T) {
	lipgloss.SetColorProfile(termenv.Ascii)
	s := loadSample(t)
	for _, m := range []tea.Model{program(s, ready(), &s.stopped, nil, nil), program(s, ready(), &s.running, s.job, s.tail),
		program(s, setupSample(), nil, nil, nil), press(program(s, ready(), &s.stopped, nil, nil), "enter")} {
		lines := strings.Split(m.View(), "\n")
		if len(lines) != 24 {
			t.Fatalf("%d lines", len(lines))
		}
		for _, l := range lines {
			if w := lipgloss.Width(l); w > 80 {
				t.Fatalf("line is %d wide: %q", w, l)
			}
		}
	}
}

func TestStepStates(t *testing.T) {
	lipgloss.SetColorProfile(termenv.Ascii)
	s := loadSample(t)
	marks := func(m tea.Model) string {
		out := ""
		for _, st := range steps {
			mk, _ := st.line(facts(m))
			out += mk + " "
		}
		return out
	}
	for name, c := range map[string]struct {
		m      tea.Model
		marks  string
		cursor int
	}{
		"stopped":     {program(s, ready(), &s.stopped, nil, nil), "pass pass crossed run fail ", 2},
		"predeclared": {program(s, ready(), &s.waiting, nil, nil), "pass pass open open open ", 2},
		"no gates":    {program(s, ready(), &s.running, s.job, s.tail), "pass crossed run open open ", 1},
		"not ready":   {program(s, setupSample(), nil, nil, nil), "fail open open open open ", 0},
	} {
		if got := marks(c.m); got != c.marks || c.m.(Model).cursor != c.cursor {
			t.Errorf("%s: marks %q cursor %d", name, got, c.m.(Model).cursor)
		}
	}
	if New(seam.Client{}, jobs.Store{}).mood() != faceSleep {
		t.Fatal("an empty nursery sleeps")
	}
}

func TestKeysAndForms(t *testing.T) {
	lipgloss.SetColorProfile(termenv.Ascii)
	s := loadSample(t)
	m := program(s, ready(), &s.stopped, nil, nil)
	if m.(Model).mood() != faceWorried {
		t.Fatal("a failed gate worries")
	}
	m = press(press(m, "j"), "enter") // step 4, open
	if !m.(Model).open || m.(Model).cursor != 3 {
		t.Fatal("enter opens the step under the cursor")
	}
	m = press(m, "enter") // Score G2a
	if got := m.(Model); got.form != "score" || got.gate != "G2a" {
		t.Fatalf("enter on a gate row opens its score form: %q %q", got.form, got.gate)
	}
	m = press(press(press(m, "x"), "enter"), "q") // typed, not keys, inside the form
	if got := m.(Model); got.form != "score" || got.focus != 1 || got.inputs[0].Value() != "x" || got.inputs[1].Value() != "q" {
		t.Fatalf("form keys: focus %d %q %q", got.focus, got.inputs[0].Value(), got.inputs[1].Value())
	}
	m, _ = m.Update(tea.KeyMsg{Type: tea.KeySpace, Runes: []rune(" ")})
	if v := m.(Model).inputs[1].Value(); v != "q " {
		t.Fatalf("a space is typed into the field, got %q", v)
	}
	m = press(m, "esc")
	if m.(Model).form != "" || !m.(Model).open {
		t.Fatal("esc closes the form and keeps the step open")
	}
	m = press(m, "esc")
	if m.(Model).open {
		t.Fatal("esc goes back to the list")
	}
	if verdictActions(facts(m)) != nil {
		t.Fatal("no adopt row while gates are open")
	}
	scored := s.stopped
	scored.GateTable = []seam.Gate{{ID: "G1", Result: "pass"}}
	if len(verdictActions(Facts{Run: &scored})) != 1 {
		t.Fatal("the adopt row appears when every gate is scored")
	}
}

func TestGPUVerdictLeadsStepOne(t *testing.T) {
	small := seam.Check{ID: "gpu", Detail: "Apple M3: 11.8 GB · too small to train (needs 30 GB)",
		Memory: &seam.Memory{Name: "Apple M3", FreeGB: 11.8, TotalGB: 11.8, NeedGB: 30}}
	busy := seam.Check{ID: "gpu", Fix: "free the GPU: llama-server (pid 2976) holds 27.6 GB",
		Memory: &seam.Memory{Name: "RTX 5090", FreeGB: 3.8, TotalGB: 31.8, NeedGB: 30}}
	none := seam.Check{ID: "gpu", Detail: "no GPU found on this machine"}
	other := seam.Check{ID: "model", Label: "DAYCARE_BASE_GGUF is a file"}
	for _, tc := range []struct {
		gpu        seam.Check
		line, body string
	}{
		{small, "GPU too small: 11.8 of 30 GB · 1 more missing", "not enough GPU memory to train on this machine"},
		{busy, "GPU busy: 3.8 of 30 GB free · 1 more missing", "llama-server (pid 2976) holds 27.6 GB"},
		{none, "no GPU found · 1 more missing", "no GPU found"},
	} {
		f := Facts{Setup: &seam.Setup{Checks: []seam.Check{other, tc.gpu}}}
		if _, line := readyLine(f); line != tc.line {
			t.Errorf("line %q, want %q", line, tc.line)
		}
		body := ansi.Strip(readyBody(f))
		if !strings.Contains(strings.SplitN(body, "\n\n", 2)[0], tc.body) {
			t.Errorf("the GPU block does not lead step 1 with %q:\n%s", tc.body, body)
		}
	}
}

// detail renders the open view the way the 80x24 screen does: 21 rows under the header, note and footer.
func detail(f Facts, i int) string {
	v := viewport.New(0, 0)
	return DetailView(f, i, 0, 80, 21, &v)
}
