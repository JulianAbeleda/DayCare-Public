package ui

import (
	"fmt"
	"strings"

	"github.com/JulianAbeleda/DayCare/tui/internal/jobs"
	"github.com/JulianAbeleda/DayCare/tui/internal/seam"
)

// Facts is everything the checklist reads. Every step's state is a function of these seam facts, never of
// what the user clicked, so reopening the screen mid run lands on the same marks.
type Facts struct {
	Setup *seam.Setup
	Runs  *seam.Runs
	Run   *seam.Run
	Job   *jobs.Job
	Tail  []string
	Form  *Form
	Spin  string
}

// Form is a typed form inside a step: the score of one gate, or the adoption record.
type Form struct {
	Title  string
	Labels []string
	Values []string
	Focus  int
}

// action is one row inside a step's full view; enter on it does `do` with `arg`.
type action struct{ label, do, arg string }

// step is one row of the checklist: the playbook's sections 1 to 5. line feeds the list; body feeds the
// summary box (cut to fit) and the full view, so the two cannot disagree.
type step struct {
	title   string
	hint    string
	line    func(Facts) (mark, text string)
	body    func(Facts) string
	actions func(Facts) []action
}

var steps = []step{
	{"Ready", "enter: each missing item and its fix", readyLine, readyBody, readyActions},
	{"Predeclare", "enter: pick a run, or a new one from the run5 recipe", predeclareLine, predeclareBody, predeclareActions},
	{"Train", "enter: start, stop, the trigger window, the log", trainLine, trainBody, trainActions},
	{"Score", "enter: score a gate", scoreLine, scoreBody, scoreActions},
	{"Verdict", "enter: the rule, and adopt when every gate is scored", verdictLine, verdictBody, verdictActions},
}

// firstOpen is the step the cursor starts on: the first one not done.
func firstOpen(f Facts) int {
	for i, s := range steps {
		if m, _ := s.line(f); m != "pass" {
			return i
		}
	}
	return len(steps) - 1
}

func (f Facts) alive() bool { return f.Job != nil && f.Job.Alive }

func (f Facts) gates() []seam.Gate {
	if f.Run == nil {
		return nil
	}
	return f.Run.GateTable
}

func (f Facts) tally() map[string]int {
	n := map[string]int{}
	for _, g := range f.gates() {
		n[g.Result]++
	}
	return n
}

// --- 1 Ready -------------------------------------------------------------------------------------------------

func missing(s *seam.Setup) []seam.Check {
	out := []seam.Check{}
	for _, c := range s.Checks {
		if !c.OK {
			out = append(out, c)
		}
	}
	return out
}

// gpuCheck is the setup's GPU memory check, or nil when the setup has none.
func gpuCheck(s *seam.Setup) *seam.Check {
	if s == nil {
		return nil
	}
	for i := range s.Checks {
		if s.Checks[i].ID == "gpu" {
			return &s.Checks[i]
		}
	}
	return nil
}

// gpuVerdict is the short sentence that decides whether this machine can train at all.
func gpuVerdict(c *seam.Check) string {
	m := c.Memory
	switch {
	case m == nil && strings.HasPrefix(c.Detail, "no GPU"):
		return "no GPU found"
	case m == nil:
		return "the GPU could not be read"
	case m.TotalGB < m.NeedGB:
		return fmt.Sprintf("GPU too small: %.1f of %g GB", m.TotalGB, m.NeedGB)
	case m.FreeGB < m.NeedGB:
		return fmt.Sprintf("GPU busy: %.1f of %g GB free", m.FreeGB, m.NeedGB)
	}
	return fmt.Sprintf("GPU ok: %.1f GB free", m.FreeGB)
}

// gpuBlock is the GPU memory against what training needs, drawn first in step 1 because it decides the rest.
func gpuBlock(c *seam.Check) string {
	m := c.Memory
	if m == nil {
		return fmt.Sprintf("%s %s\n    %s", mark("fail"), stBad.Render(gpuVerdict(c)), c.Fix)
	}
	have := m.FreeGB
	if m.TotalGB < m.NeedGB {
		have = m.TotalGB
	}
	verdict := mark("pass") + " " + stOK.Render("enough GPU memory to train")
	switch {
	case m.TotalGB < m.NeedGB:
		verdict = mark("fail") + " " + stBad.Render("not enough GPU memory to train on this machine")
	case !c.OK:
		verdict = mark("fail") + " " + stBad.Render("GPU memory is in use") + "  " + c.Fix
	}
	return fmt.Sprintf("%s  %s · %.1f GB of memory\n     %s %s\n     %s",
		stHeader.Render("GPU"), m.Name, m.TotalGB, bar(have/m.NeedGB, 30),
		stMuted.Render(fmt.Sprintf("%.1f of %g GB needed", have, m.NeedGB)), verdict)
}

func readyLine(f Facts) (string, string) {
	switch {
	case f.Setup == nil:
		return "open", "checking…"
	case f.Setup.Ready:
		return "pass", "everything a run needs is here"
	}
	if g := gpuCheck(f.Setup); g != nil && !g.OK {
		text := gpuVerdict(g)
		if n := len(missing(f.Setup)) - 1; n > 0 {
			text += fmt.Sprintf(" · %d more missing", n)
		}
		return "fail", text
	}
	return "fail", fmt.Sprintf("%d things missing · enter shows each fix", len(missing(f.Setup)))
}

func readyBody(f Facts) string {
	if f.Setup == nil {
		return stMuted.Render("Checking the machine…")
	}
	var b strings.Builder
	if g := gpuCheck(f.Setup); g != nil {
		b.WriteString(gpuBlock(g) + "\n\n")
	}
	for _, c := range missing(f.Setup) {
		if c.ID == "gpu" {
			continue
		}
		fmt.Fprintf(&b, "%s %s\n    %s %s\n", mark("fail"), c.Label, stMuted.Render("now"), stMuted.Render(c.Detail))
		if c.Fix != "" {
			fmt.Fprintf(&b, "    %s %s\n", stInfo.Render("fix"), c.Fix)
		}
	}
	for _, c := range f.Setup.Checks {
		if c.OK && c.ID != "gpu" {
			fmt.Fprintf(&b, "%s %s  %s\n", mark("pass"), c.Label, stMuted.Render(c.Detail))
		}
	}
	return strings.TrimRight(b.String(), "\n")
}

func readyActions(f Facts) []action {
	out := []action{}
	if f.Setup != nil {
		for _, c := range missing(f.Setup) {
			if c.Generate != nil {
				out = append(out, action{"Generate the " + *c.Generate, "generate", *c.Generate})
			}
		}
	}
	return out
}

// --- 2 Predeclare --------------------------------------------------------------------------------------------

func predeclareLine(f Facts) (string, string) {
	switch {
	case f.Run == nil && f.Setup != nil && !f.Setup.Ready:
		return "open", "needs step 1"
	case f.Run == nil:
		return "open", "no run yet · enter to pick or make one"
	case len(f.Run.GateTable) == 0:
		return "crossed", f.Run.ID + " · no predeclaration in this folder"
	}
	return "pass", fmt.Sprintf("%s · %d gates", f.Run.ID, len(f.Run.GateTable))
}

func predeclareBody(f Facts) string {
	r := f.Run
	if r == nil {
		return stMuted.Render("No run picked. The checklist follows the run you open here.")
	}
	var b strings.Builder
	if r.Hypothesis != nil && *r.Hypothesis != "" {
		fmt.Fprintf(&b, "%s %s\n", stInfo.Render("hypothesis"), *r.Hypothesis)
	}
	if r.Protocol != nil {
		fmt.Fprintf(&b, "%s research/%s\n", stMuted.Render("record"), *r.Protocol)
	}
	if r.PredeclaredAt != nil {
		fmt.Fprintf(&b, "%s %s\n", stMuted.Render("predeclared"), *r.PredeclaredAt)
	}
	if len(r.Recipe) > 0 {
		fmt.Fprintf(&b, "%s\n", stMuted.Render("recipe rloo_posttool train "+strings.Join(r.Recipe, " ")))
	}
	if b.Len() == 0 {
		return stMuted.Render("This folder has no predeclaration.xml. The gates were never written down.")
	}
	return strings.TrimRight(b.String(), "\n")
}

func predeclareActions(f Facts) []action {
	out := []action{{"New run from the run5 recipe", "generate", "predeclaration"}}
	if f.Runs != nil {
		for _, s := range f.Runs.Runs {
			label := fmt.Sprintf("Open run %s  %s", s.ID, word(plainState, s.State))
			if f.Run != nil && s.ID == f.Run.ID {
				label += stMuted.Render("  shown")
			}
			out = append(out, action{label, "run", s.ID})
		}
	}
	return out
}

// --- 3 Train -------------------------------------------------------------------------------------------------

func trainLine(f Facts) (string, string) {
	r := f.Run
	if r == nil || len(r.GateTable) == 0 && r.State == "predeclared" {
		return "open", "needs step 2"
	}
	progress := updates(r.Summary)
	switch {
	case f.alive() || r.State == "in_progress":
		return "run", progress
	case r.State == "predeclared" && gpuCheck(f.Setup) != nil && !gpuCheck(f.Setup).OK:
		return "fail", "cannot start here · " + gpuVerdict(gpuCheck(f.Setup))
	case r.State == "predeclared":
		return "open", "waiting to start · enter to start"
	case r.State == "complete":
		return "pass", progress + " · finished"
	case r.State == "stopped":
		return "crossed", progress + " · stopped: " + crossed(r.Window)
	}
	return "fail", progress + " · " + word(plainState, r.State)
}

func trainBody(f Facts) string {
	r := f.Run
	if r == nil {
		return stMuted.Render("No run picked.")
	}
	var b strings.Builder
	fmt.Fprintf(&b, "%s %s\n", state(r.Summary), updates(r.Summary))
	if r.Stopped != nil {
		fmt.Fprintf(&b, "%s %s\n", stWarn.Render(glyphWarn+" stopped:"), r.Stopped.Reason)
	}
	b.WriteString(windowBody(r.Window))
	b.WriteString(adapterBody(r))
	if len(r.Updates) > 0 {
		rows := [][]string{{"STEP", "KL", "ENTROPY", "REWARD", "LENGTH"}}
		for _, u := range r.Updates {
			rows = append(rows, []string{count(u.Step), num(u.KL), num(u.Entropy), num(u.MeanReward), num(u.FinishedLength)})
		}
		b.WriteString("\n" + stHeader.Render("Last updates") + "\n" + table(rows))
	}
	if f.Job != nil {
		run := mark("pass") + " finished"
		if f.alive() {
			run = mark("run") + " running " + f.Spin
		}
		fmt.Fprintf(&b, "\n%s  %s\n%s", run, stMuted.Render(fmt.Sprintf("pid %d · log %s", f.Job.PID, f.Job.LogPath)),
			stMuted.Render(strings.Join(lastLines(f.Tail, 12), "\n")))
	}
	return strings.TrimRight(b.String(), "\n")
}

func trainActions(f Facts) []action {
	switch {
	case f.alive():
		return []action{{"Stop the run (x)", "stop", ""}}
	case f.Run != nil && f.Run.State == "predeclared" && len(f.Run.GateTable) > 0:
		if g := gpuCheck(f.Setup); g != nil && !g.OK {
			return nil // the seam would refuse; step 1 says why
		}
		return []action{{"Start training", "start", ""}}
	}
	return nil
}

// --- 4 Score -------------------------------------------------------------------------------------------------

func scoreLine(f Facts) (string, string) {
	gates := f.gates()
	if len(gates) == 0 {
		return "open", "needs step 2"
	}
	n := f.tally()
	text := fmt.Sprintf("%d pass · %d fail · %d open", n["pass"], n["fail"], n["open"])
	switch {
	case n["open"] == 0:
		return "pass", text
	case n["open"] < len(gates):
		return "run", text
	}
	return "open", text
}

func scoreBody(f Facts) string {
	gates := f.gates()
	if len(gates) == 0 {
		return stMuted.Render("No gates. They come from the predeclaration in step 2.")
	}
	rows := [][]string{{" ", "GATE", "", "MEASURED"}}
	for _, g := range gates {
		result := stMuted.Render("not scored")
		if g.Measured != nil {
			result = measured(g)
		}
		rows = append(rows, []string{mark(g.Result), g.ID, g.Name, result})
	}
	var b strings.Builder
	b.WriteString(table(rows) + "\n" + stHeader.Render("Rules and predictions") + "\n")
	for _, g := range gates {
		fmt.Fprintf(&b, "%-4s %s\n     %s\n", g.ID, ruleText(g), stMuted.Render(g.Prediction))
	}
	return strings.TrimRight(b.String(), "\n")
}

func scoreActions(f Facts) []action {
	out := []action{}
	for _, g := range f.gates() {
		if g.Result == "open" {
			out = append(out, action{fmt.Sprintf("Score %s  %s", g.ID, g.Name), "score", g.ID})
		}
	}
	return out
}

// --- 5 Verdict -----------------------------------------------------------------------------------------------

func failed(f Facts) []string {
	ids := []string{}
	for _, g := range f.gates() {
		if g.Result == "fail" {
			ids = append(ids, g.ID)
		}
	}
	return ids
}

func verdictLine(f Facts) (string, string) {
	if f.Run == nil {
		return "open", "needs step 4"
	}
	switch f.Run.Verdict {
	case "fail":
		return "fail", "fail (" + strings.Join(failed(f), ", ") + ") · a failed gate ends the experiment"
	case "pass":
		return "pass", "pass · every gate passed"
	case "adopted":
		return "pass", "adopted · every gate passed"
	case "adopted (exception)":
		return "crossed", "adopted as an exception by " + f.Run.Adoption.By
	}
	return "open", "needs step 4"
}

func verdictBody(f Facts) string {
	if f.Run == nil || len(f.Run.GateTable) == 0 {
		return stMuted.Render("No gates, so no verdict.")
	}
	var b strings.Builder
	fmt.Fprintf(&b, "Verdict: %s\n", f.Run.Verdict)
	b.WriteString(stMuted.Render("Rule: one failed gate fails the run. An open gate keeps it open.") + "\n")
	if ids := failed(f); len(ids) > 0 {
		fmt.Fprintf(&b, "%s %s. No rescue runs.\n", stBad.Render(glyphFail+" failed:"), strings.Join(ids, ", "))
	}
	if a := f.Run.Adoption; a != nil {
		fmt.Fprintf(&b, "%s by %s at %s: %s\n", stWarn.Render("adopted"), a.By, a.At, a.Exception)
	}
	return strings.TrimRight(b.String(), "\n")
}

func verdictActions(f Facts) []action {
	if f.Run == nil || len(f.Run.GateTable) == 0 || f.Run.Adoption != nil || f.tally()["open"] > 0 {
		return nil
	}
	return []action{{"Adopt the adapter (writes who and why into the predeclaration)", "adopt", ""}}
}

// crossed names the trigger readings past their limit in plain words; the full reason is in the full view.
func crossed(w seam.Window) string {
	names := []string{}
	for _, rd := range w.Readings {
		if rd.Crossed {
			names = append(names, word(plainMetric, rd.Metric))
		}
	}
	if len(names) == 0 {
		return "see why in the full view"
	}
	return strings.Join(names, ", ") + " past its limit"
}
