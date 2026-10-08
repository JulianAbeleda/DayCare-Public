// Package ui owns the screen. Every view here is a pure function of seam data and a width, so a test can
// render a screen to a string and pin it; model.go wires the same functions to Bubble Tea.
package ui

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"

	"github.com/JulianAbeleda/DayCare/tui/internal/jobs"
	"github.com/JulianAbeleda/DayCare/tui/internal/seam"
)

// Plain mode uses short literal words; technical mode uses the record's own names and adds raw values.
var plainMetric = map[string]string{
	"kl": "drift from stock (KL)", "entropy": "answer variety (entropy)", "finished_length": "answer length (tokens)",
	"capped_turn_rate": "turns cut at the cap", "mean_reward": "reward",
}

var plainState = map[string]string{
	"predeclared": "waiting to start", "in_progress": "in progress", "complete": "finished",
	"stopped": "stopped by a trigger", "failed": "failed",
}

var stateMark = map[string]string{"predeclared": "wait", "in_progress": "run", "complete": "pass", "stopped": "crossed", "failed": "fail"}

func num(v *float64) string {
	if v == nil {
		return "-"
	}
	return fmt.Sprintf("%.4g", *v)
}

func signed(v float64) string { return fmt.Sprintf("%+.1f", v) }

func state(s seam.Summary, technical bool) string {
	text := s.State
	if !technical {
		if plain, ok := plainState[s.State]; ok {
			text = plain
		}
	}
	if s.Stopped != nil {
		text += fmt.Sprintf(" at update %d", s.Stopped.Update)
	}
	return mark(stateMark[s.State]) + " " + text
}

func updates(s seam.Summary) string {
	if s.UpdatesPlanned == nil || *s.UpdatesPlanned == 0 {
		return fmt.Sprintf("%d", s.UpdatesDone)
	}
	return fmt.Sprintf("%s %d/%d", bar(float64(s.UpdatesDone)/float64(*s.UpdatesPlanned), 10), s.UpdatesDone, *s.UpdatesPlanned)
}

func verdict(v string) string {
	switch v {
	case "pass", "adopted":
		return stOK.Render(v)
	case "fail":
		return stBad.Render(v)
	case "adopted (exception)":
		return stWarn.Render(v)
	}
	return stMuted.Render(v)
}

func gateCounts(marks []seam.GateMark) string {
	if len(marks) == 0 {
		return stMuted.Render("no gates")
	}
	count := map[string]int{}
	for _, m := range marks {
		count[m.Result]++
	}
	parts := []string{}
	for _, k := range []string{"pass", "fail", "open"} {
		if count[k] > 0 {
			parts = append(parts, fmt.Sprintf("%s %d", mark(k), count[k]))
		}
	}
	return strings.Join(parts, "  ")
}

func gateMarks(marks []seam.GateMark) string {
	parts := make([]string, 0, len(marks))
	for _, m := range marks {
		parts = append(parts, mark(m.Result)+m.ID)
	}
	return strings.Join(parts, " ")
}

// table lays out rows two spaces apart; the first row is the header, `cursor` names the highlighted row.
func table(rows [][]string, cursor int) string {
	widths := []int{}
	for _, row := range rows {
		for i, cell := range row {
			if i >= len(widths) {
				widths = append(widths, 0)
			}
			if w := lipgloss.Width(cell); w > widths[i] {
				widths[i] = w
			}
		}
	}
	var b strings.Builder
	for r, row := range rows {
		for i, cell := range row {
			if r == 0 {
				cell = stHeader.Render(cell)
			}
			b.WriteString(cell + strings.Repeat(" ", widths[i]-lipgloss.Width(cell)))
			if i < len(row)-1 {
				b.WriteString("  ")
			}
		}
		if cursor >= 0 && r == cursor+1 {
			b.WriteString(stCursor.Render("  ◂"))
		}
		b.WriteString("\n")
	}
	return b.String()
}

func mode(technical bool) string {
	if technical {
		return stMuted.Render("technical")
	}
	return stMuted.Render("plain")
}

func cursorMark(i, cursor int) string {
	if i == cursor {
		return stCursor.Render("▸")
	}
	return " "
}

// StatusView: what is missing before a counted run is possible.
func StatusView(s *seam.Setup, cursor int, technical bool, width int) string {
	if s == nil {
		return box("Setup", stMuted.Render("Checking…"), width, true)
	}
	var b strings.Builder
	missing := 0
	for _, c := range s.Checks {
		if !c.OK {
			missing++
		}
	}
	if s.Ready {
		b.WriteString(stOK.Render(glyphPass+" Ready.") + " Everything a run needs is here.\n\n")
	} else {
		fmt.Fprintf(&b, "%s %d things are missing. Each one says how to fix it.\n\n", stWarn.Render(glyphWarn+" Not ready yet."), missing)
	}
	for i, c := range s.Checks {
		m := mark("fail")
		if c.OK {
			m = mark("pass")
		}
		fmt.Fprintf(&b, "%s %s %s\n", cursorMark(i, cursor), m, c.Label)
		if technical || !c.OK {
			fmt.Fprintf(&b, "    %s %s\n", stMuted.Render("now"), stMuted.Render(c.Detail))
		}
		if !c.OK && c.Fix != "" {
			fmt.Fprintf(&b, "    %s %s\n", stInfo.Render("fix"), c.Fix)
		}
		if !c.OK && c.Generate != nil {
			fmt.Fprintf(&b, "    %s\n", stAccent.Render("press g to generate the "+*c.Generate))
		}
	}
	if technical {
		fmt.Fprintf(&b, "\n%s %s\n%s %s\n%s %s\n", stMuted.Render("python"), s.Python, stMuted.Render("repo  "), s.Repo, stMuted.Render("runs  "), s.Root)
	}
	return box("Setup", b.String(), width, false)
}

// RunsView: one row per run folder.
func RunsView(r *seam.Runs, cursor int, technical bool, width int) string {
	if r == nil {
		return box("Runs", stMuted.Render("Reading the runs folder…"), width, true)
	}
	if len(r.Runs) == 0 {
		return box("Runs", fmt.Sprintf("No runs yet. The nursery is quiet.\n%s", stMuted.Render(r.Root)), width, true)
	}
	header := []string{"RUN", "STATE", "UPDATES", "VERDICT", "GATES"}
	if technical {
		header = append(header, "LAST kl / entropy / reward")
	}
	rows := [][]string{header}
	for _, s := range r.Runs {
		gates := gateCounts(s.Gates)
		if technical {
			gates = gateMarks(s.Gates)
			if gates == "" {
				gates = "-"
			}
		}
		row := []string{s.ID, state(s, technical), updates(s), verdict(s.Verdict), gates}
		if technical {
			row = append(row, stMuted.Render(fmt.Sprintf("%s / %s / %s", num(s.Last.KL), num(s.Last.Entropy), num(s.Last.MeanReward))))
		}
		rows = append(rows, row)
	}
	return box("Runs", table(rows, cursor), width, false)
}

func measured(g seam.Gate) string {
	m := g.Measured
	if m == nil {
		return ""
	}
	if m.Lo == nil || m.Hi == nil {
		return fmt.Sprintf("count %.0f", m.Diff)
	}
	return fmt.Sprintf("%s [%s, %s]", signed(m.Diff), signed(*m.Lo), signed(*m.Hi))
}

func ruleText(g seam.Gate, technical bool) string {
	if technical {
		return fmt.Sprintf("%s %.4g", g.Rule, g.Value)
	}
	return strings.Replace(g.RuleText, "value", fmt.Sprintf("%.4g", g.Value), 1)
}

// danger is how close a reading sits to its limit: 1 is at the limit, beyond it the bar is full.
func danger(rd seam.Reading) float64 {
	if rd.Now == nil || rd.Limit == nil || *rd.Limit == 0 || *rd.Now == 0 {
		return 0
	}
	if rd.Side == ">" {
		return *rd.Now / *rd.Limit
	}
	return *rd.Limit / *rd.Now
}

// RunView: one run, predictions against results and the stop-trigger window.
func RunView(r *seam.Run, technical bool, width int) string {
	if r == nil {
		return box("Run", stMuted.Render("Reading the run…"), width, true)
	}
	var head strings.Builder
	fmt.Fprintf(&head, "%s   %s   %s %s\n", state(r.Summary, technical), updates(r.Summary), stMuted.Render("verdict"), verdict(r.Verdict))
	if r.Protocol != nil {
		fmt.Fprintf(&head, "%s research/%s\n", stMuted.Render("record"), *r.Protocol)
	}
	if r.Stopped != nil {
		fmt.Fprintf(&head, "%s %s\n", stWarn.Render(glyphWarn+" stopped:"), r.Stopped.Reason)
	}
	if r.Adoption != nil {
		fmt.Fprintf(&head, "%s by %s: %s\n", stWarn.Render("adopted as an exception"), r.Adoption.By, r.Adoption.Exception)
	}
	if r.Hypothesis != nil && *r.Hypothesis != "" {
		fmt.Fprintf(&head, "%s %s\n", stMuted.Render("hypothesis"), *r.Hypothesis)
	}
	out := []string{box(r.ID, head.String(), width, false)}

	var gates strings.Builder
	if len(r.GateTable) == 0 {
		gates.WriteString(stMuted.Render("No predeclaration in this run folder.") + "\n")
	} else {
		rows := [][]string{{" ", "ID", "GATE", "MEASURED", "RULE", "PREDICTION"}}
		for _, g := range r.GateTable {
			rows = append(rows, []string{mark(g.Result), g.ID, g.Name, measured(g), stMuted.Render(ruleText(g, technical)), stMuted.Render(g.Prediction)})
		}
		gates.WriteString(table(rows, -1))
	}
	out = append(out, box("Gates: predicted vs measured", gates.String(), width, false))

	var win strings.Builder
	w := r.Window
	if !w.Enabled {
		win.WriteString(stMuted.Render("No trigger readings yet: the run has no record, or triggers were off.") + "\n")
	} else {
		fmt.Fprintf(&win, "%s\n", stMuted.Render(fmt.Sprintf("last %d stepped updates against the first %d · %d rows · from the %s", w.Window, w.Base, w.Rows, w.Source)))
		if !w.Ready {
			win.WriteString(stInfo.Render(glyphWait+" Not enough updates yet for a reading.") + "\n")
		}
		rows := [][]string{{" ", "METRIC", "NEAR LIMIT", "NOW", "BASE", "LIMIT"}}
		for _, rd := range w.Readings {
			name, m := rd.Metric, mark("pass")
			if !technical {
				name = plainMetric[rd.Metric]
			}
			if rd.Crossed {
				m = mark("crossed")
			}
			rows = append(rows, []string{m, name, bar(danger(rd), 12), num(rd.Now), stMuted.Render(num(rd.Base)), stMuted.Render(rd.Side + " " + num(rd.Limit))})
		}
		win.WriteString(table(rows, -1))
		if w.Tripped != nil {
			fmt.Fprintf(&win, "%s update %d: %s\n", stWarn.Render(glyphWarn+" tripped at"), w.Tripped.Update, w.Tripped.Reason)
		}
	}
	out = append(out, box("Stop-trigger window", win.String(), width, false))

	var ad strings.Builder
	switch {
	case r.Adapter.Present:
		fmt.Fprintf(&ad, "%s saved: %s (gguf: %t)\n", mark("pass"), r.Adapter.Path, r.Adapter.GGUF)
	case r.Adapter.SHA256 != nil:
		fmt.Fprintf(&ad, "%s recorded but not at %s\n", mark("crossed"), r.Adapter.Path)
	default:
		ad.WriteString(stMuted.Render("None yet.") + "\n")
	}
	if r.Verify != nil {
		if r.Verify.BitExact {
			fmt.Fprintf(&ad, "%s reloads bit-exactly\n", mark("pass"))
		} else {
			fmt.Fprintf(&ad, "%s does not reload bit-exactly\n", mark("fail"))
		}
	}
	if technical {
		for _, kv := range [][2]*string{{ptr("adapter sha256"), r.Adapter.SHA256}, {ptr("daycare revision"), r.Revision}, {ptr("model sha256"), r.ModelSHA256}} {
			if kv[1] != nil {
				fmt.Fprintf(&ad, "%s %s\n", stMuted.Render(*kv[0]), *kv[1])
			}
		}
	}
	out = append(out, box("Adapter", ad.String(), width, true))

	if technical && len(r.Recipe) > 0 {
		out = append(out, box("Recipe", "rloo_posttool train --root <run> "+strings.Join(r.Recipe, " "), width, true))
	}
	if technical && len(r.Updates) > 0 {
		rows := [][]string{{"STEP", "KL", "ENTROPY", "REWARD", "LENGTH", "CAPPED", "SKIPPED"}}
		for _, u := range r.Updates {
			step := "-"
			if u.Step != nil {
				step = fmt.Sprintf("%d", *u.Step)
			}
			rows = append(rows, []string{step, num(u.KL), num(u.Entropy), num(u.MeanReward), num(u.FinishedLength), num(u.CappedTurnRate), fmt.Sprintf("%t", u.Skipped)})
		}
		out = append(out, box("Last updates", table(rows, -1), width, true))
	}
	return strings.Join(out, "\n")
}

func ptr(s string) *string { return &s }

// AdaptersView: the runs that saved an adapter, or recorded one.
func AdaptersView(r *seam.Runs, technical bool, width int) string {
	if r == nil {
		return box("Adapters", stMuted.Render("Reading the runs folder…"), width, true)
	}
	rows := [][]string{{"RUN", "STATE", "VERDICT", "ADAPTER", "GGUF"}}
	if technical {
		rows[0] = append(rows[0], "SHA256")
	}
	for _, s := range r.Runs {
		if !s.Adapter.Present && s.Adapter.SHA256 == nil {
			continue
		}
		adapter := mark("crossed") + " missing from the run folder"
		if s.Adapter.Present {
			adapter = mark("pass") + " saved"
		}
		row := []string{s.ID, state(s, technical), verdict(s.Verdict), adapter, fmt.Sprintf("%t", s.Adapter.GGUF)}
		if technical {
			sha := "-"
			if s.Adapter.SHA256 != nil {
				sha = *s.Adapter.SHA256
			}
			row = append(row, stMuted.Render(sha))
		}
		rows = append(rows, row)
	}
	if len(rows) == 1 {
		return box("Adapters", "No run has saved an adapter yet.", width, true)
	}
	return box("Adapters", table(rows, -1), width, false)
}

// JobsView: the process the TUI started for a run, and the tail of its output.
func JobsView(id string, job *jobs.Job, tail []string, technical bool, width int) string {
	if id == "" {
		return box("Job", "No run selected. Open a run and press s to start it.", width, true)
	}
	if job == nil {
		return box("Job", fmt.Sprintf("No job was started for %s from this machine.", id), width, true)
	}
	var b strings.Builder
	running := mark("pass") + " finished"
	if job.Alive {
		running = mark("run") + " running"
	}
	fmt.Fprintf(&b, "%s   %s\n", running, stMuted.Render(fmt.Sprintf("pid %d · started %s", job.PID, job.StartedAt)))
	if technical {
		fmt.Fprintf(&b, "%s %s\n%s %s\n", stMuted.Render("log "), job.LogPath, stMuted.Render("argv"), strings.Join(job.Argv, " "))
	}
	b.WriteString("\n")
	for _, line := range tail {
		b.WriteString(line + "\n")
	}
	return box(id, b.String(), width, !job.Alive)
}
