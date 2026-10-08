// Package ui owns the screen. Every view here is a pure function of seam facts and a size, so a test can
// render a screen to a string and pin it; model.go wires the same functions to Bubble Tea.
package ui

import (
	"fmt"
	"github.com/charmbracelet/bubbles/viewport"
	"strings"

	"github.com/charmbracelet/lipgloss"

	"github.com/JulianAbeleda/DayCare/tui/internal/seam"
)

// Short literal words. Main lines use the plain word only; full views put the record's own name beside it, muted.
var plainMetric = map[string]string{
	"kl": "drift from stock", "entropy": "answer variety", "finished_length": "answer length",
	"capped_turn_rate": "turns cut at the cap", "mean_reward": "reward",
}

var plainState = map[string]string{
	"predeclared": "waiting to start", "in_progress": "in progress", "complete": "finished",
	"stopped": "stopped by a trigger", "failed": "failed",
}

var stateMark = map[string]string{"predeclared": "wait", "in_progress": "run", "complete": "pass", "stopped": "crossed", "failed": "fail"}

// word is the plain word for a record key, or the key itself when no plain word exists.
func word(table map[string]string, key string) string {
	if plain, ok := table[key]; ok {
		return plain
	}
	return key
}

// named is the plain word with the record's own name beside it, muted: the full views' wording.
func named(table map[string]string, key string) string {
	if plain, ok := table[key]; ok {
		return plain + "  " + stMuted.Render(key)
	}
	return key
}

func num(v *float64) string {
	if v == nil {
		return "-"
	}
	return fmt.Sprintf("%.4g", *v)
}

func count(v *int) string {
	if v == nil {
		return "-"
	}
	return fmt.Sprint(*v)
}

func deref(s *string) string {
	if s == nil {
		return "-"
	}
	return *s
}

func signed(v float64) string { return fmt.Sprintf("%+.1f", v) }

func state(s seam.Summary) string {
	return mark(stateMark[s.State]) + " " + named(plainState, s.State)
}

func updates(s seam.Summary) string {
	if s.UpdatesPlanned == nil || *s.UpdatesPlanned == 0 {
		return fmt.Sprintf("%d updates", s.UpdatesDone)
	}
	return fmt.Sprintf("%s %d of %d", bar(float64(s.UpdatesDone)/float64(*s.UpdatesPlanned), 10), s.UpdatesDone, *s.UpdatesPlanned)
}

func measured(g seam.Gate) string {
	m := g.Measured
	if m.Lo == nil || m.Hi == nil {
		return fmt.Sprintf("count %.0f", m.Diff)
	}
	return fmt.Sprintf("%s [%s, %s]", signed(m.Diff), signed(*m.Lo), signed(*m.Hi))
}

func ruleText(g seam.Gate) string {
	return strings.Replace(g.RuleText, "value", fmt.Sprintf("%.4g", g.Value), 1)
}

func lastLines(lines []string, n int) []string {
	if len(lines) > n {
		return lines[len(lines)-n:]
	}
	return lines
}

// table lays out rows two spaces apart; the first row is the header.
func table(rows [][]string) string {
	widths := []int{}
	for _, row := range rows {
		for i, cell := range row {
			if i >= len(widths) {
				widths = append(widths, 0)
			}
			widths[i] = max(widths[i], lipgloss.Width(cell))
		}
	}
	var b strings.Builder
	for r, row := range rows {
		for i, cell := range row {
			if r == 0 {
				cell = stHeader.Render(cell)
			}
			b.WriteString(cell)
			if i < len(row)-1 {
				b.WriteString(strings.Repeat(" ", widths[i]-lipgloss.Width(cell)) + "  ")
			}
		}
		b.WriteString("\n")
	}
	return b.String()
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

func windowBody(w seam.Window) string {
	var b strings.Builder
	b.WriteString("\n" + stHeader.Render("Stop triggers") + "\n")
	if !w.Enabled {
		return b.String() + stMuted.Render("No readings yet: the run has no record, or triggers were off.") + "\n"
	}
	fmt.Fprintf(&b, "%s\n", stMuted.Render(fmt.Sprintf("last %d stepped updates against the first %d · %d rows", w.Window, w.Base, w.Rows)))
	if !w.Ready {
		b.WriteString(stInfo.Render(glyphWait+" Not enough updates yet for a reading.") + "\n")
	}
	rows := [][]string{{" ", "METRIC", "NEAR LIMIT", "NOW", "LIMIT"}}
	for _, rd := range w.Readings {
		m := mark("pass")
		if rd.Crossed {
			m = mark("crossed")
		}
		rows = append(rows, []string{m, named(plainMetric, rd.Metric), bar(danger(rd), 10), num(rd.Now), stMuted.Render(rd.Side + " " + num(rd.Limit))})
	}
	b.WriteString(table(rows))
	return b.String()
}

func adapterBody(r *seam.Run) string {
	var b strings.Builder
	b.WriteString("\n" + stHeader.Render("Adapter") + "\n")
	switch {
	case r.Adapter.Present:
		fmt.Fprintf(&b, "%s saved %s\n", mark("pass"), stMuted.Render(r.Adapter.Path))
	case r.Adapter.SHA256 != nil:
		fmt.Fprintf(&b, "%s recorded, but not in the folder %s\n", mark("crossed"), stMuted.Render(r.Adapter.Path))
	default:
		b.WriteString(stMuted.Render("None yet.") + "\n")
	}
	if r.Verify != nil {
		if r.Verify.BitExact {
			fmt.Fprintf(&b, "%s reloads bit-exactly  %s\n", mark("pass"), stMuted.Render(fmt.Sprintf("%d rows", r.Verify.Rows)))
		} else {
			fmt.Fprintf(&b, "%s does not reload bit-exactly\n", mark("fail"))
		}
	}
	for _, kv := range [][2]any{{"adapter sha256", r.Adapter.SHA256}, {"revision", r.Revision}, {"model sha256", r.ModelSHA256}} {
		if v := kv[1].(*string); v != nil {
			b.WriteString(stMuted.Render(fmt.Sprintf("%s %s", kv[0], *v)) + "\n")
		}
	}
	return b.String()
}

// --- the checklist and the full view ----------------------------------------------------------------------

func listLine(i int, f Facts, cursor bool) string {
	m, text := steps[i].line(f)
	title := fmt.Sprintf("%d  %-11s", i+1, steps[i].title)
	if cursor {
		return stCursor.Render("▸ ") + mark(m) + " " + stCursor.Render(title) + " " + text
	}
	return "  " + mark(m) + " " + title + " " + text
}

func title(i int, f Facts) string {
	t := fmt.Sprintf("%d  %s", i+1, steps[i].title)
	if f.Run != nil && i > 0 {
		t += " · " + f.Run.ID
	}
	return t
}

// ChecklistView is the main screen: the five steps, then the chosen step's summary cut to fit `height` lines.
// ChecklistView fills the screen: the chosen step's results on top, stretched to the height, and the five steps
// pinned at the bottom where the keys act.
func ChecklistView(f Facts, cursor, width, height int) string {
	lines := make([]string, len(steps))
	for i := range steps {
		lines[i] = truncate(listLine(i, f, i == cursor), width-4)
	}
	list := stBox.Width(width - 2).Render(strings.Join(lines, "\n"))
	hint := steps[cursor].hint
	if f.alive() {
		hint = "x stop · " + hint
	}
	room := max(height-lipgloss.Height(list)-3, 2) // the results box: two borders and its title
	body := fill(strings.Split(steps[cursor].body(f), "\n"), room-1)
	return box(title(cursor, f), strings.Join(body, "\n")+"\n"+stMuted.Render(hint), width, false) + "\n" + list
}

// fill cuts or pads lines to exactly n, ending a cut with "…", so the boxes always span the screen.
func fill(lines []string, n int) []string {
	if len(lines) > n {
		return append(lines[:max(n-1, 0)], stMuted.Render("…"))
	}
	return append(lines, make([]string, n-len(lines))...)
}

// DetailBody is one step's whole result, the part that scrolls on top of the open view.
func DetailBody(f Facts, i int) string { return steps[i].body(f) }

// DetailActions is the bottom of the open view: the step's line, then its action rows or the open form.
func DetailActions(f Facts, i, row, width int) string {
	var b strings.Builder
	_, text := steps[i].line(f)
	b.WriteString(text)
	if f.Form != nil {
		b.WriteString("\n" + strings.TrimRight(formView(*f.Form), "\n"))
	} else if steps[i].actions != nil {
		for j, a := range steps[i].actions(f) {
			if j == row {
				b.WriteString("\n" + stCursor.Render("▸ ") + a.label)
			} else {
				b.WriteString("\n  " + a.label)
			}
		}
	}
	return box("What next", b.String(), width, false)
}

// DetailView is the open view at a given height: results on top in a fixed box, actions at the bottom.
func DetailView(f Facts, i, row, width, height int, scroll *viewport.Model) string {
	actions := DetailActions(f, i, row, width)
	scroll.Width, scroll.Height = width-4, max(height-lipgloss.Height(actions)-3, 1)
	scroll.SetContent(DetailBody(f, i))
	return box(title(i, f), scroll.View(), width, false) + "\n" + actions
}

func formView(form Form) string {
	var b strings.Builder
	b.WriteString(stHeader.Render(form.Title) + "\n")
	for i, label := range form.Labels {
		cur, value := "  ", form.Values[i]
		if i == form.Focus {
			cur, value = stCursor.Render("▸ "), value+stCursor.Render("▏")
		}
		fmt.Fprintf(&b, "%s%-10s %s\n", cur, label, value)
	}
	b.WriteString(stMuted.Render("enter next field, then save · esc cancel") + "\n")
	return b.String()
}
