package ui

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/JulianAbeleda/DayCare/tui/internal/jobs"
	"github.com/JulianAbeleda/DayCare/tui/internal/seam"
)

// Five keys. Everything else is a row inside a step's full view.
var (
	keyUp    = key.NewBinding(key.WithKeys("up", "k"))
	keyDown  = key.NewBinding(key.WithKeys("down", "j"))
	keyEnter = key.NewBinding(key.WithKeys("enter"))
	keyBack  = key.NewBinding(key.WithKeys("esc"))
	keyStop  = key.NewBinding(key.WithKeys("x"))
	keyQuit  = key.NewBinding(key.WithKeys("q", "ctrl+c"))
)

// Model is the checklist: seam facts, the step cursor, whether that step's full view is open, and a form.
type Model struct {
	client  seam.Client
	store   jobs.Store
	f       Facts
	runSet  bool // the user picked a run; the checklist no longer follows the default one
	cursor  int
	moved   bool // the user moved; the cursor no longer follows the first open step
	open    bool
	row     int
	ticking bool
	form    string // "score" or "adopt" while a form is open
	gate    string
	inputs  []textinput.Model
	focus   int
	width   int
	height  int
	note    string
	view    viewport.Model
	spin    spinner.Model
}

type setupMsg struct {
	setup *seam.Setup
	err   error
}
type runsMsg struct {
	runs *seam.Runs
	err  error
}
type runMsg struct {
	run *seam.Run
	err error
}
type jobMsg struct {
	job  *jobs.Job
	tail []string
}
type noteMsg string
type tickMsg time.Time

func New(client seam.Client, store jobs.Store) Model {
	return Model{client: client, store: store, width: 80, height: 24, view: viewport.New(80, 21),
		spin: spinner.New(spinner.WithSpinner(spinner.MiniDot), spinner.WithStyle(stAccent))}
}

// Start runs the program on the terminal.
func Start(client seam.Client, store jobs.Store) error {
	_, err := tea.NewProgram(New(client, store), tea.WithAltScreen()).Run()
	return err
}

func (m Model) Init() tea.Cmd { return tea.Batch(m.loadSetup(), m.loadRuns(), m.spin.Tick) }

func (m Model) loadSetup() tea.Cmd {
	return func() tea.Msg { s, _, err := m.client.Setup(); return setupMsg{s, err} }
}

func (m Model) loadRuns() tea.Cmd {
	return func() tea.Msg { r, _, err := m.client.List(); return runsMsg{r, err} }
}

func (m Model) loadRun(id string) tea.Cmd {
	return func() tea.Msg { r, _, err := m.client.Show(id); return runMsg{r, err} }
}

func (m Model) loadJob(id string) tea.Cmd {
	return func() tea.Msg {
		job, err := m.store.Status(id)
		if err != nil {
			return jobMsg{nil, nil}
		}
		lines, _ := m.store.Tail(id, 200)
		return jobMsg{&job, lines}
	}
}

func tick() tea.Cmd {
	return tea.Tick(2*time.Second, func(t time.Time) tea.Msg { return tickMsg(t) })
}

func (m Model) runID() string {
	if m.f.Run != nil {
		return m.f.Run.ID
	}
	return ""
}

// defaultRun is the run the checklist is about when the user picked none: one in progress, else the last.
func (m Model) defaultRun() string {
	if m.f.Runs == nil || len(m.f.Runs.Runs) == 0 {
		return ""
	}
	for _, r := range m.f.Runs.Runs {
		if r.State == "in_progress" {
			return r.ID
		}
	}
	return m.f.Runs.Runs[len(m.f.Runs.Runs)-1].ID
}

func (m Model) generate(thing string) tea.Cmd {
	setup, runs := m.f.Setup, m.f.Runs
	return func() tea.Msg {
		switch thing {
		case "tasks":
			name := seam.NextName(runs, "posttool-tasks")
			if _, err := m.client.FreezeTasks(name); err != nil {
				return noteMsg("Generate failed: " + err.Error())
			}
			return noteMsg("Generated tasks under " + name + ". The states need the GPU harvest; see the fix text.")
		case "predeclaration":
			p := predeclarationFromSetup(setup, seam.NextName(runs, "posttool-predeclared"))
			if _, err := m.client.Predeclare(p); err != nil {
				return noteMsg("Generate failed: " + err.Error())
			}
			return noteMsg("Predeclared " + p.ID + " from template run5. Write the research record before the first update.")
		}
		return noteMsg("Unknown generator " + thing)
	}
}

// predeclarationFromSetup fills the template's recipe with the states file the setup check found and the
// envelope from the environment; a missing one stays missing, and step 1 keeps saying so.
func predeclarationFromSetup(s *seam.Setup, name string) seam.Predeclaration {
	p := seam.Predeclaration{ID: name, Protocol: "rloo-" + name + ".md", Template: "run5", Envelope: os.Getenv("DAYCARE_ENVELOPE")}
	if s != nil {
		for _, c := range s.Checks {
			if c.ID == "states" && c.OK {
				p.States = c.Detail
			}
		}
	}
	return p
}

func (m Model) startRun() tea.Cmd {
	run := m.f.Run
	return func() tea.Msg {
		dir, err := m.client.RunDir(run.ID)
		if err != nil {
			return noteMsg(err.Error())
		}
		if _, err := m.store.Start(run.ID, m.client.Repo, m.client.TrainArgv(dir, run.Recipe)); err != nil {
			return noteMsg("Start failed: " + err.Error())
		}
		return noteMsg("Started " + run.ID + ".")
	}
}

func (m Model) stopRun() tea.Cmd {
	id, alive := m.runID(), m.f.alive()
	return func() tea.Msg {
		if !alive {
			return noteMsg("No run is going from here.")
		}
		if _, err := m.store.Stop(id); err != nil {
			return noteMsg("Stop failed: " + err.Error())
		}
		return noteMsg("Sent SIGTERM to " + id + ". The loop keeps its last checkpoint.")
	}
}

// openForm starts a typed form; the labels are the seam command's own arguments.
func (m *Model) openForm(kind, gate string, labels ...string) tea.Cmd {
	m.form, m.gate, m.focus, m.inputs = kind, gate, 0, make([]textinput.Model, len(labels))
	for i := range labels {
		m.inputs[i] = textinput.New()
		m.inputs[i].Prompt = ""
	}
	return m.inputs[0].Focus()
}

var formLabels = map[string][]string{"score": {"diff", "lo", "hi", "note"}, "adopt": {"by", "exception"}}

func (m Model) formFacts() *Form {
	if m.form == "" {
		return nil
	}
	values := make([]string, len(m.inputs))
	for i, in := range m.inputs {
		values[i] = in.Value()
	}
	title := "Score " + m.gate + " · lo and hi are the interval; both empty for a count"
	if m.form == "adopt" {
		title = "Adopt: who decides, and the exception with its evidence"
	}
	return &Form{Title: title, Labels: formLabels[m.form], Values: values, Focus: m.focus}
}

// submit sends the form to the seam. The seam owns every rule: it refuses a rescored gate or an early adoption.
func (m Model) submit() tea.Cmd {
	id, gate, kind := m.runID(), m.gate, m.form
	v := make([]string, len(m.inputs))
	for i, in := range m.inputs {
		v[i] = strings.TrimSpace(in.Value())
	}
	return func() tea.Msg {
		var err error
		if kind == "adopt" {
			_, err = m.client.Adopt(id, v[0], v[1])
		} else {
			diff, perr := strconv.ParseFloat(v[0], 64)
			if perr != nil {
				return noteMsg("diff must be a number.")
			}
			var lo, hi *float64
			if v[1] != "" || v[2] != "" {
				l, e1 := strconv.ParseFloat(v[1], 64)
				h, e2 := strconv.ParseFloat(v[2], 64)
				if e1 != nil || e2 != nil {
					return noteMsg("lo and hi must both be numbers, or both empty.")
				}
				lo, hi = &l, &h
			}
			_, err = m.client.Score(id, gate, diff, lo, hi, v[3])
		}
		if err != nil {
			return noteMsg("The seam refused: " + err.Error())
		}
		return noteMsg("Saved " + kind + " for " + id + ".")
	}
}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	next, cmd := m.update(msg)
	next.f.Form = next.formFacts()
	if !next.moved {
		next.cursor = firstOpen(next.f)
	}
	return next, cmd
}

func (m Model) update(msg tea.Msg) (Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
	case spinner.TickMsg:
		var cmd tea.Cmd
		m.spin, cmd = m.spin.Update(msg)
		m.f.Spin = m.spin.View()
		return m, cmd
	case setupMsg:
		m.f.Setup = msg.setup
		if msg.err != nil {
			m.f.Setup, m.note = nil, "The setup check failed: "+msg.err.Error()
		}
	case runsMsg:
		m.f.Runs = msg.runs
		if msg.err != nil {
			m.note = "The runs folder could not be read: " + msg.err.Error()
			return m, nil
		}
		if n := len(msg.runs.Unreadable); n > 0 {
			ids := make([]string, n)
			for i, u := range msg.runs.Unreadable {
				ids[i] = u.ID
			}
			m.note = fmt.Sprintf("⚠ %d run(s) could not be read: %s (%s)", n, strings.Join(ids, ", "), msg.runs.Unreadable[0].Error)
		}
		if id := m.defaultRun(); !m.runSet && id != "" && id != m.runID() {
			return m, m.loadRun(id)
		}
	case runMsg:
		if msg.err != nil {
			m.note = "The run could not be read: " + msg.err.Error()
			return m, nil
		}
		changed := m.runID() != msg.run.ID
		m.f.Run = msg.run
		if changed {
			m.f.Job, m.f.Tail = nil, nil
			return m, m.loadJob(msg.run.ID)
		}
	case jobMsg:
		m.f.Job, m.f.Tail = msg.job, msg.tail
		if m.f.alive() && !m.ticking {
			m.ticking = true
			return m, tick()
		}
	case noteMsg:
		m.note = string(msg)
		cmds := []tea.Cmd{m.loadSetup(), m.loadRuns()}
		if id := m.runID(); id != "" {
			cmds = append(cmds, m.loadRun(id), m.loadJob(id))
		}
		return m, tea.Batch(cmds...)
	case tickMsg:
		id := m.runID()
		if m.f.alive() {
			return m, tea.Batch(m.loadJob(id), m.loadRun(id), tick())
		}
		m.ticking = false
		return m, tea.Batch(m.loadRuns(), m.loadRun(id))
	case tea.KeyMsg:
		if m.form != "" {
			return m.formKey(msg)
		}
		return m.key(msg)
	}
	return m, nil
}

func (m Model) formKey(msg tea.KeyMsg) (Model, tea.Cmd) {
	switch msg.Type {
	case tea.KeyEsc:
		m.form = ""
		return m, nil
	case tea.KeyEnter, tea.KeyDown, tea.KeyUp:
		if msg.Type == tea.KeyEnter && m.focus == len(m.inputs)-1 {
			cmd := m.submit()
			m.form = ""
			return m, cmd
		}
		m.inputs[m.focus].Blur()
		if msg.Type == tea.KeyUp {
			m.focus = max(m.focus-1, 0)
		} else {
			m.focus = min(m.focus+1, len(m.inputs)-1)
		}
		return m, m.inputs[m.focus].Focus()
	}
	var cmd tea.Cmd
	m.inputs[m.focus], cmd = m.inputs[m.focus].Update(msg)
	return m, cmd
}

func (m Model) actions() []action {
	if a := steps[m.cursor].actions; a != nil {
		return a(m.f)
	}
	return nil
}

func (m Model) key(msg tea.KeyMsg) (Model, tea.Cmd) {
	if m.open {
		m.view.Width, m.view.Height = m.width, m.height-3
		m.view.SetContent(DetailView(m.f, m.cursor, m.row, m.width))
	}
	switch {
	case key.Matches(msg, keyQuit):
		return m, tea.Quit
	case key.Matches(msg, keyStop):
		return m, m.stopRun()
	case key.Matches(msg, keyBack):
		m.open = false
	case key.Matches(msg, keyDown):
		switch {
		case !m.open:
			m.moved, m.cursor = true, min(m.cursor+1, len(steps)-1)
		case m.row < len(m.actions())-1:
			m.row++
		default:
			m.view.LineDown(1)
		}
	case key.Matches(msg, keyUp):
		switch {
		case !m.open:
			m.moved, m.cursor = true, max(m.cursor-1, 0)
		case m.view.YOffset > 0:
			m.view.LineUp(1)
		case m.row > 0:
			m.row--
		}
	case key.Matches(msg, keyEnter):
		if !m.open {
			m.moved, m.open, m.row = true, true, 0
			m.view.GotoTop()
			return m, nil
		}
		if acts := m.actions(); m.row < len(acts) {
			return m.do(acts[m.row])
		}
	}
	return m, nil
}

// do runs one action row of the open step.
func (m Model) do(a action) (Model, tea.Cmd) {
	switch a.do {
	case "generate":
		return m, m.generate(a.arg)
	case "run":
		m.runSet = true
		return m, m.loadRun(a.arg)
	case "start":
		return m, m.startRun()
	case "stop":
		return m, m.stopRun()
	case "score":
		return m, m.openForm("score", a.arg, formLabels["score"]...)
	case "adopt":
		return m, m.openForm("adopt", "", formLabels["adopt"]...)
	}
	return m, nil
}

// mood picks the plush's face from the facts on screen.
func (m Model) mood() string {
	r := m.f.Run
	switch {
	case m.f.alive() || r != nil && r.State == "in_progress":
		return faceBusy
	case r != nil && (r.Window.Tripped != nil || r.Verdict == "fail"):
		return faceWorried
	case r != nil && (r.Verdict == "pass" || r.Verdict == "adopted"):
		return faceHappy
	case m.f.Runs == nil || len(m.f.Runs.Runs) == 0:
		return faceSleep
	}
	return faceIdle
}

func footer(open bool) string {
	pairs := [][2]string{{"↑↓", "step"}, {"enter", "open"}, {"q", "quit"}}
	if open {
		pairs = [][2]string{{"↑↓", "move"}, {"enter", "pick"}, {"esc", "back"}, {"x", "stop"}, {"q", "quit"}}
	}
	parts := []string{}
	for _, p := range pairs {
		parts = append(parts, stAccent.Render(p[0])+" "+stMuted.Render(p[1]))
	}
	return " " + strings.Join(parts, stMuted.Render(" · "))
}

func (m Model) View() string {
	left := stAccent.Render(glyphHeart) + " " + gradient("DayCare")
	right := stAccent.Render(m.mood())
	if m.f.alive() {
		right = m.spin.View() + " " + right
	}
	header := left + strings.Repeat(" ", max(m.width-lipgloss.Width(left)-lipgloss.Width(right), 1)) + right
	room := m.height - 3
	var body string
	if m.open {
		view := m.view
		view.Width, view.Height = m.width, room
		view.SetContent(DetailView(m.f, m.cursor, m.row, m.width))
		body = view.View()
	} else {
		body = ChecklistView(m.f, m.cursor, m.width, room)
		if pad := room - lipgloss.Height(body); pad > 0 {
			body += strings.Repeat("\n", pad)
		}
	}
	foot := footer(m.open)
	if id := m.runID(); id != "" {
		run := stMuted.Render("run " + id)
		foot += strings.Repeat(" ", max(m.width-lipgloss.Width(foot)-lipgloss.Width(run), 1)) + run
	}
	return header + "\n" + body + "\n" + truncate(m.note, m.width) + "\n" + truncate(foot, m.width)
}
