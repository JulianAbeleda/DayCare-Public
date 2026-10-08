package ui

import (
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/help"
	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/JulianAbeleda/DayCare/tui/internal/jobs"
	"github.com/JulianAbeleda/DayCare/tui/internal/seam"
)

type screen int

const (
	screenStatus screen = iota
	screenRuns
	screenRun
	screenAdapters
	screenJobs
)

var tabs = []string{"Setup", "Runs", "Adapters", "Job"}

type keyMap struct{ Tabs, Move, Open, Back, Start, Stop, Generate, Mode, Refresh, Quit key.Binding }

var keys = keyMap{
	Tabs:     key.NewBinding(key.WithKeys("1", "2", "3", "4"), key.WithHelp("1-4", "screens")),
	Move:     key.NewBinding(key.WithKeys("j", "k", "up", "down"), key.WithHelp("j/k", "move")),
	Open:     key.NewBinding(key.WithKeys("enter"), key.WithHelp("enter", "open")),
	Back:     key.NewBinding(key.WithKeys("esc"), key.WithHelp("esc", "back")),
	Start:    key.NewBinding(key.WithKeys("s"), key.WithHelp("s", "start")),
	Stop:     key.NewBinding(key.WithKeys("x"), key.WithHelp("x", "stop")),
	Generate: key.NewBinding(key.WithKeys("g"), key.WithHelp("g", "generate")),
	Mode:     key.NewBinding(key.WithKeys("t"), key.WithHelp("t", "plain/technical")),
	Refresh:  key.NewBinding(key.WithKeys("r"), key.WithHelp("r", "refresh")),
	Quit:     key.NewBinding(key.WithKeys("q", "ctrl+c"), key.WithHelp("q", "quit")),
}

func (k keyMap) ShortHelp() []key.Binding {
	return []key.Binding{k.Tabs, k.Move, k.Open, k.Start, k.Stop, k.Generate, k.Mode, k.Refresh, k.Quit}
}
func (k keyMap) FullHelp() [][]key.Binding { return [][]key.Binding{k.ShortHelp()} }

// Model holds the seam data each screen renders. Every load is a tea.Cmd that calls the seam off the UI thread.
type Model struct {
	client    seam.Client
	store     jobs.Store
	screen    screen
	technical bool
	cursor    int
	width     int
	height    int
	setup     *seam.Setup
	runs      *seam.Runs
	run       *seam.Run
	jobID     string
	job       *jobs.Job
	tail      []string
	note      string
	view      viewport.Model
	spin      spinner.Model
	help      help.Model
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
	s := spinner.New(spinner.WithSpinner(spinner.MiniDot), spinner.WithStyle(stAccent))
	h := help.New()
	h.Styles.ShortKey, h.Styles.ShortDesc, h.Styles.ShortSeparator = stAccent, stMuted, stMuted
	return Model{client: client, store: store, width: 100, height: 30, view: viewport.New(100, 25), spin: s, help: h}
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

func (m Model) generate() tea.Cmd {
	if m.setup == nil || m.cursor >= len(m.setup.Checks) || m.setup.Checks[m.cursor].Generate == nil {
		return func() tea.Msg { return noteMsg("Nothing to generate for this item.") }
	}
	thing := *m.setup.Checks[m.cursor].Generate
	return func() tea.Msg {
		switch thing {
		case "tasks":
			name := seam.NextName(m.runs, "posttool-tasks")
			if _, err := m.client.FreezeTasks(name); err != nil {
				return noteMsg("Generate failed: " + err.Error())
			}
			return noteMsg("Generated tasks under " + name + ". The states need the GPU harvest; see the fix text.")
		case "predeclaration":
			p := predeclarationFromSetup(m.setup, seam.NextName(m.runs, "posttool-predeclared"))
			if _, err := m.client.Predeclare(p); err != nil {
				return noteMsg("Generate failed: " + err.Error())
			}
			return noteMsg("Predeclared " + p.ID + " from template run5. Write the research record before the first update.")
		}
		return noteMsg("Unknown generator " + thing)
	}
}

// predeclarationFromSetup fills the template's recipe with the states file the setup check found and the
// envelope from the environment; a missing one stays missing, and the status screen keeps saying so.
func predeclarationFromSetup(s *seam.Setup, name string) seam.Predeclaration {
	p := seam.Predeclaration{ID: name, Protocol: "rloo-" + name + ".md", Template: "run5",
		Envelope: os.Getenv("DAYCARE_ENVELOPE")}
	for _, c := range s.Checks {
		if c.ID == "states" && c.OK {
			p.States = c.Detail
		}
	}
	return p
}

func (m Model) startRun() tea.Cmd {
	run := m.run
	return func() tea.Msg {
		if run == nil || run.State != "predeclared" {
			return noteMsg("Only a predeclared run can start.")
		}
		dir, err := m.client.RunDir(run.ID)
		if err != nil {
			return noteMsg(err.Error())
		}
		if _, err := m.store.Start(run.ID, m.client.Repo, m.client.TrainArgv(dir, run.Recipe)); err != nil {
			return noteMsg("Start failed: " + err.Error())
		}
		return noteMsg("Started " + run.ID + ". Press 4 to watch its output.")
	}
}

func (m Model) stopRun() tea.Cmd {
	id := m.jobID
	return func() tea.Msg {
		if id == "" {
			return noteMsg("No run selected.")
		}
		if _, err := m.store.Stop(id); err != nil {
			return noteMsg("Stop failed: " + err.Error())
		}
		return noteMsg("Sent SIGTERM to " + id + ". The loop keeps its last checkpoint.")
	}
}

func (m Model) rows() int {
	switch m.screen {
	case screenStatus:
		if m.setup != nil {
			return len(m.setup.Checks)
		}
	case screenRuns:
		if m.runs != nil {
			return len(m.runs.Runs)
		}
	}
	return 0
}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
		m.view.Width, m.view.Height = msg.Width, msg.Height-4
	case spinner.TickMsg:
		var cmd tea.Cmd
		m.spin, cmd = m.spin.Update(msg)
		return m, cmd
	case setupMsg:
		m.setup = msg.setup
		if msg.err != nil {
			m.note = "Setup check failed: " + msg.err.Error()
		}
	case runsMsg:
		m.runs = msg.runs
		if msg.err != nil {
			m.note = "Runs could not be read: " + msg.err.Error()
		}
	case runMsg:
		m.run = msg.run
		if msg.err != nil {
			m.note = "Run could not be read: " + msg.err.Error()
		} else {
			m.screen, m.jobID = screenRun, msg.run.ID
			m.view.GotoTop()
		}
	case jobMsg:
		m.job, m.tail = msg.job, msg.tail
	case noteMsg:
		m.note = string(msg)
		return m, tea.Batch(m.loadSetup(), m.loadRuns())
	case tickMsg:
		if m.screen == screenJobs {
			return m, tea.Batch(m.loadJob(m.jobID), tick())
		}
	case tea.KeyMsg:
		return m.key(msg)
	}
	return m, nil
}

func (m Model) key(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	scrolls := m.screen == screenRun || m.screen == screenJobs
	switch {
	case key.Matches(msg, keys.Quit):
		return m, tea.Quit
	case key.Matches(msg, keys.Tabs):
		m.screen, m.cursor = map[string]screen{"1": screenStatus, "2": screenRuns, "3": screenAdapters, "4": screenJobs}[msg.String()], 0
		m.view.GotoTop()
		if m.screen == screenJobs {
			return m, tea.Batch(m.loadJob(m.jobID), tick())
		}
	case key.Matches(msg, keys.Move):
		down := msg.String() == "j" || msg.String() == "down"
		switch {
		case scrolls && down:
			m.view.LineDown(1)
		case scrolls:
			m.view.LineUp(1)
		case down && m.cursor+1 < m.rows():
			m.cursor++
		case !down && m.cursor > 0:
			m.cursor--
		}
	case key.Matches(msg, keys.Open):
		if m.screen == screenRuns && m.runs != nil && m.cursor < len(m.runs.Runs) {
			return m, m.loadRun(m.runs.Runs[m.cursor].ID)
		}
	case key.Matches(msg, keys.Back):
		if scrolls {
			m.screen = screenRuns
		}
	case key.Matches(msg, keys.Mode):
		m.technical = !m.technical
	case key.Matches(msg, keys.Refresh):
		cmds := []tea.Cmd{m.loadSetup(), m.loadRuns()}
		if m.run != nil {
			cmds = append(cmds, m.loadRun(m.run.ID))
		}
		return m, tea.Batch(cmds...)
	case key.Matches(msg, keys.Generate):
		if m.screen == screenStatus {
			return m, m.generate()
		}
	case key.Matches(msg, keys.Start):
		if m.screen == screenRun {
			return m, m.startRun()
		}
	case key.Matches(msg, keys.Stop):
		return m, m.stopRun()
	}
	return m, nil
}

// mood picks the mascot's face from what the screen shows.
func (m Model) mood() string {
	if m.job != nil && m.job.Alive {
		return faceBusy
	}
	if m.screen == screenRun && m.run != nil {
		switch {
		case m.run.Window.Tripped != nil || m.run.Verdict == "fail":
			return faceWorried
		case m.run.Verdict == "pass" || m.run.Verdict == "adopted":
			return faceHappy
		case m.run.State == "in_progress":
			return faceBusy
		}
		return faceIdle
	}
	if m.runs == nil || len(m.runs.Runs) == 0 {
		return faceSleep
	}
	for _, r := range m.runs.Runs {
		if r.State == "in_progress" {
			return faceBusy
		}
	}
	return faceIdle
}

func (m Model) body() string {
	switch m.screen {
	case screenStatus:
		return StatusView(m.setup, m.cursor, m.technical, m.width)
	case screenRuns:
		return RunsView(m.runs, m.cursor, m.technical, m.width)
	case screenRun:
		return RunView(m.run, m.technical, m.width)
	case screenAdapters:
		return AdaptersView(m.runs, m.technical, m.width)
	}
	return JobsView(m.jobID, m.job, m.tail, m.technical, m.width)
}

func (m Model) View() string {
	active := map[screen]int{screenStatus: 0, screenRuns: 1, screenRun: 1, screenAdapters: 2, screenJobs: 3}[m.screen]
	parts := []string{stAccent.Render(glyphHeart) + " " + gradient("DayCare")}
	for i, t := range tabs {
		if i == active {
			parts = append(parts, stTabOn.Render(t))
		} else {
			parts = append(parts, stTabOff.Render(t))
		}
	}
	busy := ""
	if m.job != nil && m.job.Alive {
		busy = m.spin.View() + " "
	}
	header := strings.Join(parts, "") + "  " + busy + stAccent.Render(m.mood()) + "  " + mode(m.technical)
	view := m.view
	view.Width, view.Height = m.width, m.height-4
	view.SetContent(m.body())
	note := m.note
	if note == "" {
		note = stMuted.Render("Keys below. Press t for the technical words.")
	}
	m.help.Width = m.width
	return fmt.Sprintf("%s\n%s\n%s\n%s", header, view.View(), lipgloss.NewStyle().Width(m.width).Render(note), m.help.View(keys))
}
