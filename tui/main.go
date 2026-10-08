// daycare-tui: the DayCare runs folder on a screen, or as JSON.
//
//	daycare-tui                      the Bubble Tea screens (needs a terminal)
//	daycare-tui --json <command>     the same facts as one JSON object; exit 0 ok, 1 error, 2 usage, 3 not ready
//
// Both modes read through the same seam (python -m daycare.harness.runs). Go never writes a run record.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/JulianAbeleda/DayCare/tui/internal/jobs"
	"github.com/JulianAbeleda/DayCare/tui/internal/seam"
	"github.com/JulianAbeleda/DayCare/tui/internal/ui"
)

const usage = `usage: daycare-tui [--json] [--root RUNS] [--repo DIR] [--python PY] [--state DIR] [command]

commands (each prints one JSON object):
  status                              what is missing before a run is possible (exit 3 when not ready)
  runs                                one summary per run folder
  run <id>                            gates vs results, the stop-trigger window, verify
  generate tasks [--name N]           freeze a task set (no private suites, no Countdown)
  generate predeclaration --name N --protocol P [--template run5] [--states S] [--envelope E]
                                      [--hypothesis T] [--gate 'ID|NAME|RULE|VALUE|PREDICTION']... [-- recipe...]
  start <id>                          run the predeclared recipe in the background; log under --state
  stop <id>                           SIGTERM the job started here
  tail <id> [--lines N]               the job's status and the last lines of its log
  score <id> --gate G --diff D [--lo L --hi H] [--note T]
  adopt <id> --by NAME --exception TEXT

env: DAYCARE_RUNS (runs folder), DAYCARE_PYTHON (interpreter, default python3), DAYCARE_REPO (checkout),
     DAYCARE_TUI_STATE (job pids and logs), plus the training env the status screen checks.`

func main() { os.Exit(run(os.Args[1:], os.Stdout, os.Stderr)) }

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// findRepo walks up from dir to the folder holding daycare/__init__.py.
func findRepo(dir string) string {
	for {
		if _, err := os.Stat(filepath.Join(dir, "daycare", "__init__.py")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return ""
		}
		dir = parent
	}
}

func emit(out io.Writer, raw []byte) int {
	fmt.Fprintln(out, string(raw))
	return 0
}

func fail(out io.Writer, err error) int {
	var detail string
	if e, ok := err.(*seam.Error); ok && e.Stderr != "" {
		detail = e.Stderr
	}
	raw, _ := json.Marshal(map[string]string{"schema": "daycare.tui.v1", "kind": "error", "error": err.Error(), "stderr": detail})
	fmt.Fprintln(out, string(raw))
	return 1
}

func run(args []string, out, errOut io.Writer) int {
	fs := flag.NewFlagSet("daycare-tui", flag.ContinueOnError)
	fs.SetOutput(errOut)
	jsonMode := fs.Bool("json", false, "print JSON (any command does; this also refuses the screens)")
	root := fs.String("root", os.Getenv("DAYCARE_RUNS"), "the runs folder")
	repo := fs.String("repo", os.Getenv("DAYCARE_REPO"), "the DayCare checkout")
	python := fs.String("python", envOr("DAYCARE_PYTHON", "python3"), "the interpreter with numpy and the RL modules")
	state := fs.String("state", jobs.DefaultDir(), "where job pids and logs live")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *repo == "" {
		cwd, _ := os.Getwd()
		*repo = findRepo(cwd)
		if *repo == "" {
			if exe, err := os.Executable(); err == nil {
				*repo = findRepo(filepath.Dir(exe))
			}
		}
	}
	if *repo == "" {
		fmt.Fprintln(errOut, "no DayCare checkout found: pass --repo or set DAYCARE_REPO")
		return 2
	}
	if *root == "" {
		*root = filepath.Join(*repo, "runs")
	}
	client := seam.Client{Python: *python, Repo: *repo, Root: *root}
	store := jobs.Store{Dir: *state}
	rest := fs.Args()
	if len(rest) == 0 {
		if *jsonMode {
			fmt.Fprintln(errOut, usage)
			return 2
		}
		if err := ui.Start(client, store); err != nil {
			fmt.Fprintln(errOut, err)
			return 1
		}
		return 0
	}
	return command(client, store, rest, out, errOut)
}

func command(client seam.Client, store jobs.Store, rest []string, out, errOut io.Writer) int {
	name, args := rest[0], rest[1:]
	need := func(n int) bool {
		if len(args) < n {
			fmt.Fprintln(errOut, usage)
			return false
		}
		return true
	}
	switch name {
	case "status":
		s, raw, err := client.Setup()
		if err != nil {
			return fail(out, err)
		}
		emit(out, raw)
		if !s.Ready {
			return 3
		}
		return 0
	case "runs":
		_, raw, err := client.List()
		if err != nil {
			return fail(out, err)
		}
		return emit(out, raw)
	case "run":
		if !need(1) {
			return 2
		}
		_, raw, err := client.Show(args[0])
		if err != nil {
			return fail(out, err)
		}
		return emit(out, raw)
	case "generate":
		if !need(1) {
			return 2
		}
		return generate(client, args, out, errOut)
	case "start":
		if !need(1) {
			return 2
		}
		r, _, err := client.Show(args[0])
		if err != nil {
			return fail(out, err)
		}
		if r.State != "predeclared" {
			return fail(out, fmt.Errorf("run %s is %s; only a predeclared run can start", r.ID, r.State))
		}
		dir, _ := client.RunDir(r.ID)
		job, err := store.Start(r.ID, client.Repo, client.TrainArgv(dir, r.Recipe))
		if err != nil {
			return fail(out, err)
		}
		raw, _ := json.Marshal(job)
		return emit(out, raw)
	case "stop":
		if !need(1) {
			return 2
		}
		job, err := store.Stop(args[0])
		if err != nil {
			return fail(out, err)
		}
		raw, _ := json.Marshal(job)
		return emit(out, raw)
	case "tail":
		if !need(1) {
			return 2
		}
		fs := flag.NewFlagSet("tail", flag.ContinueOnError)
		lines := fs.Int("lines", 30, "lines of log")
		if fs.Parse(args[1:]) != nil {
			return 2
		}
		job, err := store.Status(args[0])
		if err != nil {
			return fail(out, fmt.Errorf("no job was started for %s from this machine", args[0]))
		}
		tail, _ := store.Tail(args[0], *lines)
		raw, _ := json.Marshal(map[string]any{"schema": "daycare.tui.v1", "kind": "job", "job": job, "lines": tail})
		return emit(out, raw)
	case "score":
		if !need(1) {
			return 2
		}
		fs := flag.NewFlagSet("score", flag.ContinueOnError)
		gate := fs.String("gate", "", "gate id")
		diff := fs.Float64("diff", 0, "paired difference, or the count")
		lo := fs.Float64("lo", 0, "CI lower bound")
		hi := fs.Float64("hi", 0, "CI upper bound")
		note := fs.String("note", "", "note")
		if fs.Parse(args[1:]) != nil || *gate == "" {
			fmt.Fprintln(errOut, usage)
			return 2
		}
		var pl, ph *float64
		if isSet(fs, "lo") && isSet(fs, "hi") {
			pl, ph = lo, hi
		}
		raw, err := client.Score(args[0], *gate, *diff, pl, ph, *note)
		if err != nil {
			return fail(out, err)
		}
		return emit(out, raw)
	case "adopt":
		if !need(1) {
			return 2
		}
		fs := flag.NewFlagSet("adopt", flag.ContinueOnError)
		by := fs.String("by", "", "who")
		exception := fs.String("exception", "", "the exception with its evidence and known costs")
		if fs.Parse(args[1:]) != nil || *by == "" || *exception == "" {
			fmt.Fprintln(errOut, usage)
			return 2
		}
		raw, err := client.Adopt(args[0], *by, *exception)
		if err != nil {
			return fail(out, err)
		}
		return emit(out, raw)
	}
	fmt.Fprintln(errOut, usage)
	return 2
}

func isSet(fs *flag.FlagSet, name string) bool {
	set := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == name {
			set = true
		}
	})
	return set
}

type gateList []string

func (g *gateList) String() string     { return strings.Join(*g, ";") }
func (g *gateList) Set(v string) error { *g = append(*g, v); return nil }

func generate(client seam.Client, args []string, out, errOut io.Writer) int {
	thing := args[0]
	fs := flag.NewFlagSet("generate", flag.ContinueOnError)
	name := fs.String("name", "", "run or task-set folder name")
	protocol := fs.String("protocol", "", "the research record this run belongs to")
	template := fs.String("template", "", "gate and recipe template (run5)")
	hypothesis := fs.String("hypothesis", "", "one mechanism and its predictions")
	states := fs.String("states", "", "states.xml for the recipe")
	envelope := fs.String("envelope", os.Getenv("DAYCARE_ENVELOPE"), "the envelope record for the recipe")
	var gates gateList
	fs.Var(&gates, "gate", "ID|NAME|RULE|VALUE|PREDICTION (repeatable)")
	if fs.Parse(args[1:]) != nil {
		return 2
	}
	switch thing {
	case "tasks":
		if *name == "" {
			runs, _, err := client.List()
			if err != nil {
				return fail(out, err)
			}
			*name = seam.NextName(runs, "posttool-tasks")
		}
		raw, err := client.FreezeTasks(*name)
		if err != nil {
			return fail(out, err)
		}
		return emit(out, raw)
	case "predeclaration":
		if *name == "" || *protocol == "" {
			fmt.Fprintln(errOut, "generate predeclaration needs --name and --protocol")
			return 2
		}
		if *template == "" && len(gates) == 0 {
			*template = "run5"
		}
		raw, err := client.Predeclare(seam.Predeclaration{ID: *name, Protocol: *protocol, Template: *template,
			Hypothesis: *hypothesis, States: *states, Envelope: *envelope, Gates: gates, Recipe: fs.Args()})
		if err != nil {
			return fail(out, err)
		}
		return emit(out, raw)
	}
	fmt.Fprintln(errOut, "generate takes: tasks | predeclaration")
	return 2
}
