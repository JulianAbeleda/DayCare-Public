package seam

// The shapes `daycare.harness.runs` prints (schema daycare.tui.v1). Pointer fields are JSON null when the
// record does not carry them (a predeclared run has no record; an unscored gate has no measurement).

type Check struct {
	ID       string  `json:"id"`
	Label    string  `json:"label"`
	OK       bool    `json:"ok"`
	Detail   string  `json:"detail"`
	Fix      string  `json:"fix"`
	Generate *string `json:"generate"`
	// Memory is set on the gpu check only: the numbers behind its sentence, for the screen's bar.
	Memory *Memory `json:"memory"`
}

type Memory struct {
	Name    string  `json:"name"`
	FreeGB  float64 `json:"free_gb"`
	TotalGB float64 `json:"total_gb"`
	NeedGB  float64 `json:"need_gb"`
}

type Setup struct {
	Kind   string  `json:"kind"`
	Ready  bool    `json:"ready"`
	Checks []Check `json:"checks"`
	Python string  `json:"python"`
	Repo   string  `json:"repo"`
	Root   string  `json:"root"`
}

type Stopped struct {
	Update int    `json:"update"`
	Reason string `json:"reason"`
}

type GateMark struct {
	ID     string `json:"id"`
	Result string `json:"result"` // pass | fail | open
}

type Last struct {
	Step       *int     `json:"step"`
	KL         *float64 `json:"kl"`
	Entropy    *float64 `json:"entropy"`
	MeanReward *float64 `json:"mean_reward"`
}

type Adapter struct {
	Path    string  `json:"path"`
	Present bool    `json:"present"`
	GGUF    bool    `json:"gguf"`
	SHA256  *string `json:"sha256"`
}

type Summary struct {
	ID             string     `json:"id"`
	State          string     `json:"state"` // predeclared | in_progress | complete | stopped | failed
	UpdatesDone    int        `json:"updates_done"`
	UpdatesPlanned *int       `json:"updates_planned"`
	Protocol       *string    `json:"protocol"`
	Counted        *bool      `json:"counted"`
	Stopped        *Stopped   `json:"stopped"`
	Verdict        string     `json:"verdict"` // none | open | pass | fail | adopted | adopted (exception)
	Gates          []GateMark `json:"gates"`
	Last           Last       `json:"last"`
	Adapter        Adapter    `json:"adapter"`
}

type Runs struct {
	Kind string    `json:"kind"`
	Root string    `json:"root"`
	Runs []Summary `json:"runs"`
	// Unreadable names run folders whose records Python could not read, so one old record never hides the rest.
	Unreadable []Unreadable `json:"unreadable"`
}

type Unreadable struct {
	ID    string `json:"id"`
	Error string `json:"error"`
}

type Measured struct {
	Diff     float64  `json:"diff"`
	Lo       *float64 `json:"lo"`
	Hi       *float64 `json:"hi"`
	Passed   bool     `json:"passed"`
	Note     string   `json:"note"`
	ScoredAt string   `json:"scored_at"`
}

type Gate struct {
	ID         string    `json:"id"`
	Name       string    `json:"name"`
	Rule       string    `json:"rule"`
	RuleText   string    `json:"rule_text"`
	Value      float64   `json:"value"`
	Prediction string    `json:"prediction"`
	Result     string    `json:"result"`
	Measured   *Measured `json:"measured"`
}

type Reading struct {
	Metric  string   `json:"metric"`
	Now     *float64 `json:"now"`
	Base    *float64 `json:"base"`
	Limit   *float64 `json:"limit"`
	Side    string   `json:"side"`
	Crossed bool     `json:"crossed"`
}

type Window struct {
	Enabled  bool      `json:"enabled"`
	Source   string    `json:"source"`
	Rows     int       `json:"rows"`
	Window   int       `json:"window"`
	Base     int       `json:"base"`
	Ready    bool      `json:"ready"`
	Readings []Reading `json:"readings"`
	Tripped  *Stopped  `json:"tripped"`
}

type Verify struct {
	BitExact    bool    `json:"bit_exact"`
	Rows        int     `json:"rows"`
	Vocab       int     `json:"vocab"`
	MaxAbsError float64 `json:"max_abs_error"`
}

type Adoption struct {
	By        string `json:"by"`
	Exception string `json:"exception"`
	At        string `json:"at"`
}

type UpdateRow struct {
	Step           *int     `json:"step"`
	KL             *float64 `json:"kl"`
	Entropy        *float64 `json:"entropy"`
	MeanReward     *float64 `json:"mean_reward"`
	FinishedLength *float64 `json:"finished_length"`
	CappedTurnRate *float64 `json:"capped_turn_rate"`
	Skipped        bool     `json:"skipped"`
}

type Run struct {
	Summary
	Hypothesis    *string     `json:"hypothesis"`
	Recipe        []string    `json:"recipe"`
	PredeclaredAt *string     `json:"predeclared_at"`
	Adoption      *Adoption   `json:"adoption"`
	GateTable     []Gate      `json:"gate_table"`
	Window        Window      `json:"window"`
	Verify        *Verify     `json:"verify"`
	Updates       []UpdateRow `json:"updates"`
	Revision      *string     `json:"revision"`
	ModelSHA256   *string     `json:"model_sha256"`
}
