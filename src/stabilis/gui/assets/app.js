"use strict";
/* The page. Three stages, one state object, and no framework.
 *
 * The shape worth knowing: `S` is the only mutable state, every control
 * writes into it, and every render function reads from it. There is no
 * two-way binding and no component tree, because there are about thirty
 * controls here and a framework to manage them would be more code than the
 * page.
 *
 * Reactivity is a debounce plus a cached file on the server. Retyping an
 * expression, dragging a window or zooming the axis all end in one small
 * POST that slices an array already in memory -- the file is read once, at
 * `open`, and never again. That is the whole performance story; nothing
 * here is clever.
 */

const $ = id => document.getElementById(id);
const COLORS = ["#2f6fd0", "#d9534f", "#41a05a", "#b06ac2",
                "#d98d20", "#20a3b0", "#8a6d3b", "#7d7d7d"];

const S = {
  dir: "", path: null, file: null,
  labels: [], shown: [], windows: [],
  dataKind: "phase", unit: "s", rate: "", timeColumn: "", timeFormat: "",
  series: [{ expr: "", label: "" }],
  outlier: "", reason: "", join: false,
  estimator: "oadev", taus: "octave", maxAf: "", ci: true, level: "0.6827",
  together: true, stage: 1, rawView: null,
};

/* Filled in from /api/capabilities at startup. The engine owns the list of
 * estimators, tau grids and units; a second copy here is how a page ends up
 * quietly older than the library it drives. */
let UNITS = { phase: ["s"], frequency: ["fractional"] };

function esc(v) {
  return String(v === undefined || v === null ? "" : v)
    .replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;",
                                '"': "&quot;" }[c]));
}
function fail(message) { $("err").textContent = message || ""; }

function debounce(fn, ms) {
  let timer = null;
  return function () {
    const args = arguments;
    clearTimeout(timer);
    timer = setTimeout(() => fn.apply(null, args), ms);
  };
}

async function call(url, body) {
  fail("");
  const opts = body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
  let reply;
  try {
    reply = await (await fetch(url, opts)).json();
  } catch (e) {
    fail("cannot reach the server: " + e);
    return null;
  }
  if (reply.error) { fail(reply.error); return null; }
  return reply;
}

/* ------------------------------------------------------------------ files */

async function browse(dir) {
  const j = await call("/api/files?dir=" + encodeURIComponent(dir));
  if (!j) return;
  S.dir = j.here;
  $("dirpath").value = j.here;
  $("root").textContent = "root: " + j.root + (j.here ? " / " + j.here : "");
  const box = $("files");
  box.innerHTML = "";
  const add = (text, fn, isDir) => {
    const el = document.createElement("div");
    el.textContent = text;
    if (isDir) el.className = "dir";
    el.onclick = fn;
    box.appendChild(el);
    return el;
  };
  if (!j.at_root) add("../", () => browse(j.parent), true);
  for (const d of j.dirs)
    add(d + "/", () => browse(S.dir ? S.dir + "/" + d : d), true);
  for (const f of j.files)
    add(f.name + "  (" + size(f.size) + ")", ev => {
      [...box.children].forEach(c => c.classList.remove("sel"));
      ev.currentTarget.classList.add("sel");
      openFile(S.dir ? S.dir + "/" + f.name : f.name);
    }, false);
  if (!j.dirs.length && !j.files.length) {
    const el = add("no folders or data files here", null, false);
    el.className = "empty";
  }
}

function size(n) {
  return n > 1e6 ? (n / 1e6).toFixed(1) + " MB"
       : n > 1e3 ? (n / 1e3).toFixed(0) + " kB" : n + " B";
}

async function openFile(path) {
  $("filefacts").textContent = "reading…";
  const j = await call("/api/open", { path: path });
  if (!j) { $("filefacts").textContent = ""; return; }
  S.path = path;
  S.file = j;
  S.labels = j.labels.slice();
  S.windows = [];
  S.rawView = null;
  S.shown = j.columns.filter(c => !c.text && !c.monotonic)
                     .slice(0, 3).map(c => c.index);
  if (!S.shown.length) S.shown = j.columns.filter(c => !c.text).slice(0, 1)
                                          .map(c => c.index);
  S.series = S.shown.map(i => ({ expr: S.labels[i], label: "" }));
  if (!S.series.length) S.series = [{ expr: "", label: "" }];

  const stamps = j.columns.filter(c => c.monotonic);
  S.timeColumn = stamps.length ? String(stamps[0].index) : "";

  $("filefacts").innerHTML =
    '<span class="chip">' + j.n.toLocaleString() + " rows</span>"
    + '<span class="chip">' + j.n_columns + " columns</span>"
    + (j.defects ? '<br><span class="warn">tolerated: ' + esc(j.defects)
                   + "</span>" : "");
  $("headernote").textContent = j.has_header
    ? "This file names its own columns; those names are filled in below."
    : "No header line, so the columns are numbered. Name them here and the "
      + "names carry through to stage 2.";
  $("suggestion").textContent = j.suggest
    ? "Suggested grid for " + j.n.toLocaleString() + " points: "
      + j.suggest.mode + " to AF " + j.suggest.max_af + " — " + j.suggest.why
    : "";
  renderColumns();
  renderTimeColumns();
  renderSeries();
  renderWindows();
  rawPlot.autoscale();
  refreshRaw();
}

/* -------------------------------------------------- stage 1: the raw view */

function renderColumns() {
  if (!S.file) return;
  $("collist").innerHTML = S.file.columns.map(c => {
    const on = S.shown.indexOf(c.index) >= 0;
    const colour = COLORS[S.shown.indexOf(c.index) % COLORS.length];
    return '<tr><td style="width:18px">'
      + '<input type="checkbox" data-col="' + c.index + '"'
      + (on ? " checked" : "") + (c.text ? " disabled" : "") + "></td>"
      + '<td style="width:14px">' + (on
          ? '<i class="swatch" style="background:' + colour + '"></i>' : "")
      + "</td>"
      + '<td><input type="text" data-label="' + c.index + '" value="'
      + esc(S.labels[c.index]) + '"></td>'
      + '<td class="n">' + (c.text ? "text"
          : c.n_finite.toLocaleString() + " ok") + "</td></tr>";
  }).join("");
  for (const box of $("collist").querySelectorAll("input[data-col]"))
    box.onchange = () => {
      const i = Number(box.dataset.col);
      const at = S.shown.indexOf(i);
      if (at >= 0) S.shown.splice(at, 1); else S.shown.push(i);
      S.shown.sort((a, b) => a - b);
      renderColumns();
      refreshRaw();
    };
  for (const box of $("collist").querySelectorAll("input[data-label]"))
    box.oninput = debounce(() => {
      S.labels[Number(box.dataset.label)] = box.value;
      renderChips();
      renderTimeColumns();
      refreshRaw();
    }, 250);
}

function renderTimeColumns() {
  if (!S.file) return;
  $("time_column").innerHTML = '<option value="">none</option>'
    + S.file.columns.map(c =>
        '<option value="' + c.index + '"'
        + (String(c.index) === S.timeColumn ? " selected" : "") + ">"
        + esc(S.labels[c.index]) + (c.monotonic ? " ↑" : "")
        + (c.text ? " (text)" : "") + "</option>").join("");
}

const rawPlot = createPlot($("rawplot"), {
  xlabel: "sample index", ylabel: "value in the file",
  onSelect: (a, b) => {
    if (a === null) return;
    const lo = Math.max(0, Math.round(Math.min(a, b)));
    const hi = Math.min(S.file ? S.file.n : 0, Math.round(Math.max(a, b)));
    if (hi - lo < 2) return;
    S.windows.push([lo, hi]);
    renderWindows();
    refreshRaw();
  },
  onViewChange: () => refreshRawSoon(),
  onHover: (x, y) => Math.round(x).toLocaleString() + ", " + y.toPrecision(6),
});

const prepPlot = createPlot($("prepplot"), {
  xlabel: "sample index", ylabel: "value handed to the library",
  selectable: false,
  onHover: (x, y) => Math.round(x).toLocaleString() + ", " + y.toPrecision(6),
});

function renderWindows() {
  const box = $("windows");
  if (!S.windows.length) {
    box.innerHTML = '<span class="tiny muted">No windows &mdash; the whole '
      + "record will be analysed.</span>";
    return;
  }
  box.innerHTML = S.windows.map((w, i) =>
    '<span class="window">' + w[0].toLocaleString() + " – "
    + w[1].toLocaleString() + ' <button class="drop" data-w="' + i
    + '">&times;</button></span>').join("");
  for (const b of box.querySelectorAll("button[data-w]"))
    b.onclick = () => {
      S.windows.splice(Number(b.dataset.w), 1);
      renderWindows();
      refreshRaw();
    };
}

async function refreshRaw() {
  if (!S.path || !S.shown.length) {
    rawPlot.setSeries([]);
    rawPlot.setBands([]);
    rawPlot.draw();
    return;
  }
  /* Until the first response has been drawn the plot's own view is the
   * placeholder extent a plotter with no data invents, so asking it what to
   * fetch would request samples 0 to 1. Send null -- meaning the whole
   * record -- and only start honouring the view once there is one. */
  const view = S.rawView ? rawPlot.getView() : null;
  const j = await call("/api/traces", {
    path: S.path, labels: S.labels,
    series: S.shown.map(i => ({ expr: "#" + i, label: S.labels[i] })),
    windows: S.windows,
    view: view ? [Math.floor(view.x0), Math.ceil(view.x1)] : null,
    target: Math.max(200, Math.round(rawPlot.width() * 2)),
  });
  if (!j) return;
  rawPlot.setSeries(j.series.map((s, i) => ({
    kind: s.kind, i0: s.i0, lo: s.lo, hi: s.hi,
    x: s.i0, y: s.lo, color: COLORS[i % COLORS.length], label: s.label,
  })));
  rawPlot.setBands(S.windows.map(w => ({ x0: w[0], x1: w[1] })));
  if (!S.rawView) { rawPlot.autoscale(); S.rawView = true; }
  rawPlot.draw();
  $("rawinfo").textContent = "— " + j.series.map(s =>
    s.label + ": " + s.stats.n.toLocaleString() + " pts, "
    + s.stats.n_gaps.toLocaleString() + " gaps").join("   ");
}
const refreshRawSoon = debounce(refreshRaw, 130);

/* ------------------------------------------------ stage 2: what it means */

function renderUnits() {
  const list = UNITS[S.dataKind];
  $("unit").innerHTML = list.map(u =>
    '<option' + (u === S.unit ? " selected" : "") + ">" + u + "</option>").join("");
  if (list.indexOf(S.unit) < 0) S.unit = list[0];
}

function renderChips() {
  if (!S.file) { $("chips").innerHTML = ""; return; }
  $("chips").innerHTML = "columns: " + S.labels.map((name, i) =>
    '<a class="chip" data-name="' + esc(name) + '">' + esc(name)
    + "</a>").join("");
  for (const a of $("chips").querySelectorAll("a.chip"))
    a.onclick = () => {
      const box = document.querySelector(
        '#serieslist input[data-f="expr"][data-i="' + activeSeries + '"]');
      if (!box) return;
      // A trailing operator means the user has already said what they want
      // done with the next name; inserting another " - " would make
      // "NCS -  - VSL". Only supply the operator when there is none.
      const so_far = box.value.replace(/\s+$/, "");
      box.value = (!so_far || /[-+*/(]$/.test(so_far))
        ? so_far + (so_far ? " " : "") + a.dataset.name
        : so_far + " - " + a.dataset.name;
      S.series[activeSeries].expr = box.value;
      box.focus();
      refreshPrepSoon();
    };
}

let activeSeries = 0;

function renderSeries() {
  $("serieslist").innerHTML = S.series.map((s, i) =>
    '<tr><td style="width:14px"><i class="swatch" style="background:'
    + COLORS[i % COLORS.length] + '"></i></td>'
    + '<td><input data-f="expr" data-i="' + i + '" value="' + esc(s.expr)
    + '" placeholder="NCS - VSL"></td>'
    + '<td style="width:34%"><input data-f="label" data-i="' + i + '" value="'
    + esc(s.label) + '" placeholder="label (optional)"></td>'
    + '<td style="width:22px">' + (S.series.length > 1
        ? '<button class="drop" data-d="' + i + '">&times;</button>' : "")
    + "</td></tr>").join("");
  for (const el of $("serieslist").querySelectorAll("input")) {
    el.onfocus = () => { activeSeries = Number(el.dataset.i); };
    el.oninput = debounce(() => {
      S.series[Number(el.dataset.i)][el.dataset.f] = el.value;
      refreshPrep();
    }, 250);
  }
  for (const b of $("serieslist").querySelectorAll("button[data-d]"))
    b.onclick = () => {
      S.series.splice(Number(b.dataset.d), 1);
      activeSeries = 0;
      renderSeries();
      refreshPrep();
    };
  renderChips();
}

let prepShape = null;

async function refreshPrep() {
  if (!S.path || S.stage !== 2) return;
  const wanted = S.series.filter(s => s.expr.trim());
  if (!wanted.length) {
    prepPlot.setSeries([]); prepPlot.draw();
    $("prepinfo").textContent = "";
    return;
  }
  const j = await call("/api/traces", body({
    scaled: true,
    series: wanted.map(s => ({ expr: s.expr, label: s.label || s.expr })),
    target: Math.max(200, Math.round(prepPlot.width() * 2)),
  }));
  if (!j) return;
  prepPlot.setSeries(j.series.map((s, i) => ({
    kind: s.kind, i0: s.i0, lo: s.lo, hi: s.hi,
    x: s.i0, y: s.lo, color: COLORS[i % COLORS.length], label: s.label,
  })));
  prepPlot.setBands(S.windows.map(w => ({ x0: w[0], x1: w[1] })));
  prepPlot.setLabels("sample index",
    S.dataKind === "phase" ? "phase x(t) (s)" : "fractional frequency y(t)");
  // Refit only when what is being drawn actually changed. Autoscaling on
  // every keystroke would snap the axes back under someone who had zoomed
  // in to look at something.
  const fingerprint = JSON.stringify([wanted.map(s => s.expr), S.dataKind,
                                      S.unit, S.windows]);
  if (fingerprint !== prepShape) { prepShape = fingerprint; prepPlot.autoscale(); }
  prepPlot.draw();
  $("ratenote").textContent = j.rate_note
    + (j.jitter !== null && j.jitter !== undefined
       ? "; interquartile spread " + (j.jitter * 100).toFixed(3)
         + "% of the median interval" : "");
  if (j.jitter > 1e-3)
    $("ratenote").className = "tiny warn";
  else $("ratenote").className = "tiny muted";
  $("prepinfo").innerHTML = j.series.map(s =>
    "<b>" + esc(s.label) + "</b>: " + s.stats.n.toLocaleString()
    + " selected, " + s.stats.n_gaps.toLocaleString() + " gaps"
    + (s.divide_by_zero
       ? ', <span class="warn">' + s.divide_by_zero
         + " became gaps (zero divisor)</span>" : "")).join("<br>");
}
const refreshPrepSoon = debounce(refreshPrep, 130);

/* Everything both later endpoints need to know about the record. */
function body(extra) {
  return Object.assign({
    path: S.path, labels: S.labels,
    series: S.series.filter(s => s.expr.trim())
                    .map(s => ({ expr: s.expr, label: s.label || s.expr })),
    windows: S.windows,
    data_kind: S.dataKind, unit: S.unit,
    rate: S.rate, time_column: S.timeColumn, time_format: S.timeFormat,
    outlier_threshold: S.outlier, reason: S.reason, join: S.join,
  }, extra || {});
}

/* --------------------------------------------------- stage 3: the answers */

async function analyse() {
  if (!S.path) { fail("pick a file first"); return; }
  $("go").disabled = true;
  $("go").textContent = "working…";
  $("results").classList.add("busy");
  const j = await call("/api/analyze", body({
    estimator: S.estimator, taus: S.taus, max_af: S.maxAf,
    confidence: S.ci, level: Number(S.level),
  }));
  $("go").disabled = false;
  $("go").textContent = "Analyse";
  $("results").classList.remove("busy");
  if (j) renderResults(j);
}

let resultPlots = [];

function renderResults(j) {
  const box = $("results");
  // The canvases these are attached to are about to be thrown away; the
  // plotters hold a ResizeObserver each and would keep observing them.
  for (const p of resultPlots) p.destroy();
  resultPlots = [];
  box.innerHTML = "";
  const notes = j.notes.map(n => '<p class="tiny warn">' + esc(n) + "</p>").join("");
  const head = '<p class="tiny muted">' + esc(j.kind) + " · "
    + j.rate_hz.toPrecision(8) + " Hz (" + esc(j.rate_note) + ") · read as "
    + esc(j.read_as) + " in " + esc(j.unit) + "</p>";

  if (S.together) {
    const panel = document.createElement("div");
    panel.className = "panel";
    panel.innerHTML = head + notes
      + '<div class="plotbox tall"><canvas></canvas></div>'
      + '<div class="legend">' + j.series.map((s, i) =>
          '<span><i style="background:' + COLORS[i % COLORS.length]
          + '"></i>' + esc(s.label) + "</span>").join("") + "</div>"
      + j.series.map((s, i) => table(s, j.kind, i)).join("");
    box.appendChild(panel);
    sigmaPlot(panel.querySelector("canvas"), j.series, j.kind);
    return;
  }

  const grid = document.createElement("div");
  grid.className = "results-grid";
  box.appendChild(grid);
  const notice = document.createElement("div");
  notice.innerHTML = head + notes;
  box.insertBefore(notice, grid);
  j.series.forEach((s, i) => {
    const panel = document.createElement("div");
    panel.className = "panel";
    panel.innerHTML = "<h2>" + esc(s.label) + "</h2>"
      + '<div class="plotbox"><canvas></canvas></div>' + table(s, j.kind, i);
    grid.appendChild(panel);
    sigmaPlot(panel.querySelector("canvas"), [s], j.kind, i);
  });
}

function sigmaPlot(canvas, list, kind, offset) {
  const plot = createPlot(canvas, {
    xlog: true, ylog: true, selectable: false,
    xlabel: "averaging time tau (s)", ylabel: kind,
    onHover: (x, y) => "tau " + x.toPrecision(4) + " s, " + y.toExponential(3),
  });
  plot.setSeries(list.map((s, i) => ({
    kind: "line", x: s.rows.tau, y: s.rows.dev,
    lo: s.rows.lo || null, hi: s.rows.hi || null,
    color: COLORS[((offset || 0) + i) % COLORS.length], label: s.label,
  })));
  plot.autoscale();
  resultPlots.push(plot);
}

function table(s, kind, index) {
  const r = s.rows;
  const ci = s.has_ci;
  let h = '<div class="scroll"><table class="numbers"><tr><th>AF</th>'
    + "<th>tau (s)</th><th>#</th><th>" + esc(kind) + "</th>"
    + (ci ? "<th>lo</th><th>hi</th><th>EDOF</th><th>via</th>" : "") + "</tr>";
  for (let i = 0; i < r.af.length; i++) {
    h += "<tr><td>" + r.af[i] + "</td><td>" + num(r.tau[i], 5) + "</td><td>"
       + r.n_terms[i] + "</td><td>" + exp(r.dev[i]) + "</td>";
    if (ci) h += "<td>" + exp(r.lo[i]) + "</td><td>" + exp(r.hi[i])
       + "</td><td>" + num(r.edof[i], 3) + '</td><td class="via">'
       + esc(r.via[i]) + "</td>";
    h += "</tr>";
  }
  h += "</table></div>";
  h += '<p class="tiny muted">' + s.summary.n.toLocaleString() + " points, "
     + s.summary.n_gaps.toLocaleString() + " gaps ("
     + (s.summary.gap_fraction * 100).toFixed(2) + "%), "
     + s.summary.runs + " runs, longest "
     + s.summary.longest_run.toLocaleString() + "</p>";
  h += '<p class="tiny muted">the same thing without the mouse — a '
     + "selection that exists only as a gesture is not evidence:</p>";
  h += '<pre class="calls">' + esc(s.calls.join("\n")) + "</pre>";
  h += '<details><summary class="tiny muted">step log &mdash; every decision '
     + "taken on your behalf (" + s.steps.length + ")</summary>"
     + '<pre class="calls">' + s.steps.map(st =>
         esc(st.module + "." + st.action + ": " + st.outcome)).join("\n")
     + "</pre></details>";
  return h;
}

const num = (v, d) => (v === null ? "—" : v.toPrecision(d));
const exp = v => (v === null ? "—" : v.toExponential(4));

/* ------------------------------------------------------------- the wiring */

function bind(id, field, after) {
  const el = $(id);
  const kind = el.type === "checkbox" ? "change" : "input";
  el.addEventListener(kind, () => {
    S[field] = el.type === "checkbox" ? el.checked : el.value;
    if (after) after();
  });
}

$("godir").onclick = () => browse($("dirpath").value.trim());
$("dirpath").addEventListener("keydown", ev => {
  if (ev.key === "Enter") { ev.preventDefault(); browse($("dirpath").value.trim()); }
});
$("clearwindows").onclick = () => {
  S.windows = [];
  renderWindows();
  refreshRaw();
};
$("addseries").onclick = () => {
  S.series.push({ expr: "", label: "" });
  activeSeries = S.series.length - 1;
  renderSeries();
};
$("go").onclick = analyse;

bind("data_kind", "dataKind", () => { renderUnits(); refreshPrep(); });
bind("unit", "unit", refreshPrep);
bind("rate", "rate", refreshPrepSoon);
bind("time_column", "timeColumn", refreshPrep);
bind("time_format", "timeFormat", refreshPrep);
bind("outlier", "outlier");
bind("reason", "reason");
bind("join", "join");
bind("estimator", "estimator");
bind("taus", "taus");
bind("max_af", "maxAf");
bind("level", "level");
bind("wantci", "ci");
bind("together", "together");

for (const tab of document.querySelectorAll("#stages button"))
  tab.onclick = () => {
    S.stage = Number(tab.dataset.stage);
    for (const other of document.querySelectorAll("#stages button"))
      other.classList.toggle("on", other === tab);
    for (const section of document.querySelectorAll(".stage"))
      section.classList.toggle("on", section.id === "stage" + S.stage);
    if (S.stage === 1) { rawPlot.draw(); refreshRaw(); }
    if (S.stage === 2) refreshPrep();
    /* Stage 4 lives in bench.js, which loads after this file. Called through
     * a name check rather than imported, because these are classic scripts
     * sharing one global lexical scope and the page must still work if that
     * file is ever dropped. */
    if (S.stage === 4 && typeof benchEnter === "function") benchEnter();
  };

async function loadCapabilities() {
  const j = await call("/api/capabilities");
  if (!j) return;
  UNITS = j.units;
  $("estimator").innerHTML = j.estimators.map(e =>
    "<option" + (e === S.estimator ? " selected" : "") + ">" + esc(e)
    + "</option>").join("");
  $("taus").innerHTML = j.tau_modes.map(m =>
    "<option" + (m === S.taus ? " selected" : "") + ">" + esc(m)
    + "</option>").join("");
  renderUnits();
  if (typeof benchCapabilities === "function") benchCapabilities(j);
}

renderUnits();
renderSeries();
renderWindows();
loadCapabilities();
browse("");
