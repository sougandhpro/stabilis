"use strict";
/* Stage 4: the bench. Clocks, instruments, and which record is which pair.
 *
 * Loaded after app.js and sharing its global lexical scope, so `$`, `S`,
 * `call`, `esc`, `debounce`, `COLORS` and `createPlot` are the ones defined
 * there. Kept in its own file because stage 4 is a different application
 * from stages 1-3 -- it edits a description of an experiment rather than a
 * view of one file -- and merging them would have made app.js the largest
 * thing in the repository.
 *
 * `B.bench` is deliberately the exact shape of `bench.json`. Every edit
 * writes into it and the server round-trips it through the library's own
 * `store.to_dict` on each check, so what the Save button hands to the
 * browser is what `stabilis bench check` reads, not a second serializer
 * that has to be kept in agreement with the first.
 */

const B = {
  bench: { schema: 1, created: "", notes: "",
           sources: [], devices: [], measurements: [] },
  data: {},          /* measurement id -> the file/column assignment */
  sel: null,         /* which measurement's data editor is open */
  columns: {},       /* path -> labels, so a file is probed once */
  estimator: "oadev", taus: "octave", maxAf: "", force: false,
  verdict: null, entered: false,
};

let benchPlots = [];

/* Filled from /api/capabilities via app.js, so the engine stays the only
 * place that knows the estimator list and the bench schema version. */
function benchCapabilities(caps) {
  B.bench.schema = caps.bench_schema;
  $("benchest").innerHTML = caps.estimators.map(e =>
    "<option" + (e === B.estimator ? " selected" : "") + ">" + esc(e)
    + "</option>").join("");
  $("benchtaus").innerHTML = caps.tau_modes.map(m =>
    "<option" + (m === B.taus ? " selected" : "") + ">" + esc(m)
    + "</option>").join("");
}

/* First arrival gets a bench worth looking at rather than three empty
 * tables: the smallest setup that actually determines anything, which is a
 * triangle plus the back-to-back run that pins its channel. */
function benchEnter() {
  if (!B.entered) {
    B.entered = true;
    if (!B.bench.sources.length) seedBench();
  }
  renderBench();
}

function seedBench() {
  B.bench.sources = ["A", "B", "C"].map(id => source(id));
  B.bench.devices = [device("CNT1", ["CH1"])];
  B.bench.measurements = [
    measurement("M_AB", "A", "B", "CH1"),
    measurement("M_BC", "B", "C", "CH1"),
    measurement("M_AC", "A", "C", "CH1"),
  ];
}

const source = id => ({ id: id, name: "", kind: "", nominal_hz: null,
                        spec_adev: [], spec_floor: null, notes: "" });
const channel = id => ({ id: id, label: "", floor: null, notes: "" });
const device = (id, chans) => ({ id: id, model: "", serial: "",
                                 channels: chans.map(channel), notes: "" });
const measurement = (id, a, b, ch) => ({
  id: id, channel_id: ch, source_a: a, source_b: b, path: "", sha256: "",
  kind: "phase", tau0_s: null, epoch: "", reason: "" });

function allChannels() {
  const out = [];
  for (const d of B.bench.devices)
    for (const c of d.channels) out.push({ id: c.id, device: d.id });
  return out;
}

/* ------------------------------------------------------------- the clocks */

function renderSources() {
  $("sourcelist").innerHTML =
    '<tr><th>id</th><th>name</th><th>kind</th><th></th></tr>'
    + B.bench.sources.map((s, i) =>
      '<tr><td style="width:22%"><input data-s="id" data-i="' + i + '" value="'
      + esc(s.id) + '"></td>'
      + '<td><input data-s="name" data-i="' + i + '" value="' + esc(s.name)
      + '" placeholder="maser 1"></td>'
      + '<td style="width:26%"><input data-s="kind" data-i="' + i + '" value="'
      + esc(s.kind) + '" placeholder="H-maser"></td>'
      + '<td style="width:22px"><button class="drop" data-ds="' + i
      + '">&times;</button></td></tr>').join("");
  onEdit('#sourcelist input[data-s="name"], #sourcelist input[data-s="kind"]',
         el => { B.bench.sources[Number(el.dataset.i)][el.dataset.s] = el.value; });
  onCommit('#sourcelist input[data-s="id"]', el => {
    const s = B.bench.sources[Number(el.dataset.i)];
    const was = s.id;
    s.id = el.value;
    /* Renaming a clock has to follow into the measurements that name it, or
     * the bench stops referring to itself and the model refuses it on the
     * next check with an error about a source that does not exist. */
    if (was !== el.value)
      for (const m of B.bench.measurements) {
        if (m.source_a === was) m.source_a = el.value;
        if (m.source_b === was) m.source_b = el.value;
      }
  });
  onClick("#sourcelist button[data-ds]", el => {
    B.bench.sources.splice(Number(el.dataset.ds), 1);
    renderBench();
  });
}

/* -------------------------------------------------------- the instruments */

function renderDevices() {
  $("devicelist").innerHTML = B.bench.devices.map((d, i) =>
    '<div class="device"><div class="row">'
    + '<input data-d="id" data-i="' + i + '" value="' + esc(d.id) + '">'
    + '<input data-d="model" data-i="' + i + '" value="' + esc(d.model)
    + '" placeholder="model">'
    + '<button class="drop" data-dd="' + i + '">&times;</button></div>'
    + '<table>' + d.channels.map((c, k) =>
        '<tr><td style="width:34%"><input data-c="id" data-i="' + i
        + '" data-k="' + k + '" value="' + esc(c.id) + '"></td>'
        + '<td><input data-c="floor" data-i="' + i + '" data-k="' + k
        + '" value="' + (c.floor === null ? "" : esc(c.floor))
        + '" placeholder="floor variance — blank to solve for it"></td>'
        + '<td style="width:22px"><button class="drop" data-dc="' + i + '_' + k
        + '">&times;</button></td></tr>').join("")
    + '</table><button class="inline ghost" data-ac="' + i
    + '">+ channel</button></div>').join("");

  onEdit('#devicelist input[data-d="model"]', el => {
    B.bench.devices[Number(el.dataset.i)].model = el.value;
  });
  onCommit('#devicelist input[data-d="id"]', el => {
    B.bench.devices[Number(el.dataset.i)].id = el.value;
  });
  onEdit('#devicelist input[data-c="floor"]', el => {
    /* Blank means "solve for it", which is not the same statement as zero.
     * Zero is the classical three-cornered hat's silent assumption, and
     * writing it here is what makes that assumption visible. */
    const c = B.bench.devices[Number(el.dataset.i)]
                .channels[Number(el.dataset.k)];
    const text = el.value.trim();
    c.floor = text === "" ? null : Number(text);
  });
  onCommit('#devicelist input[data-c="id"]', el => {
    const c = B.bench.devices[Number(el.dataset.i)]
                .channels[Number(el.dataset.k)];
    const was = c.id;
    c.id = el.value;
    for (const m of B.bench.measurements)
      if (m.channel_id === was) m.channel_id = el.value;
  });
  onClick("#devicelist button[data-dd]", el => {
    B.bench.devices.splice(Number(el.dataset.dd), 1);
    renderBench();
  });
  onClick("#devicelist button[data-dc]", el => {
    const [i, k] = el.dataset.dc.split("_").map(Number);
    B.bench.devices[i].channels.splice(k, 1);
    renderBench();
  });
  onClick("#devicelist button[data-ac]", el => {
    const d = B.bench.devices[Number(el.dataset.ac)];
    d.channels.push(channel(d.id + "-" + (d.channels.length + 1)));
    renderBench();
  });
}

/* ------------------------------------------------------- the measurements */

function renderMeasurements() {
  const chans = allChannels();
  const pick = (name, value, options, i) =>
    '<select data-m="' + name + '" data-i="' + i + '">'
    + options.map(o => '<option' + (o === value ? " selected" : "") + ">"
                       + esc(o) + "</option>").join("") + "</select>";
  const ids = B.bench.sources.map(s => s.id);

  $("measlist").innerHTML =
    "<tr><th>id</th><th>A</th><th>B</th><th>channel</th><th>reads</th>"
    + "<th>data</th><th></th></tr>"
    + B.bench.measurements.map((m, i) => {
      const spec = B.data[m.id];
      const b2b = m.source_a === m.source_b;
      const has = spec && spec.path;
      return '<tr' + (B.sel === m.id ? ' class="sel"' : "") + ">"
        + '<td style="width:18%"><input data-m="id" data-i="' + i + '" value="'
        + esc(m.id) + '"></td>'
        + "<td>" + pick("source_a", m.source_a, ids, i) + "</td>"
        + "<td>" + pick("source_b", m.source_b, ids, i) + "</td>"
        + "<td>" + pick("channel_id", m.channel_id,
                        chans.map(c => c.id), i) + "</td>"
        + "<td>" + pick("kind", m.kind, ["phase", "frequency"], i) + "</td>"
        + '<td><button class="inline ghost" data-md="' + m.id + '">'
        + (has ? esc(spec.path + "  ·  " + (spec.expr || "no column"))
               : "assign…") + "</button>"
        + (b2b ? '<br><span class="tiny muted">back-to-back: measures '
                 + esc(m.channel_id) + ", not a clock</span>" : "") + "</td>"
        + '<td style="width:22px"><button class="drop" data-dm="' + i
        + '">&times;</button></td></tr>';
    }).join("");

  /* Every control on this row is one the rest of the stage reads back --
   * the id keys the data assignment, and the three pickers decide what the
   * graph looks like -- so all of them commit and redraw. */
  onCommit("#measlist input[data-m], #measlist select[data-m]", el => {
    const m = B.bench.measurements[Number(el.dataset.i)];
    const field = el.dataset.m;
    if (field === "id") {
      /* The file assignment is keyed by measurement id and has to move with
       * it, or renaming a row silently unassigns its data. */
      if (B.data[m.id]) { B.data[el.value] = B.data[m.id]; delete B.data[m.id]; }
      if (B.sel === m.id) B.sel = el.value;
    }
    m[field] = el.value;
  });
  onClick("#measlist button[data-dm]", el => {
    const m = B.bench.measurements[Number(el.dataset.dm)];
    delete B.data[m.id];
    if (B.sel === m.id) B.sel = null;
    B.bench.measurements.splice(Number(el.dataset.dm), 1);
    renderBench();
  });
  onClick("#measlist button[data-md]", el => {
    B.sel = B.sel === el.dataset.md ? null : el.dataset.md;
    renderBench();
  });
}

/* Ask the server what columns a path has, once per path. This is the same
 * `/api/open` stage 1 uses, so the file is read once and the answer is the
 * same one the record view was drawn from.
 *
 * **A path that failed is remembered as failed.** Recording `undefined`
 * would leave it looking un-probed, and since the reply re-renders and the
 * render probes again, a path that does not exist would fetch in a loop for
 * as long as the stage was open. `null` means asked-and-nothing-came-back,
 * which is also what the editor draws as "no columns yet". */
async function probe(spec) {
  const path = (spec.path || "").trim();
  if (!path || B.columns[path] !== undefined) return;
  B.columns[path] = null;
  const j = await call("/api/open", { path: path });
  B.columns[path] = j ? j.labels : null;
  renderBench();
}

/* ------------------------------------------------------------ the verdict */

function renderBench() {
  renderSources();
  renderDevices();
  renderMeasurements();
  renderDataEditor();
  checkBench();
}

const checkBench = debounce(async function () {
  const j = await call("/api/bench/check", { bench: B.bench });
  if (!j) { $("benchverdict").textContent = ""; return; }
  B.verdict = j;
  $("benchverdict").textContent = j.n_measurements
    ? j.report
    : "Nothing to determine yet — add a measurement.\n\nA bench with no "
      + "rows has no unknowns, so it is trivially consistent and says "
      + "nothing about any clock.";
  $("benchverdict").className = "calls "
    + (!j.n_measurements ? "" : j.identifiable ? "good" : "bad");

  let advice = "";
  /* A pair reachable by subtracting two records already taken is a
   * different instruction from a pair needing a counter and a fortnight, so
   * it is said first and said differently. This is the ordinary three-
   * cornered hat: two channels against the house clock, the third pair
   * formed in software. */
  if (j.derivable.length) {
    advice += '<p class="tiny">These need <b>no new data</b> — the record is '
      + "a subtraction of two you already have, and the common reference "
      + "cancels exactly:</p><ul class=\"tiny derived\">"
      + j.derivable.map(e =>
          "<li><b>" + esc(e[0] + " vs " + e[1]) + "</b> = <code>"
          + esc(e[3]) + " − " + esc(e[4]) + "</code>, both against "
          + esc(e[2]) + "</li>").join("")
      + "</ul><p class=\"tiny muted\">Add the row, point it at the two "
      + "columns and write the subtraction as its column expression. Two "
      + "conditions the graph cannot check for you: the records must "
      + "overlap in time on a common timebase, and a derived record carries "
      + "<b>both</b> channels' noise — so declare the floors (zero is the "
      + "classical hat's silent assumption) or give the derived row its own "
      + "channel whose floor is the sum. See <code>stabilis docs bench</code>."
      + "</p>";
  }
  if (j.suggestions.length) {
    const free = new Set(j.derivable.map(e => e[0] + " " + e[1]));
    const costly = j.suggestions.filter(p => !free.has(p[0] + " " + p[1]));
    if (costly.length)
      advice += '<p class="tiny warn">Needs new data — add any one of these '
        + "measurements: <b>"
        + costly.map(p => esc(p[0] + " vs " + p[1])).join(", ") + "</b>.</p>";
  }
  /* Reading the returned verdict, not a second opinion about it: every
   * component already has its odd cycle and unknowns still outnumber rank,
   * so what is missing is in the channel block rather than the clock graph.
   * The two documented cures are in `stabilis docs bench`. */
  else if (!j.identifiable && j.n_measurements && j.free_channels.length
           && j.bipartite.every(bip => !bip))
    advice += '<p class="tiny warn">The clock graph is fine — every group '
      + "contains an odd cycle. What is undetermined is a channel: "
      + "<b>" + j.free_channels.map(esc).join(", ") + "</b>. A back-to-back "
      + "run (one clock at both ends) measures a channel's floor from data, "
      + "and declaring the floor instead makes the classical hat's silent "
      + "assumption visible. See <code>stabilis docs bench</code>.</p>";
  if (j.unmeasured.length)
    advice += '<p class="tiny warn">Declared but never measured: <b>'
      + j.unmeasured.map(esc).join(", ") + "</b>. They have no column in the "
      + "solve, which is not an error — benches get written before data is "
      + "taken — but nothing will be said about them.</p>";
  if (j.fixed_channels.length)
    advice += '<p class="tiny muted">Assumed rather than measured: '
      + j.fixed_channels.map(esc).join(", ")
      + " — a declared floor is a constant, not an unknown.</p>";
  advice += j.warnings.map(w =>
    '<p class="tiny warn">' + esc(w) + "</p>").join("");
  $("benchadvice").innerHTML = advice;
}, 200);

/* -------------------------------------------------------------- the solve */

async function benchSolve() {
  const data = {};
  for (const m of B.bench.measurements) {
    const spec = B.data[m.id];
    if (!spec || !spec.path) continue;
    /* The same object the preview was drawn from, so what was on screen is
     * what gets solved -- windows, delimiter, timebase and all. */
    data[m.id] = dataRequest(m, spec);
  }
  $("benchgo").disabled = true;
  $("benchgo").textContent = "solving…";
  const j = await call("/api/bench/solve", {
    bench: B.bench, data: data, estimator: B.estimator, taus: B.taus,
    max_af: B.maxAf, allow_unidentifiable: B.force,
  });
  $("benchgo").disabled = false;
  $("benchgo").textContent = "Solve";
  if (j) renderSolution(j);
}

function renderSolution(j) {
  for (const p of benchPlots) p.destroy();
  benchPlots = [];
  const box = $("benchresults");
  box.innerHTML = "";

  const panel = document.createElement("div");
  panel.className = "panel";
  const curves = j.sources.concat(j.channels);
  panel.innerHTML =
    '<h2>the clocks, separated</h2>'
    + '<p class="tiny muted">' + esc(j.estimator) + " · "
    + j.sources.length + " clocks and " + j.channels.length
    + " channel(s), one non-negative solve per tau</p>"
    + j.warnings.map(w => '<p class="tiny warn">' + esc(w) + "</p>").join("")
    + (j.any_negative
       ? '<p class="tiny warn"><b>A variance went negative.</b> That is not '
         + "numerical noise, it is the measurement saying an independence "
         + "assumption is violated — the clocks are correlated, or one is "
         + "misattributed, or a channel is worse than believed. The "
         + "constrained answer below is still non-negative and still wrong "
         + "in the same place; the signed column is the diagnostic.</p>" : "")
    + j.notes.map(n => '<p class="tiny warn">' + esc(n) + "</p>").join("")
    + '<div class="plotbox tall"><canvas></canvas></div>'
    + '<div class="legend">' + curves.map((name, i) =>
        '<span><i style="background:' + COLORS[i % COLORS.length] + '"></i>'
        + esc(name) + (j.channels.indexOf(name) >= 0 ? " (channel)" : "")
        + "</span>").join("") + "</div>"
    + solutionTable(j)
    + '<p class="tiny muted">the same thing without the mouse:</p>'
    + '<pre class="calls">' + esc(j.calls.join("\n")) + "</pre>"
    + '<details><summary class="tiny muted">step log — every decision taken '
    + "on your behalf (" + j.steps.length + ")</summary><pre class=\"calls\">"
    + j.steps.map(st => esc(st.module + "." + st.action + ": " + st.outcome))
        .join("\n") + "</pre></details>";
  box.appendChild(panel);

  const plot = createPlot(panel.querySelector("canvas"), {
    xlog: true, ylog: true, selectable: false,
    xlabel: "averaging time tau (s)", ylabel: j.estimator,
    onHover: (x, y) => "tau " + x.toPrecision(4) + " s, " + y.toExponential(3),
  });
  plot.setSeries(curves.map((name, i) => ({
    kind: "line", x: j.tau_s, y: curveOf(j, i).dev,
    color: COLORS[i % COLORS.length], label: name,
  })));
  plot.autoscale();
  plot.draw();
  benchPlots.push(plot);
}

/* Sources and channels are separate namespaces in the solution, and a bench
 * is free to name a clock and a channel alike -- the model does not forbid
 * it. So a curve is found by its position in `sources ++ channels` rather
 * than by looking one name up in both maps and taking whichever answers. */
function curveOf(j, i) {
  const n = j.sources.length;
  const name = i < n ? j.sources[i] : j.channels[i - n];
  return i < n
    ? { dev: j.source_dev[name], signed: j.source_var_signed[name] }
    : { dev: j.channel_dev[name], signed: j.channel_var_signed[name] };
}

function solutionTable(j) {
  const cols = j.sources.concat(j.channels);
  let h = '<div class="scroll"><table class="numbers"><tr><th>tau (s)</th>'
    + cols.map(c => "<th>" + esc(c) + "</th>").join("")
    + "<th>residual</th></tr>";
  for (let i = 0; i < j.tau_s.length; i++) {
    h += "<tr><td>" + j.tau_s[i].toPrecision(5) + "</td>"
      + cols.map((_c, k) => {
          const curve = curveOf(j, k);
          const dev = curve.dev[i];
          return "<td" + (curve.signed[i] < 0 ? ' class="warn"' : "") + ">"
            + (dev === null ? "—" : dev.toExponential(4)) + "</td>";
        }).join("")
      + "<td>" + j.residual[i].toExponential(2) + "</td></tr>";
  }
  h += "</table></div>";
  if (j.any_negative) {
    h += '<p class="tiny muted">unconstrained variances, sign intact — '
       + "cells above are flagged where this went below zero:</p>"
       + '<div class="scroll"><table class="numbers"><tr><th>tau (s)</th>'
       + cols.map(c => "<th>" + esc(c) + "</th>").join("") + "</tr>";
    for (let i = 0; i < j.tau_s.length; i++) {
      h += "<tr><td>" + j.tau_s[i].toPrecision(5) + "</td>"
        + cols.map((_c, k) => {
            const v = curveOf(j, k).signed[i];
            return "<td" + (v < 0 ? ' class="warn"' : "") + ">"
              + v.toExponential(3) + "</td>";
          }).join("") + "</tr>";
    }
    h += "</table></div>";
  }
  h += '<p class="tiny muted">Prepared: ' + j.measurements.map(m =>
      esc(m.id) + " (" + m.n.toLocaleString() + " pts at "
      + m.rate_hz.toPrecision(6) + " Hz, " + esc(m.rate_note) + ")").join("; ")
    + "</p>";
  return h;
}

/* ----------------------------------------------------- opening and saving */

async function benchOpen() {
  const path = $("benchpath").value.trim();
  if (!path) { fail("name a bench.json under the root"); return; }
  const j = await call("/api/bench/load", { path: path });
  if (!j) return;
  B.bench = j.bench;
  B.data = {};
  B.sel = null;
  B.entered = true;
  /* A bench arriving from disk already names a file per measurement, so the
   * assignment starts half made -- path known, column still to say, because
   * `bench.json` does not record one and this page will not invent it. */
  for (const m of j.bench.measurements)
    if (m.path)
      B.data[m.id] = Object.assign(newSpec(m),
                                   { path: m.path, reason: m.reason });
  $("benchfile").innerHTML = "opened <b>" + esc(j.path) + "</b>"
    + (j.hash_problems.length
       ? '<br><span class="warn">' + j.hash_problems.map(esc).join("<br>")
         + "</span>"
       : "");
  renderBench();
}

function benchDownload() {
  /* The bench the *server* normalized on the last check, not the draft this
   * page happens to be holding -- so the file is what the library writes and
   * a bench the model would reject cannot be saved at all. */
  if (!B.verdict || !B.verdict.bench) {
    fail("the bench has not passed a check yet, so there is nothing "
         + "validated to save");
    return;
  }
  const text = JSON.stringify(B.verdict.bench, null, 2) + "\n";
  const url = URL.createObjectURL(new Blob([text], { type: "application/json" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = "bench.json";
  a.click();
  URL.revokeObjectURL(url);
  $("benchfile").textContent =
    "handed to your browser's downloads as bench.json";
}

/* ------------------------------------------------------------- the wiring */

/* One listener per control, re-attached on every render because the tables
 * are rebuilt wholesale. Three ways to attach one, and the difference is
 * entirely about what has to happen afterwards.
 *
 * `onEdit` is free text nothing else derives from -- a clock's name, an
 * instrument's model. Debounced, writes into the bench, re-runs the check,
 * and deliberately does **not** redraw: rebuilding the table under a field
 * being typed in takes the caret with it.
 *
 * `onCommit` is ids and dropdowns. The rest of the stage is built out of
 * these -- the measurement rows list the clock ids, the channel picker
 * lists the channel ids -- so a change has to redraw or the page starts
 * showing a select whose value is not among its own options. On `change`
 * rather than `input`, so the redraw happens when the field is left.
 */
function onClick(selector, fn) {
  for (const el of document.querySelectorAll(selector))
    el.onclick = () => fn(el);
}

function onEdit(selector, fn) {
  for (const el of document.querySelectorAll(selector))
    el.oninput = debounce(() => { fn(el); checkBench(); }, 250);
}

function onCommit(selector, fn) {
  for (const el of document.querySelectorAll(selector))
    el.onchange = () => { fn(el); renderBench(); };
}

$("addsource").onclick = () => {
  B.bench.sources.push(source("S" + (B.bench.sources.length + 1)));
  renderBench();
};
$("adddevice").onclick = () => {
  const n = B.bench.devices.length + 1;
  B.bench.devices.push(device("CNT" + n, ["CNT" + n + "-1"]));
  renderBench();
};
$("addmeas").onclick = () => {
  const ids = B.bench.sources.map(s => s.id);
  const chans = allChannels();
  if (ids.length < 1 || !chans.length) {
    fail("add at least one clock and one channel first");
    return;
  }
  B.bench.measurements.push(measurement(
    "M" + (B.bench.measurements.length + 1), ids[0],
    ids[Math.min(1, ids.length - 1)], chans[0].id));
  renderBench();
};
$("benchgo").onclick = benchSolve;
$("benchload").onclick = benchOpen;
$("benchsave").onclick = benchDownload;
$("benchest").onchange = () => { B.estimator = $("benchest").value; };
$("benchtaus").onchange = () => { B.taus = $("benchtaus").value; };
$("benchmaxaf").oninput = () => { B.maxAf = $("benchmaxaf").value; };
$("benchforce").onchange = () => { B.force = $("benchforce").checked; };
