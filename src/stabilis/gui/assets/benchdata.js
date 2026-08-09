"use strict";
/* Stage 4, the data half: pointing one measurement at one file, and looking
 * at what is in it.
 *
 * Loaded after bench.js and sharing its globals -- `B`, `renderBench`,
 * `probe` -- as well as app.js's `$`, `S`, `call`, `esc`, `debounce` and
 * `createPlot`.
 *
 * **Nothing here assumes the record is clean, because bench records are the
 * least clean data this program sees.** Every other stage looks at one file
 * the user just chose and is already looking at. A bench is six files from
 * six instruments, each with its own delimiter, its own idea of a timestamp,
 * its own dropouts and its own glitch at the moment somebody opened the rack
 * door. So this editor draws the record, in the units the library will
 * receive it in, and lets stretches be kept or dropped by dragging -- the
 * same min/max envelope stage 1 uses, so a single-sample spike in two
 * million points is still visible.
 *
 * The drawing endpoint is `/api/traces`, unchanged and shared with stages 1
 * and 2. A measurement's assignment is deliberately shaped as a request
 * object, so it can be posted as one.
 *
 * One honest limit, captioned on the page: the preview shows the record
 * *before* outlier rejection. Rejection has to happen on the full series
 * rather than on a decimated envelope, which makes it the solver's job --
 * so the count it removed comes back with the solution instead.
 */

let dataPlot = null;
let dataPlotFor = null;      /* which measurement the canvas belongs to */

/* Everything the drawing and solving endpoints need about one assignment.
 * `windows` is the list the server takes; `window` is not sent. */
function dataRequest(m, spec, extra) {
  return Object.assign({
    path: spec.path,
    series: [{ expr: spec.expr || "", label: m.id }],
    windows: spec.windows || [],
    data_kind: m.kind,
    unit: spec.unit,
    rate: spec.rate,
    time_column: spec.time_column,
    time_format: spec.time_format,
    delimiter: spec.delimiter,
    outlier_threshold: spec.outlier_threshold,
    reason: spec.reason,
  }, extra || {});
}

function newSpec(m) {
  return { path: "", expr: "", unit: m.kind === "phase" ? "s" : "fractional",
           rate: "", time_column: "", time_format: "", delimiter: "",
           windows: [], outlier_threshold: "", reason: "" };
}

function renderDataEditor() {
  const box = $("measdata");
  if (!B.sel) { box.innerHTML = ""; dropPlot(); return; }
  const m = B.bench.measurements.find(x => x.id === B.sel);
  if (!m) { B.sel = null; box.innerHTML = ""; dropPlot(); return; }
  const spec = B.data[m.id] || (B.data[m.id] = newSpec(m));
  const labels = B.columns[spec.path] || null;
  const units = UNITS[m.kind] || ["s"];
  const opt = (list, chosen) => list.map(o =>
    '<option value="' + esc(o[0]) + '"' + (o[0] === chosen ? " selected" : "")
    + ">" + esc(o[1]) + "</option>").join("");

  box.innerHTML = '<div class="panel inset"><h2>' + esc(m.id) + " — data</h2>"
    + '<p class="tiny muted">The bench says this is <b>' + esc(m.kind)
    + "</b>; that belongs to the experiment and is set on the row above. "
    + "Everything here describes the <i>file</i>, which the bench says "
    + "nothing about — and every measurement has its own, because every "
    + "instrument writes differently.</p>"

    + '<div class="row"><input id="mdpath" value="' + esc(spec.path)
    + '" placeholder="path under the root">'
    + '<button id="mdbrowse" class="inline ghost">browse</button>'
    + '<button id="mdhere" class="inline ghost">use the open file</button>'
    + "</div>"
    + '<div id="mdpicker"></div>'

    + '<div class="row">'
    + '<div style="flex:2"><label for="mdexpr">column expression</label>'
    + '<input id="mdexpr" value="' + esc(spec.expr)
    + '" placeholder="dt_ns, or NCS - VSL"></div>'
    + '<div><label for="mdunit">units in the file</label><select id="mdunit">'
    + opt(units.map(u => [u, u]), spec.unit) + "</select></div>"
    + '<div><label for="mddelim">delimiter</label><select id="mddelim">'
    + opt([["", "work it out"], [",", "comma"], ["\t", "tab"],
           [";", "semicolon"], [" ", "space"]], spec.delimiter)
    + "</select></div>"
    + "</div>"
    + (labels ? '<p class="tiny">columns: ' + labels.map(name =>
         '<a class="chip" data-col="' + esc(name) + '">' + esc(name)
         + "</a>").join("") + "</p>"
       : '<p class="tiny muted">Pick a file and its columns appear here to '
         + "click.</p>")

    + '<div class="row">'
    + '<div><label for="mdrate">rate (Hz)</label><input id="mdrate" value="'
    + esc(spec.rate) + '" placeholder="blank = from time column"></div>'
    + '<div><label for="mdtau0">tau0 (s), on the bench</label>'
    + '<input id="mdtau0" value="' + (m.tau0_s === null ? "" : esc(m.tau0_s))
    + '" placeholder="blank = use the rate"></div>'
    + '<div><label for="mdtcol">time column</label><input id="mdtcol" value="'
    + esc(spec.time_column) + '" placeholder="index"></div>'
    + '<div><label for="mdtfmt">what the timestamps are</label>'
    + '<select id="mdtfmt">'
    + opt([["", "work it out from the values"], ["elapsed", "elapsed seconds"],
           ["unix", "Unix seconds"], ["mjd", "MJD (days)"],
           ["iso", "ISO-8601 text"]], spec.time_format)
    + "</select></div>"
    + "</div>"
    + '<p id="mdrate_note" class="tiny muted"></p>'

    + '<h2>the record</h2>'
    + '<div class="plotbox"><canvas id="mdplot"></canvas></div>'
    + '<p class="tiny muted">Drag to keep a stretch; drag again to keep '
    + "another. Shift-drag pans, the wheel zooms, double-click fits. Two or "
    + "more stretches become <b>one record with the gaps between them "
    + "marked as gaps</b> — a measurement is one row of the bench, so it "
    + "cannot become two curves. Nothing is interpolated.</p>"
    + '<div id="mdwindows"></div>'
    + '<p id="mdstats" class="tiny"></p>'

    + '<div class="row">'
    + '<div><label for="mdout">outlier threshold (MAD sigmas)</label>'
    + '<input id="mdout" value="' + esc(spec.outlier_threshold)
    + '" placeholder="blank = keep every point"></div>'
    + '<div style="flex:2"><label for="mdreason">reason (goes in the record)'
    + '</label><input id="mdreason" value="' + esc(spec.reason)
    + '" placeholder="after warm-up, cable touched 14:05"></div>'
    + "</div>"
    + '<p class="tiny muted">The threshold is applied when you solve, not '
    + "in the picture above — rejection needs the whole series, not the "
    + "envelope drawn here — and the solution reports how many points it "
    + "took. tau0 on the bench wins over the rate here, because it is the "
    + "field <code>bench.json</code> carries to the command line.</p></div>";

  wireDataEditor(m, spec, box);
  probe(spec);
  refreshDataPlot(m, spec);
}

function wireDataEditor(m, spec, box) {
  const field = (id, key) => {
    const el = $(id);
    if (el) el.oninput = debounce(() => {
      spec[key] = el.value;
      if (key === "expr" || key === "rate") refreshDataPlot(m, spec);
    }, 250);
  };
  field("mdexpr", "expr");
  field("mdrate", "rate");
  field("mdtcol", "time_column");
  field("mdout", "outlier_threshold");
  field("mdreason", "reason");
  const reread = (id, key) => {
    $(id).onchange = () => { spec[key] = $(id).value; renderBench(); };
  };
  reread("mdunit", "unit");
  reread("mdtfmt", "time_format");
  /* A delimiter changes how the file parses, so the cached table keyed on it
   * is a different table and the columns have to be probed again. */
  $("mddelim").onchange = () => {
    spec.delimiter = $("mddelim").value;
    delete B.columns[spec.path];
    renderBench();
  };
  $("mdtau0").oninput = debounce(() => {
    const text = $("mdtau0").value.trim();
    m.tau0_s = text === "" ? null : Number(text);
  }, 250);
  /* Committed on blur or Enter: reading a new path redraws this editor, and
   * redrawing the field being typed in takes the caret with it. */
  $("mdpath").onchange = () => {
    spec.path = $("mdpath").value.trim();
    spec.windows = [];
    renderBench();
  };
  $("mdhere").onclick = () => {
    if (!S.path) { fail("no file is open in stage 1"); return; }
    spec.path = S.path;
    spec.windows = [];
    renderBench();
  };
  $("mdbrowse").onclick = () => browseFor(spec, folderOf(spec.path));
  for (const a of box.querySelectorAll("a.chip"))
    a.onclick = () => {
      const el = $("mdexpr");
      const so_far = el.value.replace(/\s+$/, "");
      el.value = (!so_far || /[-+*/(]$/.test(so_far))
        ? so_far + (so_far ? " " : "") + a.dataset.col
        : so_far + " - " + a.dataset.col;
      spec.expr = el.value;
      el.focus();
      refreshDataPlot(m, spec);
    };
  renderWindowChips(m, spec);
}

/* ------------------------------------------------------------ the picker */

const folderOf = path => {
  const at = String(path || "").lastIndexOf("/");
  return at < 0 ? "" : path.slice(0, at);
};

/* A list rather than the sidebar, because the sidebar drives stage 1 and a
 * bench row is usually a different file in a different folder from whatever
 * is open there. Six counters, six directories. */
async function browseFor(spec, dir) {
  const j = await call("/api/files?dir=" + encodeURIComponent(dir || ""));
  if (!j) return;
  const box = $("mdpicker");
  if (!box) return;
  const rows = [];
  if (!j.at_root) rows.push(["../", () => browseFor(spec, j.parent), true]);
  for (const d of j.dirs)
    rows.push([d + "/", () => browseFor(spec, j.here ? j.here + "/" + d : d),
               true]);
  for (const f of j.files)
    rows.push([f.name, () => {
      spec.path = j.here ? j.here + "/" + f.name : f.name;
      spec.windows = [];
      renderBench();
    }, false]);

  box.innerHTML = '<div class="picker"></div>';
  const list = box.firstChild;
  if (!rows.length) {
    list.innerHTML = '<div class="empty">no folders or data files here</div>';
    return;
  }
  for (const [text, fn, isDir] of rows) {
    const el = document.createElement("div");
    el.textContent = text;
    if (isDir) el.className = "dir";
    el.onclick = fn;
    list.appendChild(el);
  }
}

/* ----------------------------------------------------- the record itself */

function dropPlot() {
  if (dataPlot) dataPlot.destroy();
  dataPlot = null;
  dataPlotFor = null;
}

function renderWindowChips(m, spec) {
  const box = $("mdwindows");
  if (!box) return;
  if (!spec.windows.length) {
    box.innerHTML = '<span class="tiny muted">No windows — the whole record '
      + "is analysed.</span>";
    return;
  }
  box.innerHTML = spec.windows.map((w, i) =>
    '<span class="window">' + w[0].toLocaleString() + " – "
    + w[1].toLocaleString() + ' <button class="drop" data-mw="' + i
    + '">&times;</button></span>').join("")
    + ' <button id="mdclearw" class="inline ghost">clear every window</button>';
  for (const b of box.querySelectorAll("button[data-mw]"))
    b.onclick = () => {
      spec.windows.splice(Number(b.dataset.mw), 1);
      renderWindowChips(m, spec);
      refreshDataPlot(m, spec);
    };
  $("mdclearw").onclick = () => {
    spec.windows = [];
    renderWindowChips(m, spec);
    refreshDataPlot(m, spec);
  };
}

async function refreshDataPlot(m, spec) {
  const canvas = $("mdplot");
  if (!canvas) return;
  if (!spec.path || !(spec.expr || "").trim()) {
    dropPlot();
    $("mdstats").innerHTML = '<span class="muted">Give a file and a column '
      + "and the record is drawn here.</span>";
    return;
  }
  if (dataPlotFor !== m.id || !dataPlot) {
    dropPlot();
    dataPlot = createPlot(canvas, {
      xlabel: "sample index",
      ylabel: m.kind === "phase" ? "phase x(t) (s)" : "fractional frequency",
      onSelect: (a, b) => {
        if (a === null) return;
        const lo = Math.max(0, Math.round(Math.min(a, b)));
        const hi = Math.round(Math.max(a, b));
        if (hi - lo < 2) return;
        spec.windows.push([lo, hi]);
        renderWindowChips(m, spec);
        refreshDataPlot(m, spec);
      },
      onViewChange: () => refreshDataPlotSoon(m, spec),
      onHover: (x, y) =>
        Math.round(x).toLocaleString() + ", " + y.toPrecision(6),
    });
    dataPlotFor = m.id;
    dataPlot.autoscale();
  }

  const view = dataPlot.getView();
  const j = await call("/api/traces", dataRequest(m, spec, {
    scaled: true,
    view: [Math.floor(view.x0), Math.ceil(view.x1)],
    target: Math.max(200, Math.round(dataPlot.width() * 2)),
  }));
  if (!j) { $("mdstats").innerHTML = ""; return; }
  const s = j.series[0];
  dataPlot.setSeries([{ kind: s.kind, i0: s.i0, lo: s.lo, hi: s.hi,
                        x: s.i0, y: s.lo, color: COLORS[0], label: s.label }]);
  dataPlot.setBands(spec.windows.map(w => ({ x0: w[0], x1: w[1] })));
  if (fitFor !== fingerprint(m, spec)) {
    fitFor = fingerprint(m, spec);
    dataPlot.autoscale();
  }
  dataPlot.draw();

  $("mdrate_note").textContent = j.rate_note
    + (j.jitter !== null && j.jitter !== undefined
       ? "; timetag spread " + (j.jitter * 100).toFixed(3)
         + "% of the median interval" : "");
  $("mdrate_note").className =
    j.jitter > 1e-3 ? "tiny warn" : "tiny muted";
  $("mdstats").innerHTML =
    "<b>" + s.stats.n.toLocaleString() + "</b> points selected of "
    + j.n.toLocaleString() + ", <b>" + s.stats.n_gaps.toLocaleString()
    + "</b> gaps"
    + (s.divide_by_zero ? ', <span class="warn">' + s.divide_by_zero
       + " became gaps (zero divisor)</span>" : "")
    + (j.defects ? '<br><span class="warn">tolerated while reading: '
       + esc(j.defects) + "</span>" : "");
}

/* Refit the axes only when what is drawn actually changed, so zooming in to
 * look at a glitch is not undone by the next keystroke. */
let fitFor = null;
const fingerprint = (m, spec) => JSON.stringify(
  [m.id, spec.path, spec.expr, spec.unit, spec.delimiter, spec.windows]);

const refreshDataPlotSoon = debounce(refreshDataPlot, 130);
