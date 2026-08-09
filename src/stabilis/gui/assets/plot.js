"use strict";
/* A canvas plotter, because matplotlib on the server made every zoom a
 * round trip through a PNG.
 *
 * Two shapes of data, which is all this page needs:
 *
 *   envelope  {i0, lo, hi}  one min/max pair per pixel column, drawn as a
 *                           vertical segment. This is how a two-million
 *                           point record is shown honestly at 900 px wide
 *                           -- a spike survives as a spike, where stride
 *                           sampling would have thrown it away.
 *   line      {x, y, lo, hi} points, optionally with asymmetric error bars,
 *                           which is what a chi-square interval on a
 *                           variance actually is.
 *
 * null means "no value here" and breaks the line rather than being drawn as
 * a zero. That is the gap representation all the way from the reader.
 *
 * Nothing here knows what a variance is. It is given numbers and axis
 * flags, and it draws them.
 */

function createPlot(canvas, opts) {
  const o = Object.assign({
    xlog: false, ylog: false, xlabel: "", ylabel: "",
    onSelect: null, onHover: null, onViewChange: null, selectable: true,
  }, opts || {});

  const pad = { l: 74, r: 14, t: 12, b: 40 };
  let series = [], bands = [], view = null, hover = null;
  let drag = null, size = { w: 0, h: 0 };

  const fx = v => (o.xlog ? Math.log10(v) : v);
  const gx = v => (o.xlog ? Math.pow(10, v) : v);
  const fy = v => (o.ylog ? Math.log10(v) : v);
  const gy = v => (o.ylog ? Math.pow(10, v) : v);

  const plotW = () => Math.max(1, size.w - pad.l - pad.r);
  const plotH = () => Math.max(1, size.h - pad.t - pad.b);
  const sx = v => pad.l + (fx(v) - view.x0) / (view.x1 - view.x0) * plotW();
  const sy = v => pad.t + (1 - (fy(v) - view.y0) / (view.y1 - view.y0)) * plotH();
  const ux = px => gx(view.x0 + (px - pad.l) / plotW() * (view.x1 - view.x0));
  const uy = py => gy(view.y0 + (1 - (py - pad.t) / plotH()) * (view.y1 - view.y0));

  function resize() {
    const ratio = window.devicePixelRatio || 1;
    const box = canvas.getBoundingClientRect();
    size = { w: Math.max(80, box.width), h: Math.max(60, box.height) };
    canvas.width = Math.round(size.w * ratio);
    canvas.height = Math.round(size.h * ratio);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    draw();
  }

  /* The bounds of everything plotted, in transformed (log or linear) space.
   * Non-positive values are skipped on a log axis rather than clamped: zero
   * is not a small number on a log scale, it is absent, and drawing it at
   * the axis floor would invent a measurement. */
  function extent() {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    const seeX = v => { if (v !== null && isFinite(v) && (!o.xlog || v > 0)) {
      const t = fx(v); if (t < x0) x0 = t; if (t > x1) x1 = t; } };
    const seeY = v => { if (v !== null && isFinite(v) && (!o.ylog || v > 0)) {
      const t = fy(v); if (t < y0) y0 = t; if (t > y1) y1 = t; } };
    for (const s of series) {
      if (s.hidden) continue;
      if (s.kind === "envelope") {
        for (let i = 0; i < s.i0.length; i++) {
          seeX(s.i0[i]); seeY(s.lo[i]); seeY(s.hi[i]);
        }
      } else {
        for (let i = 0; i < s.x.length; i++) {
          seeX(s.x[i]); seeY(s.y[i]);
          if (s.lo) { seeY(s.lo[i]); seeY(s.hi[i]); }
        }
      }
    }
    if (!isFinite(x0)) { x0 = 0; x1 = 1; }
    if (!isFinite(y0)) { y0 = 0; y1 = 1; }
    if (x1 - x0 < 1e-12) { x1 = x0 + 0.5; x0 -= 0.5; }
    if (y1 - y0 < 1e-12) { y1 = y0 + 0.5; y0 -= 0.5; }
    const my = (y1 - y0) * 0.06;
    return { x0: x0, x1: x1, y0: y0 - my, y1: y1 + my };
  }

  function autoscale() { view = extent(); draw(); }

  /* Tick positions that land on readable numbers: 1, 2 or 5 times a power of
   * ten for a linear axis, whole decades for a log one. */
  function linearTicks(lo, hi, want) {
    const raw = (hi - lo) / Math.max(1, want);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 5, 10].find(m => m * mag >= raw) * mag;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) out.push(v);
    return { ticks: out, step: step };
  }

  function logTicks(lo, hi) {
    const out = [], minor = [];
    for (let e = Math.floor(lo); e <= Math.ceil(hi); e++) {
      if (e >= lo && e <= hi) out.push(Math.pow(10, e));
      for (let m = 2; m < 10; m++) {
        const t = e + Math.log10(m);
        if (t >= lo && t <= hi) minor.push(Math.pow(10, t));
      }
    }
    return { ticks: out, minor: minor };
  }

  function fmt(v, step) {
    if (v === 0) return "0";
    const a = Math.abs(v);
    if (a >= 1e5 || a < 1e-3) return v.toExponential(a >= 1e5 ? 2 : 2);
    const dp = step ? Math.max(0, -Math.floor(Math.log10(step))) : 3;
    return v.toFixed(Math.min(dp, 6));
  }

  function css(name) {
    return getComputedStyle(canvas).getPropertyValue(name).trim();
  }

  function draw() {
    if (!view) { if (series.length) view = extent(); else return; }
    const ctx = canvas.getContext("2d");
    const ink = css("--ink") || "#222";
    const faint = css("--faint") || "rgba(128,128,128,.28)";
    ctx.clearRect(0, 0, size.w, size.h);

    const xt = o.xlog ? logTicks(view.x0, view.x1)
                      : linearTicks(view.x0, view.x1, Math.floor(plotW() / 90));
    const yt = o.ylog ? logTicks(view.y0, view.y1)
                      : linearTicks(view.y0, view.y1, Math.floor(plotH() / 44));

    ctx.save();
    ctx.beginPath();
    ctx.rect(pad.l, pad.t, plotW(), plotH());
    ctx.clip();

    ctx.strokeStyle = faint;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (const v of xt.minor || []) {
      const p = Math.round(sx(v)) + 0.5;
      ctx.moveTo(p, pad.t); ctx.lineTo(p, pad.t + plotH());
    }
    for (const v of yt.minor || []) {
      const p = Math.round(sy(v)) + 0.5;
      ctx.moveTo(pad.l, p); ctx.lineTo(pad.l + plotW(), p);
    }
    ctx.globalAlpha = 0.45; ctx.stroke(); ctx.globalAlpha = 1;

    ctx.beginPath();
    for (const v of xt.ticks) {
      const p = Math.round(sx(v)) + 0.5;
      ctx.moveTo(p, pad.t); ctx.lineTo(p, pad.t + plotH());
    }
    for (const v of yt.ticks) {
      const p = Math.round(sy(v)) + 0.5;
      ctx.moveTo(pad.l, p); ctx.lineTo(pad.l + plotW(), p);
    }
    ctx.stroke();

    for (const b of bands) {
      ctx.fillStyle = b.color || "rgba(255,150,0,.16)";
      const a = sx(b.x0), c = sx(b.x1);
      ctx.fillRect(Math.min(a, c), pad.t, Math.abs(c - a), plotH());
    }
    for (const s of series) if (!s.hidden) drawSeries(ctx, s);
    ctx.restore();

    ctx.strokeStyle = ink;
    ctx.lineWidth = 1;
    ctx.strokeRect(pad.l + 0.5, pad.t + 0.5, plotW(), plotH());

    ctx.fillStyle = ink;
    ctx.font = "11px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const v of xt.ticks)
      ctx.fillText(o.xlog ? fmt(v) : fmt(v, xt.step), sx(v), pad.t + plotH() + 6);
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const v of yt.ticks)
      ctx.fillText(o.ylog ? fmt(v) : fmt(v, yt.step), pad.l - 7, sy(v));

    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    if (o.xlabel) ctx.fillText(o.xlabel, pad.l + plotW() / 2, size.h - 2);
    if (o.ylabel) {
      ctx.save();
      ctx.translate(11, pad.t + plotH() / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textBaseline = "top";
      ctx.fillText(o.ylabel, 0, 0);
      ctx.restore();
    }

    if (drag && drag.mode === "select") {
      const a = Math.min(drag.from, drag.to), b = Math.max(drag.from, drag.to);
      ctx.fillStyle = "rgba(90,140,255,.22)";
      ctx.fillRect(a, pad.t, b - a, plotH());
      ctx.strokeStyle = "rgba(90,140,255,.9)";
      ctx.beginPath();
      ctx.moveTo(a + 0.5, pad.t); ctx.lineTo(a + 0.5, pad.t + plotH());
      ctx.moveTo(b + 0.5, pad.t); ctx.lineTo(b + 0.5, pad.t + plotH());
      ctx.stroke();
    }
    if (hover) drawHover(ctx, ink);
  }

  function drawSeries(ctx, s) {
    ctx.strokeStyle = s.color;
    ctx.fillStyle = s.color;
    ctx.lineWidth = 1;
    if (s.kind === "envelope") {
      ctx.beginPath();
      for (let i = 0; i < s.i0.length; i++) {
        const lo = s.lo[i], hi = s.hi[i];
        if (lo === null || hi === null) continue;
        const px = Math.round(sx(s.i0[i])) + 0.5;
        let a = sy(hi), b = sy(lo);
        if (b - a < 1) b = a + 1;
        ctx.moveTo(px, a); ctx.lineTo(px, b);
      }
      ctx.stroke();
      return;
    }
    if (s.lo) {
      ctx.beginPath();
      for (let i = 0; i < s.x.length; i++) {
        if (s.y[i] === null || s.lo[i] === null || s.hi[i] === null) continue;
        if (o.ylog && s.lo[i] <= 0) continue;
        const px = Math.round(sx(s.x[i])) + 0.5;
        ctx.moveTo(px, sy(s.lo[i])); ctx.lineTo(px, sy(s.hi[i]));
        ctx.moveTo(px - 3, sy(s.lo[i])); ctx.lineTo(px + 3, sy(s.lo[i]));
        ctx.moveTo(px - 3, sy(s.hi[i])); ctx.lineTo(px + 3, sy(s.hi[i]));
      }
      ctx.stroke();
    }
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    let pen = false;
    for (let i = 0; i < s.x.length; i++) {
      const y = s.y[i];
      if (y === null || !isFinite(y) || (o.ylog && y <= 0)) { pen = false; continue; }
      const px = sx(s.x[i]), py = sy(y);
      if (pen) ctx.lineTo(px, py); else ctx.moveTo(px, py);
      pen = true;
    }
    ctx.stroke();
    if (s.marker !== false) {
      for (let i = 0; i < s.x.length; i++) {
        const y = s.y[i];
        if (y === null || !isFinite(y) || (o.ylog && y <= 0)) continue;
        ctx.beginPath();
        ctx.arc(sx(s.x[i]), sy(y), 2.6, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  function drawHover(ctx, ink) {
    ctx.save();
    ctx.strokeStyle = "rgba(128,128,128,.7)";
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(Math.round(hover.px) + 0.5, pad.t);
    ctx.lineTo(Math.round(hover.px) + 0.5, pad.t + plotH());
    ctx.stroke();
    ctx.restore();
    if (!hover.text) return;
    ctx.font = "11px system-ui, sans-serif";
    const w = ctx.measureText(hover.text).width + 10;
    const left = Math.min(hover.px + 8, size.w - pad.r - w);
    ctx.fillStyle = css("--panel") || "#fff";
    ctx.globalAlpha = 0.92;
    ctx.fillRect(left, pad.t + 6, w, 18);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = "rgba(128,128,128,.5)";
    ctx.strokeRect(left + 0.5, pad.t + 6.5, w, 18);
    ctx.fillStyle = ink;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(hover.text, left + 5, pad.t + 15);
  }

  function inside(ev) {
    const box = canvas.getBoundingClientRect();
    return { px: ev.clientX - box.left, py: ev.clientY - box.top };
  }

  /* The drag handlers go on the window, because letting go outside the plot
   * has to finish the gesture rather than leave it stuck -- but they are
   * added on mousedown and removed on mouseup rather than living there for
   * the life of the plot. Every press of Analyse builds new result plots,
   * and permanent window listeners would leave the old ones redrawing
   * detached canvases on every mouse move for as long as the page is open. */
  function onDragMove(ev) {
    if (!drag) return;
    const p = inside(ev);
    drag.to = p.px;
    if (drag.mode === "pan") {
      const dx = (drag.to - drag.from) / plotW() * (drag.view.x1 - drag.view.x0);
      const dy = (p.py - drag.fromY) / plotH() * (drag.view.y1 - drag.view.y0);
      view = { x0: drag.view.x0 - dx, x1: drag.view.x1 - dx,
               y0: drag.view.y0 + dy, y1: drag.view.y1 + dy };
      if (o.onViewChange) o.onViewChange(getView());
    }
    draw();
  }

  function onDragEnd() {
    window.removeEventListener("mousemove", onDragMove);
    window.removeEventListener("mouseup", onDragEnd);
    if (!drag) return;
    const done = drag;
    drag = null;
    if (done.mode === "select" && o.onSelect) {
      if (Math.abs(done.to - done.from) > 3) {
        o.onSelect(ux(Math.min(done.from, done.to)),
                   ux(Math.max(done.from, done.to)));
      } else {
        o.onSelect(null, null);   // a click, not a drag
      }
    }
    draw();
  }

  canvas.addEventListener("mousedown", ev => {
    if (!view) return;
    ev.preventDefault();
    const p = inside(ev);
    const mode = (ev.shiftKey || ev.button === 1 || !o.selectable) ? "pan" : "select";
    drag = { mode: mode, from: p.px, to: p.px, fromY: p.py,
             view: Object.assign({}, view) };
    window.addEventListener("mousemove", onDragMove);
    window.addEventListener("mouseup", onDragEnd);
  });

  canvas.addEventListener("mousemove", ev => {
    if (drag || !view || !o.onHover) return;
    const p = inside(ev);
    if (p.px < pad.l || p.px > size.w - pad.r || p.py < pad.t
        || p.py > size.h - pad.b) {
      if (hover) { hover = null; draw(); }
      return;
    }
    hover = { px: p.px, text: o.onHover(ux(p.px), uy(p.py)) };
    draw();
  });

  canvas.addEventListener("mouseleave", () => {
    if (hover) { hover = null; draw(); }
  });

  canvas.addEventListener("wheel", ev => {
    if (!view) return;
    ev.preventDefault();
    const p = inside(ev);
    const k = Math.pow(1.0015, ev.deltaY);
    if (ev.shiftKey) {
      const at = view.y0 + (1 - (p.py - pad.t) / plotH()) * (view.y1 - view.y0);
      view = { x0: view.x0, x1: view.x1,
               y0: at + (view.y0 - at) * k, y1: at + (view.y1 - at) * k };
    } else {
      const at = view.x0 + (p.px - pad.l) / plotW() * (view.x1 - view.x0);
      view = { x0: at + (view.x0 - at) * k, x1: at + (view.x1 - at) * k,
               y0: view.y0, y1: view.y1 };
    }
    if (o.onViewChange) o.onViewChange(getView());
    draw();
  }, { passive: false });

  canvas.addEventListener("dblclick", () => {
    autoscale();
    if (o.onViewChange) o.onViewChange(getView());
  });

  function getView() {
    return view ? { x0: gx(view.x0), x1: gx(view.x1),
                    y0: gy(view.y0), y1: gy(view.y1) } : null;
  }

  const observer = new ResizeObserver(resize);
  observer.observe(canvas);
  resize();

  return {
    setSeries(list) { series = list || []; },
    setBands(list) { bands = list || []; },
    setLabels(x, y) { o.xlabel = x; o.ylabel = y; },
    setScales(xlog, ylog) { o.xlog = !!xlog; o.ylog = !!ylog; },
    setView(v) { if (v) view = { x0: fx(v.x0), x1: fx(v.x1),
                                 y0: fy(v.y0), y1: fy(v.y1) }; },
    getView: getView,
    width() { return plotW(); },
    autoscale: autoscale,
    draw: draw,
    destroy() {
      observer.disconnect();
      window.removeEventListener("mousemove", onDragMove);
      window.removeEventListener("mouseup", onDragEnd);
    },
  };
}
