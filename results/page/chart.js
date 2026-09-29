// Draws the two charts on the write-up page from the JSON in #data.
// Colours come from CSS custom properties so both themes work.
(function () {
  const D = JSON.parse(document.getElementById("data").textContent);
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs, parent, text) => {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (text != null) n.textContent = text;
    if (parent) parent.appendChild(n);
    return n;
  };

  function accuracy(svg) {
    const W = 420, H = 260, L = 40, R = 12, T = 18, B = 46;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.innerHTML = "";
    const y = (v) => T + (H - T - B) * (1 - v / 100);
    for (const t of [0, 25, 50, 75, 100]) {
      el("line", { x1: L, x2: W - R, y1: y(t), y2: y(t), stroke: css("--rule"), "stroke-width": 1 }, svg);
      el("text", { x: L - 8, y: y(t) + 4, "text-anchor": "end", class: "tick" }, svg, t + "%");
    }
    const groups = [["from memory", "closed_book"], ["with the sections", "retrieval"]];
    const models = [["base", "--muted-mark"], ["aligned", "--accent"]];
    const gw = (W - L - R) / groups.length, bw = 44;
    groups.forEach(([label, key], gi) => {
      const cx = L + gw * gi + gw / 2;
      models.forEach(([m, col], mi) => {
        const v = 100 * D.acc[`${key}/${m}`];
        const x = cx + (mi - 1) * (bw + 6) + 3;
        el("rect", { x, y: y(v), width: bw, height: y(0) - y(v), fill: css(col), rx: 2 }, svg);
        el("text", { x: x + bw / 2, y: y(v) - 6, "text-anchor": "middle", class: "val" }, svg, Math.round(v) + "%");
        el("text", { x: x + bw / 2, y: y(0) + 15, "text-anchor": "middle", class: "tick" }, svg, m);
      });
      el("text", { x: cx, y: H - 8, "text-anchor": "middle", class: "axis" }, svg, label);
    });
  }

  function filtering(svg) {
    const W = 460, H = 300, L = 44, R = 16, T = 16, B = 44;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.innerHTML = "";
    const x = (v) => L + (W - L - R) * v / 100, y = (v) => T + (H - T - B) * (1 - v / 100);
    for (const t of [0, 25, 50, 75, 100]) {
      el("line", { x1: L, x2: W - R, y1: y(t), y2: y(t), stroke: css("--rule"), "stroke-width": 1 }, svg);
      el("text", { x: L - 8, y: y(t) + 4, "text-anchor": "end", class: "tick" }, svg, t + "%");
      el("text", { x: x(t), y: H - 26, "text-anchor": "middle", class: "tick" }, svg, t + "%");
    }
    el("text", { x: (L + W - R) / 2, y: H - 6, "text-anchor": "middle", class: "axis" }, svg, "share of responses shown to the user");
    el("text", { x: 12, y: (T + H - B) / 2, "text-anchor": "middle", class: "axis", transform: `rotate(-90 12 ${(T + H - B) / 2})` }, svg, "shown answers that are wrong");
    // Nugen confidence sweep as a step line, highest threshold (left) to none (right)
    const pts = D.sweep.filter((p) => p.answered).sort((a, b) => a.coverage - b.coverage);
    let d = "";
    pts.forEach((p, i) => {
      const px = x(100 * p.coverage), py = y(100 * p.error);
      d += i ? ` H${px} V${py}` : `M${px} ${py}`;
    });
    el("path", { d, fill: "none", stroke: css("--accent"), "stroke-width": 2.2 }, svg);
    const last = pts[pts.length - 1];
    el("text", { x: x(100 * last.coverage) - 4, y: y(100 * last.error) - 10, "text-anchor": "end", class: "lab accent" }, svg, "Nugen confidence threshold");
    const dot = (p, col, label, dy) => {
      el("circle", { cx: x(100 * p.coverage), cy: y(100 * p.error), r: 6, fill: css(col), stroke: css("--paper"), "stroke-width": 2 }, svg);
      el("text", { x: x(100 * p.coverage) + 10, y: y(100 * p.error) + dy, class: "lab" }, svg, label);
    };
    dot(D.checker, "--ok", "checker alone", 4);
    if (D.nofilter) dot(D.nofilter, "--muted-mark", "no filter", 16);
    if (D.app) {
      const ax = x(100 * D.app.coverage), ay = y(100 * D.app.error);
      el("rect", { x: ax - 6, y: ay - 6, width: 12, height: 12, transform: `rotate(45 ${ax} ${ay})`, fill: css("--ink"), stroke: css("--paper"), "stroke-width": 2 }, svg);
      el("text", { x: ax + 11, y: ay - 14, class: "lab" }, svg, "checker + confidence ≥ 87");
      el("text", { x: ax + 11, y: ay - 1, class: "tick" }, svg, "cross-validated");
    }
  }

  const draw = () => {
    accuracy(document.getElementById("chart-acc"));
    filtering(document.getElementById("chart-filter"));
  };
  draw();
  if (window.matchMedia) window.matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", draw);
  new MutationObserver(draw).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
})();
