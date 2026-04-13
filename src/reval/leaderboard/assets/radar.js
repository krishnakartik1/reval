/* REVAL — tiny SVG radar renderer for per-row score sparklines.
 *
 * No dependencies. Attached to window so Alpine x-init callbacks
 * can call it synchronously:
 *
 *   <svg class="radar-svg"
 *        x-init="renderRadar($el, [0.92, 0.81, 0.88, 0.85, 0.72])">
 *   </svg>
 *
 * Expects:
 *   svg:    an <svg> element. Will be cleared and populated.
 *   scores: array of 0..1 numbers (null treated as 0).
 *
 * Uses a 100x100 viewBox so callers can control size via CSS.
 */

(function (global) {
  "use strict";

  var NS = "http://www.w3.org/2000/svg";

  function el(name, attrs) {
    var node = document.createElementNS(NS, name);
    for (var k in attrs) {
      if (Object.prototype.hasOwnProperty.call(attrs, k)) {
        node.setAttribute(k, attrs[k]);
      }
    }
    return node;
  }

  function renderRadar(svg, scores) {
    if (!svg || !scores || scores.length < 3) return;

    var N = scores.length;
    var cx = 50, cy = 50, R = 42;

    // Clear any prior content (re-renders on data change)
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    svg.setAttribute("viewBox", "0 0 100 100");
    svg.setAttribute("xmlns", NS);
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-hidden", "true");

    // Axis lines (one per category) — hairline
    for (var i = 0; i < N; i++) {
      var angle = (Math.PI * 2 * i) / N - Math.PI / 2;
      var x = (cx + Math.cos(angle) * R).toFixed(2);
      var y = (cy + Math.sin(angle) * R).toFixed(2);
      svg.appendChild(el("line", {
        x1: cx, y1: cy, x2: x, y2: y, class: "radar-axis"
      }));
    }

    // Filled polygon of scores
    var pts = scores.map(function (s, idx) {
      var v = (s == null || isNaN(s)) ? 0 : Math.max(0, Math.min(1, s));
      var a = (Math.PI * 2 * idx) / N - Math.PI / 2;
      return (cx + Math.cos(a) * R * v).toFixed(2)
             + "," + (cy + Math.sin(a) * R * v).toFixed(2);
    }).join(" ");
    svg.appendChild(el("polygon", { points: pts, class: "radar-fill" }));

    // Centre hub dot
    svg.appendChild(el("circle", {
      cx: cx, cy: cy, r: 1.5, class: "radar-hub"
    }));
  }

  global.renderRadar = renderRadar;
})(window);
