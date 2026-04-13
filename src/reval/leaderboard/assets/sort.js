/* Client-side sort + filter for the REVAL leaderboard.
 *
 * No framework, no build step. ~80 lines of vanilla JS. All data is
 * carried in data-* attributes on the <tr> elements so there's no
 * round-trip to the JSON file.
 */

(function () {
  "use strict";

  const table = document.getElementById("leaderboard-table");
  if (!table) return;

  const tbody = table.tBodies[0];
  const rows = Array.from(tbody.rows);

  // ── Sorting ─────────────────────────────────────────────────────
  const headers = table.querySelectorAll("th.sortable");
  let currentSort = { key: "overall_score", dir: "desc" };

  function sortBy(key) {
    if (currentSort.key === key) {
      currentSort.dir = currentSort.dir === "asc" ? "desc" : "asc";
    } else {
      currentSort = { key: key, dir: "desc" };
    }

    const sorted = rows.slice().sort((a, b) => {
      const av = a.dataset[key] != null ? a.dataset[key] : "";
      const bv = b.dataset[key] != null ? b.dataset[key] : "";
      // Try numeric compare first
      const an = parseFloat(av);
      const bn = parseFloat(bv);
      let cmp;
      if (!isNaN(an) && !isNaN(bn)) {
        cmp = an - bn;
      } else {
        cmp = String(av).localeCompare(String(bv));
      }
      return currentSort.dir === "asc" ? cmp : -cmp;
    });

    // Re-append rows in sorted order
    sorted.forEach((row) => tbody.appendChild(row));

    // Update header indicators
    headers.forEach((h) => {
      const hKey = h.dataset.sortKey;
      const label = h.textContent.replace(/[↑↓]\s*$/, "").trim();
      h.textContent =
        hKey === currentSort.key
          ? `${label} ${currentSort.dir === "asc" ? "↑" : "↓"}`
          : label;
    });
  }

  headers.forEach((h) => {
    h.addEventListener("click", () => sortBy(h.dataset.sortKey));
  });

  // ── Filtering ───────────────────────────────────────────────────
  const filters = document.querySelectorAll("[data-filter]");
  const rowCount = document.getElementById("row-count");

  function applyFilters() {
    let visible = 0;
    rows.forEach((row) => {
      let match = true;
      filters.forEach((f) => {
        const key = f.dataset.filter;
        const value = f.value;
        if (!value) return;
        const rowValue =
          row.dataset[key] || row.dataset["model_" + key] || "";
        if (rowValue !== value) {
          match = false;
        }
      });
      if (match) {
        row.removeAttribute("data-hidden");
        visible += 1;
      } else {
        row.setAttribute("data-hidden", "true");
      }
    });
    if (rowCount) rowCount.textContent = visible;
  }

  filters.forEach((f) => f.addEventListener("change", applyFilters));

  // Initial sort: overall score descending
  sortBy("overall_score");
})();
