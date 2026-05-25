// Dashboard logic. Reads docs/data/performance.json and renders stats,
// the cumulative return chart, open-portfolio cards, and the closed-
// portfolios table. No build step -- vanilla JS + Chart.js from CDN.

function fmtPct(x, digits = 2) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  const sign = x > 0 ? "+" : "";
  return `${sign}${(x * 100).toFixed(digits)}%`;
}

function fmtNum(x) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return new Intl.NumberFormat("en-US").format(x);
}

function classForReturn(x) {
  if (x === null || x === undefined || Number.isNaN(x)) return "";
  return x > 0 ? "positive" : x < 0 ? "negative" : "";
}

function fmtDate(iso) {
  if (!iso) return "—";
  return iso.slice(0, 10);
}

// === Tab switching (portfolio is the only internal one; news is external) ===
document.querySelectorAll(".tab-btn[data-section]").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn[data-section]").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    document.querySelectorAll(".tab-section").forEach(s => s.classList.remove("active"));
    document.getElementById(btn.dataset.section).classList.add("active");
  });
});

// === Main render ===
async function render() {
  let perf;
  try {
    const res = await fetch("data/performance.json", { cache: "no-store" });
    if (!res.ok) throw new Error(`fetch ${res.status}`);
    perf = await res.json();
  } catch (e) {
    document.querySelector("main .container").insertAdjacentHTML("afterbegin",
      `<div style="background:#fef2f2;border:1px solid #fca5a5;padding:1em;border-radius:6px;margin-bottom:1em;color:#7a1f1f">
       Couldn't load <code>data/performance.json</code>: ${e.message}.
       Run <code>python -m weekly_strategy.run_stage3 --skip-news --skip-reddit --skip-conviction --skip-macro</code>
       once to populate it.</div>`
    );
    return;
  }

  // === Stats ===
  const stats = perf.stats || {};
  setStat("stat-cum", stats.cumulative_return);
  setStat("stat-open-pnl", stats.current_open_pnl);
  document.getElementById("stat-n-closed").textContent = fmtNum(stats.n_closed ?? 0);
  document.getElementById("stat-n-open").textContent = fmtNum(stats.n_open ?? 0);

  // === Chart ===
  renderChart(perf.cumulative_series || []);

  // === Open portfolios ===
  renderOpen(perf.open_portfolios || []);

  // === Closed table ===
  renderClosed(perf.cumulative_series || [], perf.closed_portfolios || []);

  // === Latest report link ===
  setLatestReport(perf.open_portfolios || [], perf.closed_portfolios || []);

  // === All historical reports (grouped by year/month) ===
  renderAllReports(perf.open_portfolios || [], perf.closed_portfolios || []);

  // === Footer timestamp ===
  document.getElementById("last-updated").textContent =
    perf.generated_at ? perf.generated_at.replace("T", " ") + " UTC" : "—";
}

function setStat(id, val) {
  const el = document.getElementById(id);
  el.textContent = fmtPct(val);
  el.classList.remove("positive", "negative");
  if (val > 0) el.classList.add("positive");
  else if (val < 0) el.classList.add("negative");
}

function renderChart(series) {
  const ctx = document.getElementById("cum-chart");
  if (!ctx) return;
  if (series.length === 0) {
    ctx.replaceWith(Object.assign(document.createElement("div"), {
      className: "sub",
      textContent: "No closed portfolios yet — chart will populate after the first 5-day exit."
    }));
    return;
  }
  new Chart(ctx, {
    type: "line",
    data: {
      labels: series.map(p => p.exit_date),
      datasets: [{
        label: "Cumulative L/S return",
        data: series.map(p => p.cumulative_return),
        borderColor: "#2a5fa0",
        backgroundColor: "rgba(42,95,160,0.10)",
        fill: true,
        tension: 0.18,
        pointRadius: 3,
        pointHoverRadius: 5,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `Cumulative: ${fmtPct(ctx.parsed.y, 2)}`,
          },
        },
      },
      scales: {
        y: {
          ticks: { callback: (v) => fmtPct(v, 1) },
          grid: { color: "#e0e4ea" },
        },
        x: { grid: { display: false } },
      },
    },
  });
}

function renderOpen(open) {
  const target = document.getElementById("open-list");
  if (open.length === 0) {
    target.innerHTML = `<p class="sub">No open portfolios right now.</p>`;
    return;
  }
  target.innerHTML = open.map(p => {
    const ret = p.current_ls_return;
    const longHtml = (p.longs || []).map(pos => `
      <div class="pos">
        <span>${pos.ticker}</span>
        <span class="ret ${classForReturn(pos.current_return)}">${fmtPct(pos.current_return)}</span>
      </div>
    `).join("");
    const shortHtml = (p.shorts || []).map(pos => `
      <div class="pos">
        <span>${pos.ticker}</span>
        <span class="ret ${classForReturn(pos.current_return)}">${fmtPct(pos.current_return)}</span>
      </div>
    `).join("");
    return `
      <div class="open-card">
        <div class="head">
          <span class="date">${fmtDate(p.entry_date)}</span>
          <span class="days">held ${p.days_held ?? 0} trading days</span>
          <span class="return ${classForReturn(ret)}">${fmtPct(ret)}</span>
        </div>
        <div class="legs">
          <div class="leg">
            <div class="leg-label">Longs (${(p.longs || []).length})</div>
            ${longHtml || `<div class="sub">none</div>`}
          </div>
          <div class="leg">
            <div class="leg-label">Shorts (${(p.shorts || []).length})</div>
            ${shortHtml || `<div class="sub">none</div>`}
          </div>
        </div>
        <a class="report-link" href="portfolio/${p.entry_date}.html">Open full report →</a>
      </div>
    `;
  }).join("");
}

function renderClosed(cum, closedDetail) {
  const tbody = document.querySelector("#closed-table tbody");
  const empty = document.getElementById("closed-empty");
  if (cum.length === 0) {
    tbody.innerHTML = "";
    empty.style.display = "block";
    return;
  }
  empty.style.display = "none";
  // Join cum series (which has cumulative_return) with detail (long/short basket).
  const detailByEntry = Object.fromEntries(
    (closedDetail || []).map(p => [p.entry_date, p])
  );
  tbody.innerHTML = cum.slice().reverse().map(p => {
    const d = detailByEntry[p.entry_date] || {};
    const lb = d.long_basket_return;
    const sb = d.short_basket_return;
    return `
      <tr>
        <td>${fmtDate(p.entry_date)}</td>
        <td>${fmtDate(p.exit_date)}</td>
        <td>${d.days_held ?? "—"}</td>
        <td class="num ${classForReturn(lb)}">${fmtPct(lb)}</td>
        <td class="num ${classForReturn(sb)}">${fmtPct(sb)}</td>
        <td class="num ${classForReturn(p.ls_return)}">${fmtPct(p.ls_return)}</td>
        <td class="num ${classForReturn(p.cumulative_return)}">${fmtPct(p.cumulative_return)}</td>
      </tr>
    `;
  }).join("");
}

const MONTH_NAMES = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
];

function renderAllReports(open, closed) {
  const target = document.getElementById("all-reports");
  if (!target) return;

  const dates = Array.from(new Set(
    [...open, ...closed]
      .map(p => p.entry_date)
      .filter(d => typeof d === "string" && /^\d{4}-\d{2}-\d{2}/.test(d))
      .map(d => d.slice(0, 10))
  )).sort((a, b) => b.localeCompare(a));

  if (dates.length === 0) {
    target.innerHTML = `<p class="sub">No reports yet.</p>`;
    return;
  }

  // Group: year -> month -> [date, ...] (already sorted desc within)
  const byYear = new Map();
  for (const d of dates) {
    const [y, m] = d.split("-");
    if (!byYear.has(y)) byYear.set(y, new Map());
    const byMonth = byYear.get(y);
    if (!byMonth.has(m)) byMonth.set(m, []);
    byMonth.get(m).push(d);
  }

  const years = [...byYear.keys()]; // desc because dates were sorted desc
  const latestYear = years[0];

  const dateLink = (d) =>
    `<li><a href="portfolio/${d}.html">${d}</a></li>`;

  const monthBlock = (y, m, datesInMonth, isOpenDefault) => {
    const label = `${MONTH_NAMES[parseInt(m, 10) - 1]} ${y}`;
    return `
      <details class="report-month"${isOpenDefault ? " open" : ""}>
        <summary>${label} <span class="count">(${datesInMonth.length})</span></summary>
        <ul class="report-list">${datesInMonth.map(dateLink).join("")}</ul>
      </details>
    `;
  };

  // If everything fits in one year, skip the year-level fold.
  if (years.length === 1) {
    const y = years[0];
    const months = [...byYear.get(y).keys()]; // desc
    target.innerHTML = months
      .map((m, i) => monthBlock(y, m, byYear.get(y).get(m), i === 0))
      .join("");
    return;
  }

  target.innerHTML = years.map(y => {
    const months = [...byYear.get(y).keys()];
    const total = months.reduce((s, m) => s + byYear.get(y).get(m).length, 0);
    const isLatestYear = y === latestYear;
    const monthHtml = months
      .map((m, i) => monthBlock(y, m, byYear.get(y).get(m), isLatestYear && i === 0))
      .join("");
    return `
      <details class="report-year"${isLatestYear ? " open" : ""}>
        <summary>${y} <span class="count">(${total})</span></summary>
        <div class="report-year-body">${monthHtml}</div>
      </details>
    `;
  }).join("");
}

function setLatestReport(open, closed) {
  const link = document.getElementById("latest-report-link");
  const cand = open[0] || closed[0];
  if (!cand) {
    link.textContent = "No reports yet.";
    link.removeAttribute("href");
    return;
  }
  const d = cand.entry_date;
  link.href = `portfolio/${d}.html`;
  link.textContent = `Open full report for ${d} (per-position cards, news, fundamentals) →`;
}

render();
