"use strict";

const REFRESH_MS = 2000;
const FULL_RELOAD_MS = 300000;
const MAX_POINTS = 1000;
const FLEET_ROWS = 100;
const PALETTE = ["--s0", "--s1", "--s2", "--s3", "--s4", "--s5", "--s6", "--s7"];
const page = document.getElementById("page");
const refresh = document.getElementById("refresh");
const params = new URLSearchParams(location.search);

const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
));
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const query = (values) => {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(values)) {
    for (const item of Array.isArray(value) ? value : [value]) {
      if (item !== null && item !== undefined) search.append(key, item);
    }
  }
  return search.toString();
};

async function getJson(path, values = {}) {
  const response = await fetch(`${path}?${query(values)}`, { cache: "no-store" });
  if (!response.ok) {
    const error = new Error(`${path}: ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

function number(value, digits = 2) {
  if (value === null || value === undefined) return "–";
  const size = Math.abs(value);
  if (size !== 0 && (size >= 1e6 || size < 1e-3)) return value.toExponential(2);
  return value.toLocaleString(undefined, { maximumFractionDigits: size >= 100 ? 0 : digits });
}

function duration(seconds) {
  if (seconds === null || seconds === undefined) return "–";
  const total = Math.max(0, Math.round(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  if (hours) return `${hours}h ${String(minutes).padStart(2, "0")}m`;
  if (minutes) return `${minutes}m ${String(total % 60).padStart(2, "0")}s`;
  return `${total}s`;
}

function age(timestamp, now) {
  return timestamp ? `${duration(now - timestamp)} ago` : "–";
}

const pill = (status) => `<span class="pill st-${esc(status)}">${esc(status)}</span>`;
const shortId = (id) => String(id ?? "").slice(-8);
const table = (head, rows) => `<div class="table-wrap"><table><thead><tr>${head.join("")}</tr></thead><tbody>${rows.join("")}</tbody></table></div>`;
const th = (label, cls = "") => `<th class="${cls}">${esc(label)}</th>`;
const td = (value, cls = "") => `<td class="${cls}">${value}</td>`;

function progressCell(progress) {
  if (!progress) return "–";
  const { completed, total } = progress;
  if (total === null || total === undefined) return number(completed, 0);
  const share = total > 0 ? Math.min(100, (100 * completed) / total) : 0;
  return `<span class="bar"><i style="width:${share}%"></i></span>${number(completed, 0)}/${number(total, 0)}`;
}

// A refresh must not reset what the viewer did: table scroll and collapsed details.
function setHtml(target, html) {
  if (target.dataset.html === html) return;
  const scroll = [...target.querySelectorAll(".table-wrap")].map((wrap) => wrap.scrollLeft);
  const open = new Map([...target.querySelectorAll("details")].map((item) => [item.querySelector("summary").textContent, item.open]));
  target.innerHTML = html;
  target.dataset.html = html;
  target.querySelectorAll(".table-wrap").forEach((wrap, index) => {
    wrap.scrollLeft = scroll[index] ?? 0;
  });
  target.querySelectorAll("details").forEach((item) => {
    item.open = open.get(item.querySelector("summary").textContent) ?? item.open;
  });
}

function setRefresh(ok, detail = "") {
  refresh.textContent = ok ? `Updated ${new Date().toLocaleTimeString()}` : `Refresh failed ${detail}`;
}

function showMissing(what) {
  page.dataset.html = "";
  page.innerHTML = `<p class="muted">${esc(what)} was not found. <a href="./">Back to all runs</a></p>`;
}

/* Fleet page */

async function fleetPage() {
  let filter = params.get("filter") ?? "";
  page.innerHTML = `<div class="heading"><h1>Runs</h1></div>
    <div class="controls"><input id="filter" type="search" placeholder="Filter by run, group, entry or status" aria-label="Filter runs"></div>
    <div id="fleet"><p class="muted">Scanning entries…</p></div>`;
  const input = document.getElementById("filter");
  input.value = filter;
  let latest = null;
  const paint = () => {
    if (latest) renderFleet(latest, filter);
  };
  input.addEventListener("input", () => {
    filter = input.value;
    paint();
  });
  const tick = async () => {
    try {
      latest = await getJson("api/fleet");
      paint();
      setRefresh(true);
    } catch (error) {
      setRefresh(false, error.message);
    }
  };
  await tick();
  setInterval(() => { if (!document.hidden) tick(); }, REFRESH_MS);
}

function fleetRows(fleet) {
  const rows = [];
  for (const entry of fleet.entries) {
    for (const group of entry.groups) {
      rows.push({
        kind: "group",
        name: group.group_id,
        entry: entry.id,
        status: group.active_member ? "running" : group.failed_count ? "failed" : group.completed_count === group.total_count ? "completed" : "pending",
        detail: `${group.completed_count} done · ${group.failed_count} failed · ${group.total_count} total`,
        progress: { completed: group.completed_count + group.failed_count, total: group.total_count },
        eta: group.eta_seconds,
        updated: group.finished_at ?? group.updated_at,
        href: `?${query({ entry: entry.id, group: group.group_id })}`,
      });
    }
    for (const run of entry.loose_runs) {
      rows.push({
        kind: "run",
        name: run.run_name,
        entry: entry.id,
        status: run.status,
        detail: "",
        progress: run.progress,
        eta: run.progress?.eta_seconds,
        rate: run.progress?.throughput,
        updated: run.finished_at ?? run.updated_at,
        href: `?${query({ entry: entry.id, run: run.run_name })}`,
      });
    }
  }
  const active = (row) => (row.status === "running" || row.status === "stale" ? 0 : 1);
  return rows.sort((a, b) => active(a) - active(b) || (b.updated ?? 0) - (a.updated ?? 0));
}

function renderFleet(fleet, filter) {
  const target = document.getElementById("fleet");
  const needle = filter.trim().toLowerCase();
  const all = fleetRows(fleet);
  const rows = needle
    ? all.filter((row) => `${row.name} ${row.entry} ${row.status} ${row.kind}`.toLowerCase().includes(needle))
    : all;
  if (!rows.length) {
    target.dataset.html = "";
    target.innerHTML = `<p class="muted">${!fleet.ready ? "Scanning entries…" : needle ? "No runs match this filter." : "No runs found below the configured roots."}</p>`;
    return;
  }
  const count = (status) => all.filter((row) => row.status === status).length;
  const hidden = rows.length - FLEET_ROWS;
  const summary = `<p class="meta">${all.length} runs and groups · ${count("running")} running · ${count("stale")} stale · ${count("failed")} failed${hidden > 0 ? ` · showing the ${FLEET_ROWS} most recent, filter to see others` : ""}</p>`;
  setHtml(target, summary + table(
    [th("Name"), th("Status"), th("Progress"), th("Rate", "num"), th("ETA", "num"), th("Updated"), th("Entry")],
    rows.slice(0, FLEET_ROWS).map((row) => `<tr>
      ${td(`<a href="${esc(row.href)}">${esc(row.name)}</a>${row.kind === "group" ? ' <span class="muted">group</span>' : ""}`)}
      ${td(pill(row.status))}
      ${td(progressCell(row.progress))}
      ${td(row.rate ? `${number(row.rate)} b/s` : "–", "num")}
      ${td(duration(row.eta), "num")}
      ${td(age(row.updated, fleet.time))}
      ${td(esc(row.entry), "muted")}
    </tr>`),
  ));
}

/* Group page */

async function groupPage(entry, group) {
  const tick = async () => {
    try {
      const data = await getJson("api/group", { entry, group });
      renderGroup(data);
      setRefresh(true);
    } catch (error) {
      if (error.status === 404) showMissing(`Group ${group}`);
      else setRefresh(false, error.message);
    }
  };
  await tick();
  setInterval(() => { if (!document.hidden) tick(); }, REFRESH_MS);
}

function renderGroup(data) {
  const now = Date.now() / 1000;
  setHtml(page, `<div class="heading"><h1>${esc(data.group_id)}</h1>${data.terminal_status ? pill(data.terminal_status.replace("group_", "")) : ""}</div>
    <p class="meta"><a href="./">All runs</a> · ${esc(data.entry)} · ${data.completed_count} done · ${data.failed_count} failed · ${data.total_count} total${data.eta_seconds ? ` · ETA ${duration(data.eta_seconds)}` : ""}</p>
    <h2>MEMBERS</h2>` + table(
    [th("Run"), th("Status"), th("Step"), th("Progress"), th("Rate", "num"), th("ETA", "num"), th("Updated")],
    data.members.map((member) => `<tr>
      ${td(`<a href="?${esc(query({ entry: data.entry, run: member.run_name }))}">${esc(member.run_name)}</a>`)}
      ${td(pill(member.status))}
      ${td(esc(member.active_step ?? "–"))}
      ${td(progressCell(member.progress))}
      ${td(member.progress?.throughput ? `${number(member.progress.throughput)} b/s` : "–", "num")}
      ${td(duration(member.progress?.eta_seconds), "num")}
      ${td(age(member.updated_at, now))}
    </tr>`),
  ) + warningsHtml(data.warnings));
}

function warningsHtml(warnings) {
  return warnings?.length
    ? `<h2>WARNINGS</h2>${warnings.map((warning) => `<p class="notice">${esc(warning)}</p>`).join("")}`
    : "";
}

/* Run page */

const stageColor = (() => {
  const known = new Map();
  return (stage) => {
    if (!known.has(stage)) known.set(stage, css(PALETTE[known.size % PALETTE.length]));
    return known.get(stage);
  };
})();

class RunPage {
  constructor(entry, run, execution) {
    Object.assign(this, { entry, run, execution });
    this.windowSeconds = 300;
    this.charts = new Map();
    this.series = new Map();
    this.stages = { t: [], end: [], stage: [], lane: [], producer: [] };
    this.busy = false;
    this.loadedAt = 0;
    this.data = null;
    page.innerHTML = `<div id="head"><p class="muted">Loading ${esc(run)}…</p></div>
      <div id="tiles" class="tiles"></div>
      <div id="tasks"></div>
      <div id="resources"></div>
      <div id="stage-table"></div>
      <div id="metrics"></div>
      <div id="items"></div>
      <div id="details"></div>`;
    addEventListener("resize", () => this.resizeCharts());
  }

  scope() {
    return { entry: this.entry, run: this.run, execution: this.execution };
  }

  // Ticks never overlap: a reload must not interleave with a refresh in flight.
  async refresh() {
    if (this.busy) return;
    this.busy = true;
    try {
      await this.tick();
    } finally {
      this.busy = false;
    }
  }

  async tick() {
    try {
      const data = await getJson("api/run", this.scope());
      // A run that is no longer reporting has nothing recent, so it opens on its whole history.
      if (!this.data && !this.isLive(data)) this.windowSeconds = 0;
      this.data = data;
      await this.loadSeries(data);
      this.render(data);
      setRefresh(true);
    } catch (error) {
      setRefresh(false, error.message);
      // A run can be missing only for now, such as before the server's first sweep.
      if (error.status === 404 && !this.data) {
        document.getElementById("head").innerHTML = `<p class="muted">Run ${esc(this.run)} was not found. Still checking. <a href="./">Back to all runs</a></p>`;
        document.getElementById("head").dataset.signature = "";
      }
    }
  }

  isLive(data) {
    return ["pending", "running"].includes(data.status)
      && !data.producers.some((producer) => producer.status === "stale");
  }

  // The newest moment worth showing: now while live, otherwise the last event.
  anchor(data) {
    return this.isLive(data) ? data.time : data.updated_at;
  }

  rangeSince(data) {
    return this.windowSeconds ? this.anchor(data) - this.windowSeconds : 0;
  }

  async loadSeries(data) {
    const reload = data.time * 1000 - this.loadedAt >= FULL_RELOAD_MS || this.needsReload;
    if (reload) {
      this.series.clear();
      this.loadedAt = data.time * 1000;
      this.needsReload = false;
    }
    this.stages = await getJson("api/stages", { ...this.scope(), since: this.rangeSince(data), max_points: MAX_POINTS });
    await this.fetchNewPoints(data.metrics.map((metric) => metric.name), 0);
    await this.fetchNewPoints(data.telemetry_series, this.rangeSince(data));
  }

  // Streams advance independently, so each series resumes from its own newest point.
  async fetchNewPoints(names, floor) {
    if (!names.length) return;
    const after = names.map((name) => this.lastTime(name, floor));
    const response = await getJson("api/series", { ...this.scope(), name: names, after, max_points: MAX_POINTS });
    for (const [name, incoming] of Object.entries(response.series)) {
      const current = this.series.get(name);
      if (!current) {
        this.series.set(name, incoming);
        continue;
      }
      current.t.push(...incoming.t);
      current.y.push(...incoming.y);
      if (current.x && incoming.x) current.x.push(...incoming.x);
      if (incoming.t.length) current.axis = incoming.axis;
    }
  }

  lastTime(name, floor) {
    const series = this.series.get(name);
    return series?.t.length ? series.t[series.t.length - 1] : floor;
  }

  render(data) {
    this.renderHead(data);
    this.renderTiles(data);
    this.renderTasks(data);
    this.renderResources(data);
    this.renderStageTable(data);
    this.renderMetrics(data);
    this.renderItems(data);
    this.renderDetails(data);
  }

  renderHead(data) {
    const head = document.getElementById("head");
    const attempts = data.attempts;
    const options = attempts.map((attempt) => `<option value="${esc(attempt.execution_id)}" ${attempt.execution_id === data.execution_id ? "selected" : ""}>${esc(shortId(attempt.execution_id))} · ${esc(attempt.status)}</option>`).join("");
    const html = `<div class="heading"><h1>${esc(data.run_name)}</h1>${pill(this.displayStatus(data))}</div>
      <p class="meta"><a href="./">All runs</a> · ${esc(data.entry)} · execution <span class="mono">${esc(this.execution ? data.execution_id : shortId(data.execution_id))}</span></p>
      <div class="controls">
        ${attempts.length > 1 ? `<label>Attempt <select id="attempt"><option value="">Newest</option>${options}</select></label>` : ""}
        <label>Time window <select id="window">
          <option value="60">1 min</option><option value="300">5 min</option><option value="1800">30 min</option>
          <option value="7200">2 h</option><option value="0">All</option>
        </select></label>
      </div>`;
    if (head.dataset.signature !== html) {
      head.innerHTML = html;
      head.dataset.signature = html;
      const windowSelect = document.getElementById("window");
      windowSelect.value = String(this.windowSeconds);
      windowSelect.addEventListener("change", () => {
        this.windowSeconds = Number(windowSelect.value);
        this.needsReload = true;
        this.refresh();
      });
      const attempt = document.getElementById("attempt");
      if (attempt) {
        attempt.value = this.execution ?? "";
        attempt.addEventListener("change", () => {
          location.search = query({ entry: this.entry, run: this.run, execution: attempt.value || null });
        });
      }
    }
  }

  displayStatus(data) {
    return data.producers.some((producer) => producer.status === "stale") ? "stale" : data.status;
  }

  activeTask(data) {
    const running = data.tasks.filter((task) => task.status === "running");
    const pool = running.length ? running : data.tasks;
    const roots = pool.filter((task) => task.total !== null && !task.parent_task_id);
    const candidates = roots.length ? roots : pool;
    return candidates.reduce((best, task) => (!best || (task.updated_at ?? 0) > (best.updated_at ?? 0) ? task : best), null);
  }

  windowMean(prefix, suffix, data) {
    const since = this.rangeSince(data);
    const values = [];
    for (const [name, series] of this.series) {
      if (!name.startsWith(prefix) || !name.endsWith(suffix)) continue;
      series.t.forEach((time, index) => {
        if (time >= since && series.y[index] !== null) values.push(series.y[index]);
      });
    }
    return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
  }

  renderTiles(data) {
    const task = this.activeTask(data);
    const tiles = [
      ["Progress", task && task.total !== null ? `${number(task.completed, 0)} / ${number(task.total, 0)}` : "–"],
      ["Rate", task?.throughput ? `${number(task.throughput)} b/s` : "–"],
      ["ETA", duration(task?.eta_seconds)],
      ["Elapsed", duration(data.duration_seconds)],
      ["Last event", age(data.updated_at, data.time)],
    ];
    const gpu = this.windowMean("telemetry/gpu/", "/utilization", data);
    if (gpu !== null) tiles.push(["GPU mean, window", `${number(gpu, 1)}%`]);
    const main = this.windowMean("telemetry/cpu/main_cores", "", data);
    if (main !== null) tiles.push(["Main CPU, window", `${number(main)} cores`]);
    setHtml(document.getElementById("tiles"), tiles.map(([label, value]) => `<div class="tile"><div class="label">${esc(label)}</div><div class="value">${esc(value)}</div></div>`).join(""));
  }

  renderTasks(data) {
    const target = document.getElementById("tasks");
    if (!data.tasks.length) {
      setHtml(target, "");
      return;
    }
    const messages = data.tasks.some((task) => task.message);
    setHtml(target, "<h2>TASKS</h2>" + table(
      [th("Task"), th("Phase"), th("Producer"), th("Status"), th("Progress"), th("Rate", "num"), th("ETA", "num"), ...(messages ? [th("Message")] : [])],
      data.tasks.map((task) => `<tr>
        ${td(esc(task.task_id))}${td(esc(task.phase))}${td(esc(task.producer))}${td(pill(task.status))}
        ${td(progressCell(task))}
        ${td(task.throughput ? `${number(task.throughput)} b/s` : "–", "num")}
        ${td(duration(task.eta_seconds), "num")}
        ${messages ? td(esc(task.message ?? ""), "wrap") : ""}
      </tr>`),
    ));
  }

  telemetryCharts(data) {
    const charts = [];
    const names = data.telemetry_series;
    const groupBy = (suffix, title, unit, max) => {
      const series = names.filter((name) => name.startsWith("telemetry/gpu/") && name.endsWith(suffix));
      if (series.length) {
        charts.push({
          id: title,
          title,
          unit,
          max,
          stages: true,
          series: series.map((name) => ({ name, label: `GPU ${name.split("/")[2]}` })),
        });
      }
    };
    groupBy("/utilization", "GPU utilization", "%", 100);
    groupBy("/memory_mib", "GPU memory", " MiB");
    const cpu = names.filter((name) => name.startsWith("telemetry/cpu/"));
    if (cpu.length) {
      charts.push({
        id: "CPU load",
        title: "CPU load",
        unit: " cores",
        stages: true,
        series: cpu.map((name) => ({ name, label: name.endsWith("main_cores") ? "Main process" : "Child processes" })),
      });
    }
    // Stages stay visible when no telemetry was recorded.
    if (!charts.length && data.stages.length) {
      charts.push({ id: "Stage timeline", title: "Stage timeline", stages: true, timeline: true, max: 1, series: [] });
    }
    return charts;
  }

  renderResources(data) {
    const target = document.getElementById("resources");
    const charts = this.telemetryCharts(data);
    const notes = Object.entries(data.telemetry_unavailable ?? {}).map(([source, detail]) => `<p class="notice">${esc(source.toUpperCase())} telemetry unavailable: ${esc(detail)}</p>`);
    if (!charts.length && !notes.length && !data.stages.length) {
      if (!target.dataset.signature) target.innerHTML = '<h2>RESOURCES</h2><p class="muted">No telemetry was recorded for this execution.</p>';
      return;
    }
    const legend = [...new Set(data.stages.map((item) => item.stage))]
      .map((stage) => `<span><i style="background:${esc(stageColor(stage))};opacity:.55"></i>${esc(stage)}</span>`)
      .join("");
    const signature = charts.map((chart) => chart.id + chart.series.length).join() + legend + notes.join("");
    if (target.dataset.signature !== signature) {
      target.dataset.signature = signature;
      target.innerHTML = `<h2>RESOURCES</h2>${notes.join("")}
        ${legend ? `<div class="legend">${legend}</div><p class="meta">Shading shows the main lane; strips along the top show other lanes.</p>` : ""}
        <div class="charts" id="resource-charts"></div>`;
      this.dropCharts("telemetry:");
    }
    const holder = document.getElementById("resource-charts");
    for (const chart of charts) this.drawChart(holder, `telemetry:${chart.id}`, chart, "time");
  }

  renderStageTable(data) {
    const target = document.getElementById("stage-table");
    if (!data.stage_summary.length) {
      setHtml(target, "");
      return;
    }
    const multi = new Set(data.stage_summary.map((row) => row.producer)).size > 1;
    setHtml(target, "<h2>STAGES</h2>" + table(
      [...(multi ? [th("Producer")] : []), th("Lane"), th("Stage"), th("Time share", "num"), th("Runs", "num"), th("Mean per run", "num"), th("GPU mean", "num"), th("GPU median", "num"), th("Main CPU", "num"), th("Child CPU", "num")],
      data.stage_summary.map((row) => `<tr>
        ${multi ? td(esc(row.producer)) : ""}
        ${td(esc(row.lane))}${td(esc(row.stage))}
        ${td(`${number(row.share * 100, 1)}%`, "num")}
        ${td(number(row.count, 0), "num")}
        ${td(row.mean_seconds < 10 ? `${number(row.mean_seconds)} s` : duration(row.mean_seconds), "num")}
        ${td(row.gpu_mean === null ? "–" : `${number(row.gpu_mean, 1)}%`, "num")}
        ${td(row.gpu_median === null ? "–" : `${number(row.gpu_median, 0)}%`, "num")}
        ${td(number(row.main_cores), "num")}
        ${td(number(row.worker_cores), "num")}
      </tr>`),
    ));
  }

  metricSections(data) {
    const sections = new Map();
    for (const metric of data.metrics) {
      const parts = metric.name.split("/");
      const section = parts.length > 1 ? parts[0] : "metrics";
      if (!sections.has(section)) sections.set(section, []);
      sections.get(section).push(metric);
    }
    return sections;
  }

  renderMetrics(data) {
    const target = document.getElementById("metrics");
    const sections = this.metricSections(data);
    const signature = [...sections].map(([section, items]) => section + items.map((item) => item.name).join()).join();
    if (target.dataset.signature !== signature) {
      target.dataset.signature = signature;
      target.innerHTML = [...sections.keys()].map((section, index) => `<h2>${esc(section.toUpperCase())}</h2><div class="charts" id="metric-section-${index}"></div>`).join("");
      this.dropCharts("metric:");
    }
    [...sections.values()].forEach((items, index) => {
      const holder = document.getElementById(`metric-section-${index}`);
      for (const item of items) {
        const series = this.series.get(item.name);
        this.drawChart(holder, `metric:${item.name}`, {
          title: item.name,
          subtitle: number(item.latest, 4),
          series: [{ name: item.name, label: item.name.split("/").slice(-1)[0] }],
          decimals: true,
        }, series?.axis === "epoch" ? "epoch" : "time");
      }
    });
  }

  renderItems(data) {
    const target = document.getElementById("items");
    if (!data.display_items.length) {
      setHtml(target, "");
      return;
    }
    const items = [...data.display_items].sort((a, b) => (a.kind !== "table") - (b.kind !== "table"));
    setHtml(target, "<h2>DISPLAY</h2>" + items.map((item) => {
      const title = `<div class="table-title">${esc(item.name)}</div>`;
      if (item.kind === "table" && Array.isArray(item.data.columns) && Array.isArray(item.data.rows)) {
        return title + table(
          item.data.columns.map((column) => th(column)),
          item.data.rows.map((row) => `<tr>${(Array.isArray(row) ? row : [row]).map((cell) => td(esc(cell))).join("")}</tr>`),
        );
      }
      return `<details open><summary>${esc(item.kind)}: ${esc(item.name)}</summary><pre>${esc(JSON.stringify(item.data, null, 2))}</pre></details>`;
    }).join(""));
  }

  renderDetails(data) {
    const target = document.getElementById("details");
    const now = data.time;
    const producers = data.producers.length
      ? "<h2>PRODUCERS</h2>" + table(
        [th("Producer"), th("Status"), th("Phase"), th("Exit"), th("Updated")],
        data.producers.map((producer) => `<tr>${td(esc(producer.label))}${td(pill(producer.status))}${td(esc(producer.phase ?? "–"))}${td(esc(producer.exit_code ?? producer.signal ?? "–"))}${td(age(producer.updated_at, now))}</tr>`),
      )
      : "";
    const attempts = "<h2>ATTEMPTS</h2>" + table(
      [th("Execution"), th("Status"), th("Created"), th("Duration", "num"), th("Parent")],
      [...data.attempts].reverse().map((attempt) => `<tr>
        ${td(`<a class="mono" href="?${esc(query({ entry: this.entry, run: this.run, execution: attempt.execution_id }))}">${esc(this.execution === attempt.execution_id ? attempt.execution_id : shortId(attempt.execution_id))}</a>`)}
        ${td(pill(attempt.status))}
        ${td(esc(new Date(attempt.created_at * 1000).toLocaleString()))}
        ${td(duration(attempt.duration_seconds), "num")}
        ${td(`<span class="mono">${esc(shortId(attempt.parent_execution_id) || "–")}</span>`)}
      </tr>`),
    );
    setHtml(target, producers + attempts + warningsHtml(data.warnings));
  }

  dropCharts(prefix) {
    for (const [key, chart] of this.charts) {
      if (key.startsWith(prefix)) {
        chart.plot.destroy();
        this.charts.delete(key);
      }
    }
  }

  chartData(definition, axis) {
    if (definition.timeline) {
      const end = this.anchor(this.data);
      return [[Math.min(end - 1, ...this.stages.t), end]];
    }
    const seriesData = definition.series.map((item) => this.series.get(item.name) ?? { t: [], y: [], x: null });
    const xValue = (series, index) => (axis === "epoch" && series.x ? series.x[index] : series.t[index]);
    if (seriesData.length === 1) {
      const series = seriesData[0];
      const order = series.t.map((_, index) => index).sort((a, b) => xValue(series, a) - xValue(series, b));
      return [order.map((index) => xValue(series, index)), order.map((index) => series.y[index])];
    }
    // Series sampled together differ slightly, so one row covers samples within half a second.
    const samples = [];
    seriesData.forEach((series, position) => {
      series.t.forEach((_, index) => samples.push([xValue(series, index), position, series.y[index]]));
    });
    samples.sort((a, b) => a[0] - b[0]);
    const keys = [];
    const columns = seriesData.map(() => []);
    for (const [x, position, y] of samples) {
      if (!keys.length || x - keys[keys.length - 1] > 0.5 || columns[position][keys.length - 1] != null) {
        keys.push(x);
        columns.forEach((column) => column.push(null));
      }
      columns[position][keys.length - 1] = y;
    }
    return [keys, ...columns];
  }

  drawChart(holder, key, definition, axis) {
    const data = this.chartData(definition, axis);
    const end = this.data ? this.anchor(this.data) : 0;
    const start = Math.max(end - this.windowSeconds, data[0].length ? data[0][0] : end - this.windowSeconds);
    const scale = axis === "time" && definition.stages && this.windowSeconds && start < end
      ? { min: start, max: end }
      : null;
    let chart = this.charts.get(key);
    if (chart && chart.axis !== axis) {
      chart.plot.destroy();
      chart.root.remove();
      this.charts.delete(key);
      chart = null;
    }
    if (!chart) {
      const root = document.createElement("div");
      root.className = "chart";
      root.innerHTML = `<p class="title"></p><div class="plot"></div>`;
      holder.appendChild(root);
      const plot = new uPlot(this.chartOptions(definition, axis, root), data, root.querySelector(".plot"));
      chart = { plot, axis, root };
      this.charts.set(key, chart);
    } else {
      chart.plot.setData(data, !scale);
    }
    const title = definition.subtitle ? `${definition.title} · ${definition.subtitle}` : definition.title;
    chart.root.querySelector(".title").textContent = title;
    if (scale) chart.plot.setScale("x", scale);
  }

  chartOptions(definition, axis, root) {
    const ink = css("--muted");
    const grid = css("--grid");
    const axes = [
      { stroke: ink, grid: { stroke: grid, width: 1 }, ticks: { stroke: grid, width: 1 } },
      {
        stroke: ink,
        size: 58,
        grid: { stroke: grid, width: 1 },
        ticks: { stroke: grid, width: 1 },
        values: (_, values) => values.map((value) => number(value, 3)),
      },
    ];
    if (definition.timeline) axes[1].show = false;
    if (axis === "epoch") {
      // Whole epochs read best, unless the view spans less than one.
      const whole = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000];
      axes[0].incrs = (_, __, min, max) => (max - min < 1 ? [0.1, 0.2, 0.5, ...whole] : whole);
      axes[0].values = (_, values) => values.map((value) => number(value, 1));
    }
    return {
      width: Math.max(200, root.clientWidth - 18),
      height: 210,
      scales: {
        x: { time: axis === "time" },
        y: {
          range: definition.max
            ? [0, definition.max]
            : (_, min, max) => {
              if (min === null || max === null) return [0, 1];
              if (min === max) return [min - 1, max + 1];
              const pad = (max - min) * 0.08;
              return [min - pad, max + pad];
            },
        },
      },
      axes,
      series: [
        { label: axis === "epoch" ? "Epoch" : "Time", value: axis === "epoch" ? (_, value) => number(value, 1) : undefined },
        ...definition.series.map((item, index) => ({
          label: item.label,
          stroke: css(PALETTE[index % PALETTE.length]),
          width: 1.5,
          spanGaps: false,
          points: { show: axis === "epoch" ? undefined : false },
          value: (_, value) => (value === null || value === undefined ? "–" : `${number(value, 4)}${definition.unit ?? ""}`),
        })),
      ],
      cursor: axis === "time" ? { sync: { key: "mammoth-time" } } : {},
      hooks: definition.stages ? { drawClear: [(plot) => this.drawStages(plot)] } : {},
    };
  }

  drawStages(plot) {
    const { ctx, bbox } = plot;
    const min = plot.scales.x.min;
    const max = plot.scales.x.max;
    const now = this.data ? this.anchor(this.data) : max;
    const lanes = new Map();
    this.stages.t.forEach((time, index) => {
      const key = `${this.stages.producer[index]}\u0000${this.stages.lane[index]}`;
      if (!lanes.has(key)) lanes.set(key, []);
      lanes.get(key).push({ time, end: this.stages.end[index] ?? now, stage: this.stages.stage[index], lane: this.stages.lane[index] });
    });
    let strip = 0;
    ctx.save();
    for (const items of lanes.values()) {
      const main = items[0].lane === "main";
      const top = main ? bbox.top : bbox.top + strip * 8 * devicePixelRatio;
      const height = main ? bbox.height : 6 * devicePixelRatio;
      if (!main) strip += 1;
      items.forEach((item) => {
        if (item.end < min || item.time > max) return;
        const left = plot.valToPos(Math.max(item.time, min), "x", true);
        const right = plot.valToPos(Math.min(item.end, max), "x", true);
        ctx.fillStyle = stageColor(item.stage);
        ctx.globalAlpha = main ? 0.16 : 0.85;
        ctx.fillRect(left, top, Math.max(1, right - left), height);
      });
    }
    ctx.restore();
  }

  resizeCharts() {
    for (const chart of this.charts.values()) {
      chart.plot.setSize({ width: Math.max(200, chart.root.clientWidth - 18), height: 210 });
    }
  }
}

async function runPage(entry, run, execution) {
  const view = new RunPage(entry, run, execution);
  await view.refresh();
  setInterval(() => { if (!document.hidden) view.refresh(); }, REFRESH_MS);
}

const entryParam = params.get("entry");
if (entryParam && params.get("run")) runPage(entryParam, params.get("run"), params.get("execution"));
else if (entryParam && params.get("group")) groupPage(entryParam, params.get("group"));
else fleetPage();
