(() => {
  "use strict";
  const API_BASE = (window.MHP_CONFIG && window.MHP_CONFIG.API_BASE) || "http://127.0.0.1:8000";
  const L = window.MHP;
  const $ = (id) => document.getElementById(id);
  const form = $("predict-form"), submitBtn = $("submit-btn");
  const states = { idle: $("state-idle"), loading: $("state-loading"), result: $("state-result"), error: $("state-error") };
  const GAUGE = 314;
  let noiseFloor = null, lastFocus = null;

  const LABEL = { Age: "Age", Gender: "Gender", Country: "Country", Academic_Level: "Academic level", Most_Used_Platform: "Main platform",
    Purpose_Of_Use: "Purpose of use", Avg_Daily_Usage_Hours: "Daily screen time", Daily_Unlocks: "Phone unlocks", Study_Hours: "Study hours",
    Physical_Activity_Hours: "Physical activity", Sleep_Hours_Per_Night: "Sleep", Stress_Level: "Stress level" };

  // ---- ticks
  document.querySelectorAll(".gauge-ticks").forEach((g) => {
    for (let i = 0; i <= 10; i += 2) {
      const a = Math.PI - (i / 10) * Math.PI, l = document.createElementNS("http://www.w3.org/2000/svg", "line");
      [["x1", 120 + 100 * Math.cos(a)], ["y1", 140 - 100 * Math.sin(a)], ["x2", 120 + 90 * Math.cos(a)], ["y2", 140 - 90 * Math.sin(a)]].forEach(([k, v]) => l.setAttribute(k, v.toFixed(1)));
      g.appendChild(l);
    }
  });

  // ---- tabs
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => { x.classList.toggle("active", x === t); x.setAttribute("aria-selected", String(x === t)); });
    document.querySelectorAll("[data-panel]").forEach((p) => (p.hidden = p.dataset.panel !== t.dataset.tab));
    if (t.dataset.tab === "trends") renderTrends();
  }));

  // ---- stress control
  const seg = $("stress_level_group"), stressInput = $("stress_level");
  seg.querySelectorAll(".seg-btn").forEach((b) => b.addEventListener("click", () => {
    seg.querySelectorAll(".seg-btn").forEach((x) => x.classList.remove("active")); b.classList.add("active");
    stressInput.value = b.dataset.value; clearErr(stressInput);
  }));

  // ---- field errors
  const wrap = (el) => el.closest(".field");
  function setErr(el, msg) { const w = wrap(el); if (!w) return; w.classList.add("field-error"); const m = w.querySelector(".error-msg"); if (m) m.textContent = msg; }
  function clearErr(el) { const w = wrap(el); if (!w) return; w.classList.remove("field-error"); const m = w.querySelector(".error-msg"); if (m) m.textContent = ""; }
  form.querySelectorAll("input, select").forEach((el) => { el.addEventListener("input", () => clearErr(el)); el.addEventListener("change", () => clearErr(el)); });
  const fieldEl = (name) => $(name);

  function collect() {
    const fd = new FormData(form), num = (k, int) => (fd.get(k) === "" || fd.get(k) === null ? NaN : (int ? parseInt(fd.get(k), 10) : parseFloat(fd.get(k))));
    return { age: num("age", true), gender: fd.get("gender") || "", country: (fd.get("country") || "").trim(), academic_level: fd.get("academic_level") || "",
      most_used_platform: fd.get("most_used_platform") || "", purpose_of_use: fd.get("purpose_of_use") || "", avg_daily_usage_hours: num("avg_daily_usage_hours"),
      daily_unlocks: num("daily_unlocks", true), study_hours: num("study_hours"), physical_activity_hours: num("physical_activity_hours"),
      sleep_hours_per_night: num("sleep_hours_per_night"), stress_level: fd.get("stress_level") || "" };
  }

  const show = (n) => { Object.values(states).forEach((e) => (e.hidden = true)); states[n].hidden = false; };
  function renderError(label, copy) { $("error-label").textContent = label; $("error-copy").textContent = copy; show("error"); }

  function applyServerErrors(detail) {
    if (!Array.isArray(detail)) return false; let hit = false;
    detail.forEach((e) => { const f = Array.isArray(e.loc) ? e.loc[e.loc.length - 1] : null; const el = f && fieldEl(f); if (el) { setErr(el, e.msg || "Invalid value."); hit = true; } });
    return hit;
  }

  const list = (ul, items, fn) => { ul.replaceChildren(); items.forEach((i) => ul.appendChild(fn(i))); };
  function li(text) { const x = document.createElement("li"); x.textContent = text; return x; }   // textContent only: no HTML injection

  function renderResult(r) {
    const s = r.predicted_mental_health_score, c = Math.max(0, Math.min(10, s));
    $("score-number").textContent = s.toFixed(2);
    $("score-band").textContent = r.position ? "Relative to the reference data: " + r.position.description.replace(" of the reference dataset", "").replace(" in the reference dataset", "") : "";
    $("score-context").textContent = "";
    const iv = r.interval, ib = $("interval-box");
    if (iv) { ib.hidden = false; $("interval-range").textContent = `${Math.round(iv.level * 100)}% range: ${iv.lower.toFixed(2)} – ${iv.upper.toFixed(2)}`;
      $("uncertainty-badge").textContent = iv.reliable ? `${iv.uncertainty} uncertainty (relative)` : "outside training range"; } else ib.hidden = true;
    const wb = $("warn-box"); wb.replaceChildren();
    (r.input_warnings || []).forEach((w) => wb.appendChild(li(w.message))); wb.hidden = !(r.input_warnings || []).length;
    const eb = $("explain-box");
    if (r.explanation) {
      const cs = r.explanation.contributions.slice(0, 6), mx = Math.max(...cs.map((x) => Math.abs(x.contribution)), 0.01);
      list($("explain-list"), cs, (x) => {
        const row = document.createElement("li"), t = document.createElement("span");
        t.textContent = `${LABEL[x.feature] || x.feature} = ${x.value}  →  ${x.contribution >= 0 ? "+" : ""}${x.contribution.toFixed(2)}`;
        const tr = document.createElement("div"); tr.className = "bar-track"; const f = document.createElement("div");
        f.className = "bar-fill " + (x.contribution >= 0 ? "pos" : "neg"); f.style.width = (Math.abs(x.contribution) / mx) * 50 + "%"; tr.appendChild(f);
        row.appendChild(t); row.appendChild(tr); return row;
      }); eb.hidden = false;
    } else eb.hidden = true;
    const fb = $("feedback-box"); list($("feedback-list"), r.feedback || [], li); fb.hidden = !(r.feedback || []).length;
    const g = $("gauge-fill"); g.style.transition = "none"; g.style.strokeDashoffset = String(GAUGE);
    requestAnimationFrame(() => { g.style.transition = ""; g.style.strokeDashoffset = String(GAUGE * (1 - c / 10)); });
    show("result");
  }

  async function call(path, payload) {
    const res = await fetch(`${API_BASE}${path}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    return res;
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault(); form.querySelectorAll(".field").forEach((f) => f.classList.remove("field-error"));
    const p = collect(), errs = L.validate(p);
    if (errs.length) { errs.forEach((x) => fieldEl(x.field) && setErr(fieldEl(x.field), x.message)); const f = fieldEl(errs[0].field); f && f.focus && f.focus(); return; }
    submitBtn.disabled = true; submitBtn.classList.add("loading"); show("loading");
    try {
      let res = await call("/explain", p);
      if (res.status === 404 || res.status === 405) res = await call("/predict", p);       // graceful fallback to the ORIGINAL API contract
      if (res.status === 422) {
        const b = await res.json().catch(() => null), hit = b && applyServerErrors(b.detail);
        return renderError("Check your inputs", hit ? "The API rejected a few fields — details are marked on the form." : "The API rejected this submission. Please review your inputs.");
      }
      if (res.status === 429) return renderError("Slow down a little", "Too many requests. Please wait a minute and try again.");
      if (!res.ok) { const b = await res.json().catch(() => null); return renderError("Prediction failed", (b && typeof b.detail === "string" && b.detail) || `The API responded with status ${res.status}.`); }
      const r = await res.json();
      if (typeof r.predicted_mental_health_score !== "number") return renderError("Unexpected response", "The API responded, but the score was missing or malformed.");
      renderResult(r);
      L.pushHistory(localStorage, L.makeEntry(p, r)); $("saved-note").textContent = "Saved to this browser only (see “My trend”).";
    } catch (err) {
      renderError("Can't reach the server", `Couldn't connect to ${API_BASE}. If this is a free-tier host it may be waking up; wait ~30 s and try again.`);
    } finally { submitBtn.disabled = false; submitBtn.classList.remove("loading"); }
  });

  $("reset-btn").addEventListener("click", () => show("idle"));
  $("error-retry-btn").addEventListener("click", () => show("idle"));

  // ---- trends
  function renderTrends() {
    const h = L.readHistory(localStorage), box = $("charts"); box.replaceChildren();
    $("trend-message").textContent = L.trendMessage(h, noiseFloor);
    const defs = [["Predicted score", "score", 0, 10], ["Sleep (h)", "sleep"], ["Screen time (h)", "screen"], ["Physical activity (h)", "activity"], ["Stress (0=Low … 3=Very High)", "stress", 0, 3]];
    if (!h.length) return;
    defs.forEach(([title, key, lo, hi]) => {
      const vals = h.map((e) => e[key]).filter((v) => v !== null && v !== undefined); if (!vals.length) return;
      const d = document.createElement("div"); d.className = "chart"; const t = document.createElement("h4"); t.textContent = title; d.appendChild(t);
      const ns = "http://www.w3.org/2000/svg", svg = document.createElementNS(ns, "svg"); svg.setAttribute("viewBox", "0 0 260 90"); svg.setAttribute("role", "img"); svg.setAttribute("aria-label", `${title} over ${vals.length} assessments`);
      if (key === "score" && h.every((e) => e.lower !== null)) {
        const up = L.polyline(h.map((e) => e.upper), 260, 90, 8, lo, hi).split(" "), lw = L.polyline(h.map((e) => e.lower), 260, 90, 8, lo, hi).split(" ").reverse();
        const b = document.createElementNS(ns, "polygon"); b.setAttribute("points", up.concat(lw).join(" ")); b.setAttribute("class", "band"); svg.appendChild(b);
      }
      const pl = document.createElementNS(ns, "polyline"); pl.setAttribute("points", L.polyline(vals, 260, 90, 8, lo, hi)); pl.setAttribute("class", "line"); svg.appendChild(pl);
      d.appendChild(svg); box.appendChild(d);
    });
  }
  $("clear-btn").addEventListener("click", () => { if (confirm("Delete all saved assessments from this browser?")) { L.clearHistory(localStorage); renderTrends(); } });
  $("export-btn").addEventListener("click", () => {
    const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([L.toCSV(L.readHistory(localStorage))], { type: "text/csv" })); a.download = "my_wellbeing_history.csv"; a.click(); URL.revokeObjectURL(a.href);
  });

  // ---- model facts (dynamic; nothing hard-coded)
  fetch(`${API_BASE}/model-info`).then((r) => (r.ok ? r.json() : Promise.reject())).then((m) => {
    const nf = m.noise_floor && m.noise_floor.noise_floor && m.noise_floor.noise_floor["input_noise_sd_0.25h"];
    if (nf) noiseFloor = nf.abs_change_p95;
    $("model-facts").textContent = `Model v${m.model_version} (${m.algorithm}), trained ${m.training_date}. On ${m.n_test} held-out students: R² ${m.test_metrics.R2.toFixed(3)}, mean absolute error ${m.test_metrics.MAE.toFixed(2)} score points.`;
  }).catch(() => { $("model-facts").textContent = "Model facts are unavailable (the API may be waking up or is running the original version)."; });
})();
