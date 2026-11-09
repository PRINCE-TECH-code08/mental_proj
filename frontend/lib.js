/* Pure helpers (no DOM) so they can be unit-tested in Node. */
(function (root) {
  const KEY = "mhp_history_v1", MAX = 50;
  const STRESS = { Low: 0, Medium: 1, High: 2, "Very High": 3 };
  const RULES = [["age", 10, 100], ["avg_daily_usage_hours", 0, 24], ["daily_unlocks", 0, 5000],
                 ["study_hours", 0, 24], ["physical_activity_hours", 0, 24], ["sleep_hours_per_night", 0, 24]];
  const CAVEAT = "A trend in model predictions does not establish a clinical change in mental health.";

  function validate(p) {
    const errors = [];
    RULES.forEach(([k, lo, hi]) => {
      const v = p[k];
      if (v === "" || v === null || v === undefined || Number.isNaN(v)) errors.push({ field: k, message: "This field is required." });
      else if (v < lo || v > hi) errors.push({ field: k, message: `Must be between ${lo} and ${hi}.` });
    });
    ["gender", "country", "academic_level", "most_used_platform", "purpose_of_use", "stress_level"].forEach((k) => {
      if (!p[k] || String(p[k]).trim() === "") errors.push({ field: k, message: k === "stress_level" ? "Pick a stress level." : "This field is required." });
    });
    return errors;
  }
  function readHistory(st) { try { const a = JSON.parse(st.getItem(KEY) || "[]"); return Array.isArray(a) ? a : []; } catch (e) { return []; } }
  function pushHistory(st, entry) { const h = readHistory(st); h.push(entry); while (h.length > MAX) h.shift(); try { st.setItem(KEY, JSON.stringify(h)); } catch (e) {} return h; }
  function clearHistory(st) { try { st.removeItem(KEY); } catch (e) {} }
  // data minimisation: only habit values + score are stored; no age, country, gender or identifiers
  function makeEntry(p, r, now) {
    return { t: (now || new Date()).toISOString(), score: r.predicted_mental_health_score,
             lower: r.interval ? r.interval.lower : null, upper: r.interval ? r.interval.upper : null,
             sleep: p.sleep_hours_per_night, screen: p.avg_daily_usage_hours, activity: p.physical_activity_hours,
             study: p.study_hours, stress: STRESS[p.stress_level] };
  }
  function toCSV(h) {
    const cols = ["t", "score", "lower", "upper", "sleep", "screen", "activity", "study", "stress"];
    return [cols.join(",")].concat(h.map((e) => cols.map((c) => e[c] ?? "").join(","))).join("\n");
  }
  function polyline(vals, w, h, pad, lo, hi) {
    if (!vals.length) return "";
    lo = lo ?? Math.min(...vals); hi = hi ?? Math.max(...vals); if (hi === lo) { hi += 1; lo -= 1; }
    return vals.map((v, i) => {
      const x = vals.length === 1 ? w / 2 : pad + (i * (w - 2 * pad)) / (vals.length - 1);
      const y = h - pad - ((v - lo) / (hi - lo)) * (h - 2 * pad);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ");
  }
  function trendMessage(h, noiseP95) {
    if (h.length < 2) return "Save at least two assessments to see a trend. " + CAVEAT;
    const d = h[h.length - 1].score - h[h.length - 2].score, s = (d >= 0 ? "+" : "") + d.toFixed(2);
    if (noiseP95 != null && Math.abs(d) <= noiseP95)
      return `Latest change ${s}: within the ±${noiseP95.toFixed(2)} that input noise alone can produce (assumption-based), so it should not be read as a real change. ${CAVEAT}`;
    return `Latest change ${s}: larger than the input-noise range${noiseP95 != null ? " (±" + noiseP95.toFixed(2) + ")" : ""}, but it is still a change in a model estimate. ${CAVEAT}`;
  }
  const api = { validate, readHistory, pushHistory, clearHistory, makeEntry, toCSV, polyline, trendMessage, STRESS, CAVEAT, MAX };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.MHP = api;
})(typeof window !== "undefined" ? window : globalThis);
