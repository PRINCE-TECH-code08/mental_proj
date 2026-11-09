// Run: node --test frontend/lib.test.js
const test = require("node:test"), assert = require("node:assert"), L = require("./lib.js");
const mem = () => { const m = {}; return { getItem: (k) => (k in m ? m[k] : null), setItem: (k, v) => { m[k] = v; }, removeItem: (k) => { delete m[k]; } }; };
const P = { age: 21, gender: "Male", country: "India", academic_level: "Graduate", most_used_platform: "TikTok", purpose_of_use: "News",
            avg_daily_usage_hours: 4, daily_unlocks: 100, study_hours: 2, physical_activity_hours: 1, sleep_hours_per_night: 7, stress_level: "Low" };
test("validate accepts a good payload", () => assert.deepStrictEqual(L.validate(P), []));
test("validate flags NaN, range and missing", () => {
  const e = L.validate({ ...P, age: NaN, sleep_hours_per_night: 30, stress_level: "" }).map((x) => x.field).sort();
  assert.deepStrictEqual(e, ["age", "sleep_hours_per_night", "stress_level"]);
});
test("history is capped, tolerant of corrupt storage, and clearable", () => {
  const s = mem(); for (let i = 0; i < L.MAX + 5; i++) L.pushHistory(s, L.makeEntry(P, { predicted_mental_health_score: 6 + i / 100 }));
  assert.strictEqual(L.readHistory(s).length, L.MAX);
  s.setItem("mhp_history_v1", "{not json"); assert.deepStrictEqual(L.readHistory(s), []);
  L.pushHistory(s, L.makeEntry(P, { predicted_mental_health_score: 6 })); L.clearHistory(s); assert.deepStrictEqual(L.readHistory(s), []);
});
test("entries store no age/country/gender (data minimisation)", () => {
  const e = L.makeEntry(P, { predicted_mental_health_score: 6.1, interval: { lower: 5, upper: 7 } });
  ["age", "country", "gender"].forEach((k) => assert.ok(!(k in e))); assert.strictEqual(e.stress, 0);
});
test("csv has header + rows", () => assert.strictEqual(L.toCSV([L.makeEntry(P, { predicted_mental_health_score: 6 })]).split("\n").length, 2));
test("trend message respects noise floor and always carries the caveat", () => {
  const h = (a, b) => [{ score: a }, { score: b }];
  assert.match(L.trendMessage(h(6, 6.2), 0.55), /within/); assert.match(L.trendMessage(h(6, 7), 0.55), /larger than/);
  assert.match(L.trendMessage([{ score: 6 }], 0.55), /at least two/);
  [h(6, 6.2), h(6, 7), [{ score: 1 }]].forEach((x) => assert.ok(L.trendMessage(x, 0.55).includes(L.CAVEAT)));
});
test("polyline handles flat, single and normal series", () => {
  assert.ok(L.polyline([5, 5, 5], 100, 40, 4).split(" ").length === 3); assert.ok(L.polyline([5], 100, 40, 4).length > 0); assert.strictEqual(L.polyline([], 100, 40, 4), "");
});
