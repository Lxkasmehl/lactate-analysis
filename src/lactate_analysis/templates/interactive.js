// Workout planner + similar-session comparison. Runs entirely in the browser
// on posterior draws exported by the Python model (no server, no AI calls):
// every displayed number is the median (and 80% range) over those draws.
(function () {
  const data = JSON.parse(document.getElementById("interactive-data").textContent);
  const P = data.planner;
  const HR = data.heart_rate_planner;
  const MILE_KM = 1.60934;

  // ---------- formatting ----------
  const mmss = (s) => {
    const t = Math.round(s);
    return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, "0")}`;
  };
  const paceKm = (s) => `${mmss(s)}/km`;
  const paceMi = (s) => `${mmss(s * MILE_KM)}/mi`;
  const kmh = (s) => (3600 / s).toFixed(1);
  const mph = (s) => (3600 / (s * MILE_KM)).toFixed(1);
  const signed = (v) => (Math.abs(v) < 0.05 ? "±0.0" : (v > 0 ? "+" : "") + v.toFixed(1));
  const quantile = (arr, q) => {
    const a = [...arr].sort((x, y) => x - y);
    const pos = (a.length - 1) * q;
    const lo = Math.floor(pos), hi = Math.ceil(pos);
    return a[lo] + (a[hi] - a[lo]) * (pos - lo);
  };
  const el = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  // ---------- workout parsing ----------
  // "3x10min/60s", "5 x 1000m / 1min", "4x1600m/60s + 4x400m/30s", "2x3km/90s"
  const SEGMENT = /^(\d+)\s*[x×*]\s*(\d+(?:[.,]\d+)?)\s*(m|km|mi|mile|miles|min|mins|s|sec)?\s*(?:\/\s*(\d+(?:[.,]\d+)?)\s*(s|sec|min|mins)?)?$/i;
  function parseWorkout(text) {
    const reps = [];
    for (const raw of text.split("+")) {
      const seg = raw.trim();
      if (!seg) continue;
      const m = seg.match(SEGMENT);
      if (!m) throw new Error(`Couldn't read "${seg}". Use e.g. 3x10min/60s or 5x1000m/60s.`);
      const n = parseInt(m[1], 10);
      const value = parseFloat(m[2].replace(",", "."));
      const unit = (m[3] || "m").toLowerCase();
      let rest = m[4] ? parseFloat(m[4].replace(",", ".")) : 60;
      if (m[5] && m[5].toLowerCase().startsWith("min")) rest *= 60;
      for (let i = 0; i < n; i++) {
        if (unit.startsWith("min")) reps.push({ timeS: value * 60, rest });
        else if (unit === "s" || unit === "sec") reps.push({ timeS: value, rest });
        else if (unit === "km") reps.push({ distM: value * 1000, rest });
        else if (unit.startsWith("mi")) reps.push({ distM: value * 1609.34, rest });
        else reps.push({ distM: value, rest });
      }
    }
    if (!reps.length) throw new Error("Describe at least one rep, e.g. 3x10min/60s.");
    if (reps.length > 60) throw new Error("That's more than 60 reps; split it up.");
    return reps;
  }

  // ---------- planner math ----------
  const K = Object.fromEntries(P.factor_keys.map((k, i) => [k, i]));
  const ST = P.standard;

  function plan(opts) {
    const reps = parseWorkout(opts.workout);
    const nDraws = P.steepness.length;
    const perRep = reps.map(() => ({ pace: [], hr: [], time: [] }));
    const indoor = opts.env === "Treadmill" || opts.env === "Indoor track";
    const treadmill = opts.env === "Treadmill";
    const shoeIdx = opts.shoe === "" ? -1 : P.shoe_names.indexOf(opts.shoe);

    for (let d = 0; d < nDraws; d++) {
      const c = P.cost[d];
      const b = P.baseline[d];
      const base = P.fitness_now[d] - Math.log((opts.lactate - b) / (P.reference_lactate - b)) / P.steepness[d];
      const shoe = shoeIdx >= 0 ? P.shoe_effects[d][shoeIdx] : 0;
      const fixed =
        c[K.indoor] * (indoor ? 1 : 0) +
        c[K.treadmill] * (treadmill ? 1 : 0) +
        c[K.road] * (opts.env === "Road" ? 1 : 0) +
        c[K.offroad] * (opts.env === "Grass / XC" ? 1 : 0) +
        c[K.incline] * (treadmill ? opts.incline : 0) +
        c[K.heat] * (indoor ? 0 : Math.max(opts.temperature - ST.temperature_c, 0)) +
        c[K.humidity] * (indoor || opts.humidity == null ? 0 : Math.max(opts.humidity - ST.humidity_pct, 0) / 10) +
        c[K.jog_rest] * (opts.jogRest ? 1 : 0) +
        c[K.second_session] * (opts.secondSession ? 1 : 0) +
        shoe;
      let workDoneS = 0;
      reps.forEach((rep, j) => {
        let pace = base + fixed;
        let t = rep.timeS ?? (rep.distM * pace) / 1000;
        // Rep duration depends on pace for distance reps: a few fixed-point
        // iterations converge (the effect is logarithmic in duration).
        for (let it = 0; it < 4; it++) {
          const restFrac = rep.rest / (t + rep.rest);
          const accumulated = (workDoneS + t) / 60;
          pace =
            base + fixed +
            c[K.rep_duration] * Math.log2(t / ST.rep_time_s) +
            c[K.rest_fraction] * (restFrac - ST.rest_fraction) * 10 +
            c[K.accumulated_work] * (accumulated - ST.accumulated_work_min) / 10;
          if (rep.timeS == null) t = (rep.distM * pace) / 1000;
        }
        workDoneS += t;
        perRep[j].pace.push(pace);
        perRep[j].time.push(t);
        if (HR) {
          const h = d % HR.hr_at_2_now.length;
          perRep[j].hr.push(
            HR.hr_at_2_now[h] +
              HR.bpm_per_log_lactate[h] * Math.log(opts.lactate / P.reference_lactate) +
              HR.rep_duration[h] * Math.log2(t / ST.rep_time_s) +
              HR.heat[h] * (indoor ? 0 : Math.max(opts.temperature - ST.temperature_c, 0)) +
              HR.treadmill[h] * (treadmill ? 1 : 0) +
              HR.indoor[h] * (indoor ? 1 : 0)
          );
        }
      });
    }
    return reps.map((rep, j) => ({
      rep,
      pace: { med: quantile(perRep[j].pace, 0.5), lo: quantile(perRep[j].pace, 0.1), hi: quantile(perRep[j].pace, 0.9) },
      time: quantile(perRep[j].time, 0.5),
      hr: HR ? { med: quantile(perRep[j].hr, 0.5), lo: quantile(perRep[j].hr, 0.1), hi: quantile(perRep[j].hr, 0.9) } : null,
    }));
  }

  function outdoorEquivalentIncline() {
    const values = P.cost.map((c) => -(c[K.indoor] + c[K.treadmill]) / c[K.incline]);
    return quantile(values, 0.5);
  }

  // ---------- planner UI ----------
  const shoeSelect = el("pl-shoe");
  P.shoe_names.forEach((name) => shoeSelect.insertAdjacentHTML("beforeend", `<option>${esc(name)}</option>`));

  function readOpts() {
    const num = (id) => {
      const v = parseFloat(el(id).value.replace(",", "."));
      return Number.isFinite(v) ? v : null;
    };
    return {
      workout: el("pl-workout").value,
      lactate: num("pl-lactate") ?? 2.0,
      env: el("pl-env").value,
      incline: num("pl-incline") ?? 0,
      temperature: num("pl-temp") ?? ST.temperature_c,
      humidity: num("pl-humidity"),
      jogRest: el("pl-jog").checked,
      secondSession: el("pl-second").checked,
      shoe: shoeSelect.value,
    };
  }

  function renderPlan() {
    const opts = readOpts();
    const treadmill = opts.env === "Treadmill";
    el("pl-incline-wrap").hidden = !treadmill;
    el("pl-outdoor-wrap").hidden = opts.env === "Treadmill" || opts.env === "Indoor track";
    const out = el("pl-output");
    if (opts.lactate <= Math.max(...P.baseline) + 0.1 || opts.lactate > 6) {
      out.innerHTML = `<p class="warn">Pick a target lactate between about 1.2 and 6 mmol/L.</p>`;
      return;
    }
    let rows;
    try {
      rows = plan(opts);
    } catch (e) {
      out.innerHTML = `<p class="warn">${esc(e.message)}</p>`;
      return;
    }
    const allPaces = rows.map((r) => r.pace.med);
    const avg = allPaces.reduce((a, b) => a + b, 0) / allPaces.length;
    let head = `<div class="plan-summary"><div><div class="label">Average target</div><div class="value">${paceKm(avg)}</div>
      <div class="sub">${paceMi(avg)}${treadmill ? ` · ${kmh(avg)} km/h · ${mph(avg)} mph at ${opts.incline}%` : ""}</div></div>`;
    if (rows[0].hr) {
      const hrs = rows.map((r) => r.hr.med);
      const lo = Math.round(Math.min(...hrs)), hi = Math.round(Math.max(...hrs));
      head += `<div><div class="label">Heart rate guide</div><div class="value">${lo === hi ? lo : `${lo}–${hi}`} bpm</div>
        <div class="sub">watch average per rep</div></div>`;
    }
    if (treadmill) {
      head += `<div><div class="label">Feels like outdoors at</div><div class="value">≈ ${outdoorEquivalentIncline().toFixed(1)}% incline</div>
        <div class="sub">by your own treadmill data</div></div>`;
    }
    head += "</div>";
    const repLabel = (r) =>
      r.rep.timeS != null ? `${mmss(r.rep.timeS)} min` : r.rep.distM >= 1000 ? `${(r.rep.distM / 1000).toFixed(2).replace(/\.?0+$/, "")} km` : `${Math.round(r.rep.distM)} m`;
    const body = rows
      .map(
        (r, i) => `<tr><td>${i + 1}</td><td>${repLabel(r)} <span class="muted">/ ${Math.round(r.rep.rest)} s</span></td>
        <td class="num">${paceKm(r.pace.med)}<br><span class="muted">${mmss(r.pace.lo)}–${mmss(r.pace.hi)}</span></td>
        <td class="num">${paceMi(r.pace.med)}</td>
        ${treadmill ? `<td class="num">${kmh(r.pace.med)} km/h<br><span class="muted">${mph(r.pace.med)} mph</span></td>` : ""}
        <td class="num">${r.rep.timeS != null ? `${((r.rep.timeS / r.pace.med) * 1000).toFixed(0)} m` : mmss(r.time)}</td>
        ${r.hr ? `<td class="num">${Math.round(r.hr.med)} bpm<br><span class="muted">${Math.round(r.hr.lo)}–${Math.round(r.hr.hi)}</span></td>` : ""}</tr>`
      )
      .join("");
    out.innerHTML = `${head}<div class="table-wrap"><table><tr><th>#</th><th>Rep / rest</th><th class="num">Pace (80% range)</th>
      <th class="num">per mile</th>${treadmill ? '<th class="num">Treadmill speed</th>' : ""}<th class="num">${rows[0].rep.timeS != null ? "Distance" : "Rep time"}</th>
      ${rows[0].hr ? '<th class="num">HR (80% range)</th>' : ""}</tr>${body}</table></div>`;
  }

  document.querySelectorAll("#planner input, #planner select").forEach((input) => input.addEventListener("input", renderPlan));
  renderPlan();

  // ---------- similar sessions ----------
  const sessions = [...data.sessions].sort((a, b) => (b.date + (b.period || "")).localeCompare(a.date + (a.period || "")));
  const byId = Object.fromEntries(sessions.map((s) => [s.session_id, s]));
  const picker = el("sim-session");
  sessions.forEach((s) =>
    picker.insertAdjacentHTML("beforeend", `<option value="${esc(s.session_id)}">${esc(s.date)} ${esc(s.period || "")} · ${esc(s.workout_type || "")}</option>`)
  );
  const ref = P.reference_lactate.toFixed(1);
  const SAME_THRESHOLD = 3; // s/km

  function renderSimilar() {
    const current = byId[picker.value];
    const matches = data.similar[current.session_id] || [];
    const out = el("sim-output");
    const describe = (s) => `${paceKm(s.mean_pace_s_per_km)} at ${s.mean_lactate.toFixed(1)} mmol/L`;
    let html = `<p class="note" style="margin-top:0.8rem">This session: <strong>${describe(current)}</strong> · ${esc(current.surface_group)} ·
      normalized ${paceKm(current.normalized_pace[ref])} @ ${ref} mmol/L</p>`;
    if (!matches.length) {
      out.innerHTML = html + `<p class="muted">No earlier sessions to compare with.</p>`;
      return;
    }
    // Verdict levels are explained in the legend under the table.
    const verdict = (m) => {
      if (Math.abs(m.normalized_diff) <= SAME_THRESHOLD) return "about the same";
      const dir = m.normalized_diff < 0 ? "better" : "worse";
      const corrections = Math.abs(m.conditions_correction) + Math.abs(m.curve_correction);
      return corrections <= Math.abs(m.normalized_diff) ? `clearly ${dir}` : `probably ${dir}`;
    };
    html += `<div class="table-wrap"><table><tr><th>Earlier session</th><th class="num">Ran</th><th class="num">Raw pace Δ</th>
      <th class="num">Conditions Δ</th><th class="num">Lactate-curve Δ</th><th class="num">Normalized Δ</th><th>Verdict</th></tr>` +
      matches
        .map(
          (m) => `<tr><td>${esc(m.date)} ${esc(m.period || "")}<br><span class="muted">${esc(m.workout_type || "")} · ${esc(m.surface_group)}</span></td>
          <td class="num">${describe(m)}</td>
          <td class="num">${signed(m.raw_pace_diff)}</td>
          <td class="num muted">${signed(m.conditions_correction)}</td>
          <td class="num muted">${signed(m.curve_correction)}</td>
          <td class="num"><strong>${signed(m.normalized_diff)}</strong></td>
          <td>${verdict(m)}</td></tr>`
        )
        .join("") +
      `</table></div>`;
    html += `<ul class="note legend">
      <li><strong>about the same</strong>: within ±${SAME_THRESHOLD} s/km after correction.</li>
      <li><strong>clearly better / worse</strong>: the difference is bigger than all the model's corrections together, so even
        if the model were somewhat off, the verdict would hold.</li>
      <li><strong>probably better / worse</strong>: the raw numbers alone don't show it; the verdict comes mainly from the
        corrections. Typical case: you ran a bit slower but with clearly lower lactate, and the lactate curve says that
        trade is worth more than the pace you gave up. If the curve or a condition effect is a bit off, this one could flip.</li>
      <li>Single sessions also vary about ±${Math.round(data.form_sd)} s/km from day to day (sleep, fatigue, measurement), so one
        comparison is a hint; a pattern over several sessions is what counts.</li>
    </ul>`;
    out.innerHTML = html;
  }
  picker.addEventListener("input", renderSimilar);
  renderSimilar();
})();
