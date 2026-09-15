"use strict";

(function exposeDiagnosticView(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.CustomDanceDiagnosticView = api;
})(typeof globalThis !== "undefined" ? globalThis : this, () => {
  function rkeSeries(rkeGroups, selectedGroups = []) {
    const entries = Object.entries(rkeGroups || {});
    if (!entries.length) return [];
    const frameCount = entries[0][1].length;
    if (entries.some(([, values]) => !Array.isArray(values) || values.length !== frameCount)) {
      throw new Error("RKE group curves must be aligned arrays");
    }
    const available = new Set(entries.map(([group]) => group));
    const selected = [...new Set(selectedGroups)].filter((group) => available.has(group));
    if (selected.length) {
      return selected.map((group) => ({
        key: `rke.${group}`,
        curve: `rke.${group}`,
        group,
        values: rkeGroups[group],
        peakCurves: [`rke.${group}`],
      }));
    }
    const values = Array.from({length: frameCount}, (_, frame) => (
      Math.max(0, ...entries.map(([, curve]) => Number(curve[frame]) || 0))
    ));
    return [{
      key: "rke",
      curve: "rke",
      group: null,
      values,
      peakCurves: entries.map(([group]) => `rke.${group}`),
    }];
  }

  function visiblePeakCurves(series) {
    return new Set(series.flatMap((item) => item.peakCurves || [item.curve]));
  }

  return Object.freeze({rkeSeries, visiblePeakCurves});
});
