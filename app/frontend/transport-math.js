"use strict";

(function exposeTransportMath(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.CustomDanceTransportMath = api;
})(typeof globalThis !== "undefined" ? globalThis : this, () => {
  const PLAYBACK_RATES = Object.freeze([0.5, 1, 1.5, 2]);

  function clamp(value, minimum, maximum) {
    return Math.min(maximum, Math.max(minimum, value));
  }

  function timelineTimeFromClientX(clientX, left, width, duration) {
    if (![clientX, left, width, duration].every(Number.isFinite)) {
      throw new Error("timeline pointer values must be finite");
    }
    if (width <= 0 || duration < 0) {
      throw new Error("timeline width must be positive and duration non-negative");
    }
    return clamp((clientX - left) / width, 0, 1) * duration;
  }

  function previewTime(audioTime, segmentStart, duration) {
    if (![audioTime, segmentStart, duration].every(Number.isFinite) || duration < 0) {
      throw new Error("preview timing values are invalid");
    }
    return clamp(audioTime - segmentStart, 0, duration);
  }

  function frameIndex(time, fps, frames) {
    if (![time, fps, frames].every(Number.isFinite) || fps <= 0 || frames < 1) {
      throw new Error("frame timing values are invalid");
    }
    return clamp(Math.floor(Math.max(0, time) * fps), 0, frames - 1);
  }

  function frameFromClientX(clientX, left, width, frames) {
    if (![clientX, left, width, frames].every(Number.isFinite)) {
      throw new Error("frame selection pointer values must be finite");
    }
    if (width <= 0 || !Number.isInteger(frames) || frames < 1) {
      throw new Error("selection width and frame count are invalid");
    }
    const normalized = clamp((clientX - left) / width, 0, 1);
    return Math.round(normalized * (frames - 1));
  }

  function frameRange(anchorFrame, currentFrame, frames) {
    if (![anchorFrame, currentFrame, frames].every(Number.isInteger)) {
      throw new Error("frame range values must be integers");
    }
    if (frames < 1 || anchorFrame < 0 || currentFrame < 0 || anchorFrame >= frames || currentFrame >= frames) {
      throw new Error("frame range must stay inside the motion");
    }
    return {
      startFrame: Math.min(anchorFrame, currentFrame),
      endFrame: Math.max(anchorFrame, currentFrame) + 1,
    };
  }

  function frameRulerTicks(boundaryFrames, viewportSeconds = 32) {
    if (!Number.isInteger(boundaryFrames) || boundaryFrames < 1) {
      throw new Error("ruler boundary must be a positive frame count");
    }
    const schemes = {
      16: {shortStep: 5, longStep: 30, labelStep: 60},
      32: {shortStep: 10, longStep: 60, labelStep: 120},
      48: {shortStep: 15, longStep: 90, labelStep: 180},
      64: {shortStep: 30, longStep: 120, labelStep: 240},
    };
    const scheme = schemes[viewportSeconds];
    if (!scheme) {
      throw new Error("ruler viewport must be 16, 32, 48, or 64 seconds");
    }
    const ticks = [];
    for (let frame = 0; frame <= boundaryFrames; frame += scheme.shortStep) {
      const labeled = frame % scheme.labelStep === 0;
      ticks.push({
        frame,
        label: labeled ? String(frame) : null,
        kind: labeled ? "label" : frame % scheme.longStep === 0 ? "long" : "short",
        isEnd: frame === boundaryFrames,
      });
    }
    if (boundaryFrames % scheme.shortStep !== 0) {
      ticks.push({frame: boundaryFrames, label: null, kind: "long", isEnd: true});
    }
    return ticks;
  }

  function playbackPolicy(value) {
    const rate = Number(value);
    if (!PLAYBACK_RATES.includes(rate)) {
      throw new Error("playback rate must be 0.5, 1, 1.5, or 2");
    }
    return {
      rate,
      muteAudio: rate !== 1,
      label: `${Number.isInteger(rate) ? rate.toFixed(0) : rate}×`,
    };
  }

  return Object.freeze({
    PLAYBACK_RATES,
    clamp,
    frameFromClientX,
    frameIndex,
    frameRulerTicks,
    frameRange,
    playbackPolicy,
    previewTime,
    timelineTimeFromClientX,
  });
});
