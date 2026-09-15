"use strict";

const VIEWER_MATH = window.CustomDanceViewerMath;
if (!VIEWER_MATH) throw new Error("viewer-math.js must load before app.js");
const TRANSPORT_MATH = window.CustomDanceTransportMath;
if (!TRANSPORT_MATH) throw new Error("transport-math.js must load before app.js");
const DIAGNOSTIC_VIEW = window.CustomDanceDiagnosticView;
if (!DIAGNOSTIC_VIEW) throw new Error("diagnostic-view.js must load before app.js");
const CANDIDATE_STATE = window.CustomDanceCandidateState;
if (!CANDIDATE_STATE) throw new Error("candidate-state.js must load before app.js");

const SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21];
const SMPL_NAMES = [
  "root", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
  "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
  "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
  "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
];
const FOOT_JOINTS = [7, 8, 10, 11];
const DIAGNOSTIC_COLORS = Object.freeze({
  ake: "#f8fafc",
  rke: "#22d3ee",
  torso: "#22d3ee",
  head: "#a78bfa",
  left_arm: "#fbbf24",
  right_arm: "#fb7185",
  left_leg: "#4ade80",
  right_leg: "#60a5fa",
});
const DIAGNOSTIC_LABELS = Object.freeze({
  ake: "AKE root translation",
  rke: "RKE joint rotation",
  torso: "torso",
  head: "head",
  left_arm: "left arm",
  right_arm: "right arm",
  left_leg: "left leg",
  right_leg: "right leg",
});
const JOINT_GROUP_LABELS = Object.freeze({
  head: "Head",
  torso: "Torso",
  left_arm: "Left Arm",
  right_arm: "Right Arm",
  left_leg: "Left Leg",
  right_leg: "Right Leg",
});
const TIMELINE_ZOOM_LEVELS = Object.freeze([16, 32, 48, 64]);
const CHAT_MAX_MESSAGES = 50;
const SLOT_DESCRIPTION_DURATION_MS = 3000;

const state = {
  sessionId: null,
  analysis: null,
  timeline: null,
  retrieval: [],
  currentMotion: null,
  previewMotion: null,
  previewContext: null,
  previewRequestId: 0,
  queryBySlot: new Map(),
  querySlotId: null,
  diagnostics: null,
  diagnosticSignals: new Set(["ake", "rke"]),
  selectedJointGroups: new Set(),
  favoriteIdsBySlot: new Map(),
  topK: 10,
  fixMode: false,
  rangeMode: false,
  fixSelection: null,
  rangePointerId: null,
  rangeAnchorFrame: null,
  zoomSeconds: 32,
  completed: false,
  rightPanel: "candidates",
  musicFilename: "",
  musicSizeBytes: 0,
  tempoBpm: null,
  sampleRate: null,
  chatOpen: false,
  chatUnread: 0,
  chatMessagesByKey: new Map(),
  playbackRate: 1,
  slotDescriptionTimer: 0,
  slotDescriptionSlotId: null,
  transportRaf: 0,
  seekPointerId: null,
  seekSurface: null,
  toastTimer: null,
  stage: null,
};

const el = Object.fromEntries([
  "music-file", "file-state", "audio-meta", "global-intent", "openai-controls",
  "external-consent", "analyze-button", "message-log", "audio-player",
  "timeline-panel", "timeline-shell", "timeline-content", "timeline-selection", "time-ruler", "slot-track", "playhead",
  "complete-button", "fix-toggle", "fix-range", "fix-selection-status", "smooth-button", "remake-button",
  "diagnose-button", "timeline-reset", "play-button", "pause-button",
  "export-pkl", "stage-canvas", "stage-time", "preview-label", "stop-preview", "diagnostic-timeline",
  "diagnostic-canvas", "diagnostic-empty", "diagnostic-legend", "slot-context",
  "motion-query", "motion-query-origin", "refresh-button", "motion-list", "result-count", "toast",
  "stage-shell", "stage-empty", "planning-chat", "planning-chat-close",
  "planning-chat-launcher", "planning-chat-unread",
  "viewer-status", "view-character", "reset-camera", "viewer-maximize", "viewport-panel",
  "candidates-tab", "repair-tab", "candidates-view", "repair-view", "repair-guidance",
  "range-select-toggle", "zoom-out", "zoom-in", "timeline-zoom-slider", "zoom-duration-label",
  "current-frame", "start-frame", "end-frame", "playback-speed", "playback-audio-state",
  "slot-description-popover", "slot-description-title", "slot-description-text",
].map((id) => [id, document.getElementById(id)]));

function icon(name) {
  return `<svg class="icon" aria-hidden="true"><use href="#icon-${name}"></use></svg>`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"]/g, (character) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"})[character]);
}

function toast(message, kind = "info") {
  window.clearTimeout(state.toastTimer);
  el.toast.textContent = message;
  el.toast.dataset.kind = kind;
  el.toast.hidden = false;
  state.toastTimer = window.setTimeout(() => { el.toast.hidden = true; }, 5200);
}

function addMessage(role, message, {key = null} = {}) {
  const existing = key ? state.chatMessagesByKey.get(key) : null;
  if (existing && existing.querySelector("p")?.textContent === message) return;
  const article = existing || document.createElement("article");
  article.className = `message message--${role}`;
  const roleLabel = role === "user" ? "USER" : "SYSTEM";
  article.innerHTML = `<span class="message-role">${roleLabel}</span><p>${escapeHtml(message)}</p>`;
  if (key) {
    article.dataset.messageKey = key;
    state.chatMessagesByKey.set(key, article);
  }
  if (!existing) el["message-log"].append(article);
  while (el["message-log"].children.length > CHAT_MAX_MESSAGES) {
    const oldest = el["message-log"].firstElementChild;
    if (oldest?.dataset.messageKey) state.chatMessagesByKey.delete(oldest.dataset.messageKey);
    oldest?.remove();
  }
  el["message-log"].scrollTop = el["message-log"].scrollHeight;
  if (!state.chatOpen && role === "system") {
    state.chatUnread += 1;
    updateChatBadge();
  }
}

function resetChatLog() {
  state.chatMessagesByKey.clear();
  el["message-log"].replaceChildren();
  state.chatUnread = 0;
  updateChatBadge();
}

function updateChatBadge() {
  const badge = el["planning-chat-unread"];
  badge.textContent = String(Math.min(99, state.chatUnread));
  badge.hidden = state.chatUnread === 0;
  el["planning-chat-launcher"].setAttribute(
    "aria-label",
    state.chatUnread
      ? `Open planning dialogue, ${state.chatUnread} unread update${state.chatUnread === 1 ? "" : "s"}`
      : "Open planning dialogue",
  );
}

function setChatOpen(open, {restoreFocus = true} = {}) {
  state.chatOpen = Boolean(open);
  el["planning-chat"].hidden = !state.chatOpen;
  el["planning-chat-launcher"].setAttribute("aria-expanded", String(state.chatOpen));
  if (state.chatOpen) {
    state.chatUnread = 0;
    updateChatBadge();
    el["message-log"].scrollTop = el["message-log"].scrollHeight;
    el["planning-chat-close"].focus({preventScroll: true});
  } else if (restoreFocus) {
    el["planning-chat-launcher"].focus({preventScroll: true});
  }
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  let payload = null;
  try { payload = await response.json(); } catch { payload = null; }
  if (!response.ok) {
    const detail = payload?.detail || `${response.status} ${response.statusText}`;
    throw new Error(detail);
  }
  return payload;
}

function setBusy(button, busy, label) {
  if (busy) {
    button.dataset.original = button.innerHTML;
    button.dataset.busy = "true";
    button.textContent = label;
    button.disabled = true;
  } else {
    button.innerHTML = button.dataset.original || button.innerHTML;
    delete button.dataset.busy;
    button.disabled = false;
    updateControls();
  }
}

function setControlDisabled(button, disabled) {
  button.disabled = Boolean(disabled) || button.dataset.busy === "true";
}

function focusedSlot() {
  const id = state.timeline?.current_focused_slot_id;
  return state.timeline?.slots.find((slot) => slot.slot_id === id) || null;
}

function saveFocusedQuery() {
  if (state.querySlotId) {
    state.queryBySlot.set(state.querySlotId, el["motion-query"].value);
  }
}

function defaultQueryForSlot(slot) {
  return slot?.cue?.trim() || el["global-intent"].value.trim();
}

function updateQueryOrigin(slot = focusedSlot()) {
  const badge = el["motion-query-origin"];
  const cue = slot?.cue?.trim();
  if (!cue) {
    badge.hidden = true;
    badge.classList.remove("is-edited");
    badge.removeAttribute("title");
    return;
  }
  const edited = el["motion-query"].value.trim() !== cue;
  badge.hidden = false;
  badge.classList.toggle("is-edited", edited);
  badge.textContent = edited ? "MLLM Cue · edited" : "MLLM Cue";
  badge.title = edited
    ? "This local intent was edited by the user."
    : "This editable cue was suggested by the music analysis model.";
}

function selectedSmoothTargets() {
  return {
    smoothTranslation: false,
    jointGroups: [...state.selectedJointGroups],
  };
}

function selectionHasSmoothAnchors() {
  return Boolean(
    state.fixSelection
    && state.currentMotion
    && state.fixSelection.startFrame > 0
    && state.fixSelection.endFrame < state.currentMotion.frames
  );
}

function canRepairMotion() {
  return Boolean(state.timeline && state.completed);
}

function updateControls() {
  const hasSession = Boolean(state.sessionId);
  const hasTimeline = Boolean(state.timeline);
  const canRepair = canRepairMotion();
  const focused = focusedSlot();
  const selectedGroupCount = state.selectedJointGroups.size;
  const canEditRange = canRepair && state.fixMode && state.rangeMode;
  setControlDisabled(el["analyze-button"], !hasSession);
  setControlDisabled(el["complete-button"], !hasTimeline);
  setControlDisabled(el["repair-tab"], !canRepair);
  setControlDisabled(el["fix-toggle"], !canRepair);
  setControlDisabled(el["diagnose-button"], !canRepair || !state.fixMode);
  setControlDisabled(
    el["smooth-button"],
    !canRepair
      || !state.fixMode
      || !state.rangeMode
      || !selectionHasSmoothAnchors()
      || selectedGroupCount !== 1,
  );
  setControlDisabled(
    el["remake-button"],
    !canRepair
      || !state.fixMode
      || !state.rangeMode
      || !state.fixSelection
      || !state.currentMotion
      || selectedGroupCount < 2,
  );
  setControlDisabled(el["export-pkl"], !hasTimeline);
  setControlDisabled(el["timeline-reset"], !hasTimeline);
  setControlDisabled(el["play-button"], !hasTimeline);
  setControlDisabled(el["pause-button"], !hasTimeline);
  setControlDisabled(el["range-select-toggle"], !canRepair || !state.fixMode);
  el["current-frame"].disabled = !hasTimeline;
  el["start-frame"].disabled = !canEditRange;
  el["end-frame"].disabled = !canEditRange;
  el["zoom-in"].disabled = state.zoomSeconds === TIMELINE_ZOOM_LEVELS[0];
  el["zoom-out"].disabled = state.zoomSeconds === TIMELINE_ZOOM_LEVELS.at(-1);
  setControlDisabled(el["motion-query"], !focused || focused.status === "invalid");
  setControlDisabled(
    el["refresh-button"],
    !focused || focused.status === "invalid" || !el["motion-query"].value.trim(),
  );

  const selectedLabels = [...state.selectedJointGroups].map((key) => JOINT_GROUP_LABELS[key]);
  el["smooth-button"].textContent = selectedGroupCount === 1
    ? `Smooth ${selectedLabels[0]}`
    : "Smooth";
  el["remake-button"].textContent = "Remake";
  el["smooth-button"].classList.toggle("is-primary", selectedGroupCount === 1);
  el["remake-button"].classList.toggle("is-primary", selectedGroupCount >= 2);
  if (selectedGroupCount === 0) {
    el["repair-guidance"].textContent = "Select a joint group.";
  } else if (selectedGroupCount === 1) {
    el["repair-guidance"].textContent = "Smooth selected group.";
  } else {
    el["repair-guidance"].textContent = "Remake whole body in selected range.";
  }
}

function renderCandidateContext() {
  const slot = focusedSlot();
  if (!slot) {
    state.querySlotId = null;
    el["slot-context"].textContent = "Select a timeline segment";
    el["motion-query"].value = "";
    el["motion-query"].placeholder = "Select a timeline segment first";
  } else {
    if (state.querySlotId !== slot.slot_id) {
      state.querySlotId = slot.slot_id;
      el["motion-query"].value = state.queryBySlot.has(slot.slot_id)
        ? state.queryBySlot.get(slot.slot_id)
        : defaultQueryForSlot(slot);
    }
    el["slot-context"].textContent = `${slot.start_sec.toFixed(1)}–${(slot.start_sec + slot.duration_sec).toFixed(1)} s`;
    el["motion-query"].placeholder = "Describe the movement for this segment";
  }
  updateQueryOrigin(slot);
  updateControls();
}

function timelineFps() {
  return Number.isFinite(state.currentMotion?.fps) && state.currentMotion.fps > 0
    ? state.currentMotion.fps
    : 30;
}

function timelineFrameCount() {
  if (Number.isInteger(state.currentMotion?.frames) && state.currentMotion.frames > 0) {
    return state.currentMotion.frames;
  }
  return Math.max(1, Math.round(projectDuration() * timelineFps()));
}

function timelineFrameAtTime(time) {
  return TRANSPORT_MATH.frameIndex(
    Math.max(0, Number(time) || 0),
    timelineFps(),
    timelineFrameCount(),
  );
}

function frameLabel(frame, frameCount = timelineFrameCount()) {
  const digits = Math.max(3, String(Math.max(0, frameCount - 1)).length);
  return `F${String(frame).padStart(digits, "0")}`;
}

function renderRuler(duration) {
  el["time-ruler"].replaceChildren();
  const fps = timelineFps();
  const boundaryFrames = Math.max(1, Math.round(duration * fps));
  const ticks = TRANSPORT_MATH.frameRulerTicks(boundaryFrames, state.zoomSeconds);
  for (const rulerTick of ticks) {
    const tick = document.createElement("div");
    tick.className = "ruler-tick";
    tick.classList.add(`is-${rulerTick.kind}`);
    if (rulerTick.isEnd) tick.classList.add("is-end");
    if (rulerTick.label === null) tick.classList.add("is-unlabeled");
    tick.style.left = rulerTick.isEnd
      ? "calc(100% - 1px)"
      : `${rulerTick.frame / boundaryFrames * 100}%`;
    if (rulerTick.label !== null) {
      const label = document.createElement("span");
      label.textContent = rulerTick.label;
      tick.append(label);
    }
    el["time-ruler"].append(tick);
  }
}

function updateTimelineScale() {
  const shellWidth = Math.max(1, el["timeline-shell"].clientWidth);
  const duration = projectDuration();
  const contentWidth = duration > 0
    ? Math.max(780, Math.ceil(shellWidth * Math.max(1, duration / state.zoomSeconds)))
    : Math.max(780, shellWidth);
  el["timeline-content"].style.width = `${contentWidth}px`;
  el["timeline-zoom-slider"].value = String(TIMELINE_ZOOM_LEVELS.indexOf(state.zoomSeconds));
  el["zoom-duration-label"].textContent = `${state.zoomSeconds} s`;
}

function renderSelectionOverlay() {
  const selection = state.fixSelection;
  const frameCount = timelineFrameCount();
  const visible = Boolean(state.rangeMode && selection && frameCount > 0);
  el["timeline-selection"].hidden = !visible;
  if (visible) {
    el["timeline-selection"].style.left = `${selection.startFrame / frameCount * 100}%`;
    el["timeline-selection"].style.width = `${(selection.endFrame - selection.startFrame) / frameCount * 100}%`;
  }
  syncFrameInputs();
}

function analysisSegmentForSlot(slot) {
  return [...(state.analysis?.segments || [])]
    .map((segment) => ({
      segment,
      overlap: Math.max(
        0,
        Math.min(slot.start_sec + slot.duration_sec, segment.end_sec)
          - Math.max(slot.start_sec, segment.start_sec),
      ),
    }))
    .filter((candidate) => candidate.overlap > 0)
    .sort((left, right) => right.overlap - left.overlap)[0]?.segment || null;
}

function musicDescriptionForSlot(slot) {
  const analysisSlot = state.analysis?.slots?.find((candidate) => candidate.slot_id === slot.slot_id);
  const explicit = analysisSlot?.music_description?.trim() || slot.music_description?.trim();
  const segment = analysisSegmentForSlot(slot);
  return {
    description: explicit || segment?.description?.trim() || "No music description is available for this phrase.",
    energy: segment?.energy || null,
  };
}

function hideSlotDescription(slotId = null) {
  if (slotId && state.slotDescriptionSlotId !== slotId) return;
  window.clearTimeout(state.slotDescriptionTimer);
  state.slotDescriptionTimer = 0;
  state.slotDescriptionSlotId = null;
  el["slot-description-popover"].hidden = true;
}

function showSlotDescription(slot, slotIndex, button, event) {
  hideSlotDescription();
  const detail = musicDescriptionForSlot(slot);
  const energy = detail.energy ? ` · ${detail.energy[0].toUpperCase()}${detail.energy.slice(1)} energy` : "";
  el["slot-description-title"].textContent = `Segment ${String(slotIndex + 1).padStart(2, "0")} · Music${energy}`;
  el["slot-description-text"].textContent = detail.description;
  const buttonRect = button.getBoundingClientRect();
  const pointerX = event.detail === 0 || !Number.isFinite(event.clientX) || event.clientX <= 0
    ? buttonRect.left + buttonRect.width / 2
    : event.clientX;
  const pointerY = event.detail === 0 || !Number.isFinite(event.clientY) || event.clientY <= 0
    ? buttonRect.top + buttonRect.height / 2
    : event.clientY;
  const popover = el["slot-description-popover"];
  popover.hidden = false;
  const margin = 10;
  const gap = 12;
  const popoverRect = popover.getBoundingClientRect();
  const left = TRANSPORT_MATH.clamp(
    pointerX + gap,
    margin,
    Math.max(margin, window.innerWidth - popoverRect.width - margin),
  );
  const preferredTop = pointerY + gap;
  const top = preferredTop + popoverRect.height + margin <= window.innerHeight
    ? preferredTop
    : Math.max(margin, pointerY - popoverRect.height - gap);
  popover.style.left = `${left}px`;
  popover.style.top = `${top}px`;
  state.slotDescriptionSlotId = slot.slot_id;
  state.slotDescriptionTimer = window.setTimeout(
    () => hideSlotDescription(slot.slot_id),
    SLOT_DESCRIPTION_DURATION_MS,
  );
}

function renderTimeline() {
  const track = el["slot-track"];
  track.querySelectorAll(".slot-button, .empty-state").forEach((node) => node.remove());
  if (!state.analysis || !state.timeline) {
    hideSlotDescription();
    el["time-ruler"].replaceChildren();
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "Analyze music to reveal four-second candidate slots.";
    track.prepend(empty);
    el.playhead.hidden = true;
    setCompletionState(false);
    el["timeline-panel"].classList.remove("is-complete");
    updateTimelineScale();
    renderSelectionOverlay();
    renderCandidateContext();
    return;
  }
  updateTimelineScale();
  renderRuler(state.analysis.duration_sec);
  const laneEnds = [];
  const laneBySlot = new Map();
  const indexedSlots = [...state.timeline.slots]
    .sort((left, right) => left.start_sec - right.start_sec || left.slot_id.localeCompare(right.slot_id))
    .map((slot, slotIndex) => ({slot, slotIndex}));
  const visibleSlots = state.completed
    ? indexedSlots.filter(({slot}) => slot.status === "filled" || slot.status === "accepted")
    : state.fixMode
      ? indexedSlots.filter(({slot}) => slot.status !== "invalid")
      : indexedSlots;
  for (const {slot} of visibleSlots) {
    let lane = state.completed
      ? 0
      : laneEnds.findIndex((end) => slot.start_sec >= end - 1e-9);
    if (lane < 0) lane = laneEnds.length;
    laneEnds[lane] = slot.start_sec + slot.duration_sec;
    laneBySlot.set(slot.slot_id, lane);
  }
  const laneCount = Math.max(1, laneEnds.length);
  const laneHeight = state.completed ? 54 : Math.min(46, Math.max(20, 96 / laneCount));
  for (const {slot, slotIndex} of visibleSlots) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "slot-button";
    button.dataset.slotId = slot.slot_id;
    if (slot.slot_id === state.timeline.current_focused_slot_id) button.classList.add("is-focused");
    button.dataset.status = slot.status;
    button.disabled = slot.status === "invalid";
    button.style.top = `${4 + laneBySlot.get(slot.slot_id) * laneHeight}px`;
    button.style.height = `${Math.max(18, laneHeight - 5)}px`;
    button.style.left = `${slot.start_sec / state.analysis.duration_sec * 100}%`;
    button.style.width = `${slot.duration_sec / state.analysis.duration_sec * 100}%`;
    const startFrame = Math.round(slot.start_sec * timelineFps());
    const endFrame = Math.round((slot.start_sec + slot.duration_sec) * timelineFps());
    button.setAttribute("aria-label", `Segment ${slotIndex + 1}, ${slot.status}, frames ${startFrame} to ${endFrame - 1}`);
    button.setAttribute("aria-describedby", "slot-description-popover");
    button.innerHTML = `<strong>Segment ${String(slotIndex + 1).padStart(2, "0")}</strong><small>${frameLabel(startFrame)}–${frameLabel(Math.max(startFrame, endFrame - 1))} · ${slot.status}</small>`;
    button.addEventListener("click", (event) => {
      if (!state.rangeMode) {
        showSlotDescription(slot, slotIndex, button, event);
        selectSlot(slot.slot_id);
      }
    });
    button.addEventListener("pointerleave", () => hideSlotDescription(slot.slot_id));
    track.append(button);
  }
  el.playhead.hidden = false;
  el["timeline-panel"].classList.toggle("is-complete", state.completed);
  updatePlayhead();
  renderSelectionOverlay();
  renderCandidateContext();
}

async function selectSlot(slotId) {
  saveFocusedQuery();
  try {
    const result = await api(`/api/sessions/${state.sessionId}/slots/${encodeURIComponent(slotId)}/focus`, {method: "POST"});
    state.timeline = result;
    state.retrieval = [];
    renderTimeline();
    renderMotionList();
  } catch (error) {
    toast(error.message, "error");
  }
}

function renderMotionList() {
  el["motion-list"].replaceChildren();
  const focusedId = state.timeline?.current_focused_slot_id;
  const favoriteIds = focusedId
    ? CANDIDATE_STATE.favoritesForSlot(state.favoriteIdsBySlot, focusedId)
    : new Set();
  const visibleCandidates = state.retrieval
    .slice(0, state.topK)
    .sort((left, right) => (
      Number(favoriteIds.has(right.clip_id)) - Number(favoriteIds.has(left.clip_id))
      || left._candidateNumber - right._candidateNumber
    ));
  el["result-count"].textContent = `${visibleCandidates.length} candidates`;
  if (!state.retrieval.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "Select a phrase slot and refresh to retrieve local motion candidates.";
    el["motion-list"].append(empty);
    return;
  }
  const canUse = focusedSlot()?.status === "accepted";
  visibleCandidates.forEach((motion) => {
    const candidateNumber = motion._candidateNumber;
    const favorite = favoriteIds.has(motion.clip_id);
    const card = document.createElement("article");
    card.className = "motion-card";
    const metadata = [
      motion.genre_name || `Genre ${motion.genre ?? "unknown"}`,
      `${motion.duration.toFixed(1)} seconds`,
      `score ${motion.score.toFixed(5)}`,
      motion.clip_id,
    ].join(" · ");
    card.title = metadata;
    card.setAttribute("aria-label", `Candidate ${candidateNumber}. ${metadata}`);
    card.innerHTML = `
      <div class="motion-card-header">
        <span class="motion-card-number">${String(candidateNumber).padStart(2, "0")}</span>
        <button class="candidate-star" type="button" aria-label="${favorite ? "Remove" : "Add"} candidate ${candidateNumber} ${favorite ? "from" : "to"} favorites" aria-pressed="${favorite}">${icon("star")}</button>
      </div>
      <span class="motion-card-spacer" aria-hidden="true"></span>
      <div class="motion-actions">
        <button class="button preview-action" type="button" aria-label="Preview candidate ${candidateNumber} with music">${icon("play")} Preview</button>
        <button class="button use-action" type="button" aria-label="Use candidate ${candidateNumber} in the selected phrase" ${canUse ? "" : "disabled"}>${icon("fill")} Use</button>
      </div>`;
    card.querySelector(".candidate-star").addEventListener("click", () => {
      if (!focusedId) return;
      CANDIDATE_STATE.toggleFavorite(
        state.favoriteIdsBySlot,
        focusedId,
        motion.clip_id,
      );
      renderMotionList();
    });
    card.querySelector(".preview-action").addEventListener("click", () => previewMotion(motion, candidateNumber));
    card.querySelector(".use-action").addEventListener("click", () => fillMotion(motion));
    el["motion-list"].append(card);
  });
}

async function previewMotion(item, candidateNumber) {
  const slot = focusedSlot();
  if (!slot || !state.analysis) {
    toast("Select a timeline segment before previewing a candidate.", "error");
    return;
  }
  const requestId = ++state.previewRequestId;
  const previousContext = state.previewContext;
  const previousTime = previousContext?.previousTime ?? el["audio-player"].currentTime;
  const wasPlaying = previousContext?.wasPlaying ?? !el["audio-player"].paused;
  try {
    const payload = await api(`/api/sessions/${state.sessionId}/motions/${encodeURIComponent(item.clip_id)}/preview`);
    if (requestId !== state.previewRequestId) return;
    const startSec = slot.start_sec;
    const endSec = Math.min(
      state.analysis.duration_sec,
      slot.start_sec + slot.duration_sec,
      slot.start_sec + payload.duration_sec,
    );
    if (endSec <= startSec) throw new Error("The selected preview segment is empty.");
    el["audio-player"].pause();
    state.previewMotion = payload;
    state.previewContext = {startSec, endSec, previousTime, wasPlaying, candidateNumber};
    state.stage?.setMotion(payload, {fitCamera: true});
    el["preview-label"].textContent = `Preview · Candidate ${String(candidateNumber).padStart(2, "0")}`;
    el["stop-preview"].hidden = false;
    setAudioTime(startSec, {internal: true});
    try {
      await el["audio-player"].play();
      startTransportLoop();
    } catch {
      toast("The preview is ready, but the browser blocked audio playback. Press Play in the audio control.", "error");
    }
  } catch (error) {
    toast(error.message, "error");
  }
}

function stopPreview({restoreTransport = true} = {}) {
  state.previewRequestId += 1;
  const context = state.previewContext;
  state.previewMotion = null;
  state.previewContext = null;
  if (state.currentMotion) state.stage?.setMotion(state.currentMotion, {fitCamera: true});
  el["preview-label"].textContent = "Current choreography";
  el["stop-preview"].hidden = true;
  if (context && restoreTransport) {
    el["audio-player"].pause();
    setAudioTime(context.previousTime, {internal: true});
    if (context.wasPlaying) {
      el["audio-player"].play().then(startTransportLoop).catch(() => {});
    }
  }
}

function updateFixSelectionStatus() {
  const output = el["fix-selection-status"];
  if (!state.fixMode) {
    output.textContent = "Open Repair to choose a frame range.";
  } else if (!state.rangeMode) {
    output.textContent = "Activate range selection in the Timeline toolbar.";
  } else if (state.fixSelection) {
    const {startFrame, endFrame} = state.fixSelection;
    const duration = (endFrame - startFrame) / timelineFps();
    const anchorHint = selectionHasSmoothAnchors() ? "" : " · Smooth needs one frame outside each edge";
    output.textContent = `${frameLabel(startFrame)}–${frameLabel(endFrame - 1)} · ${duration.toFixed(2)} s${anchorHint}`;
  } else {
    output.textContent = "Drag on the Timeline or enter Start and End frames.";
  }
}

function resetFixSelection({redraw = true} = {}) {
  state.fixSelection = null;
  state.rangePointerId = null;
  state.rangeAnchorFrame = null;
  updateFixSelectionStatus();
  renderSelectionOverlay();
  updateControls();
  if (redraw && state.diagnostics) {
    window.requestAnimationFrame(drawDiagnostics);
  }
}

function setRightPanel(target, {syncFixMode = true} = {}) {
  const normalized = target === "repair" && canRepairMotion() ? "repair" : "candidates";
  state.rightPanel = normalized;
  el["candidates-tab"].setAttribute("aria-selected", String(normalized === "candidates"));
  el["repair-tab"].setAttribute("aria-selected", String(normalized === "repair"));
  el["candidates-view"].hidden = normalized !== "candidates";
  el["repair-view"].hidden = normalized !== "repair";
  if (syncFixMode && normalized === "repair" && !state.fixMode) {
    setFixMode(true, {syncPanel: false});
  } else if (syncFixMode && normalized === "candidates" && state.fixMode) {
    setFixMode(false, {syncPanel: false});
  }
}

function setFixMode(active, {syncPanel = true} = {}) {
  const enabled = Boolean(active && canRepairMotion());
  state.fixMode = enabled;
  el["fix-toggle"].setAttribute("aria-pressed", String(enabled));
  clearDiagnostics();
  if (!enabled) setRangeMode(false, {clearSelection: true});
  if (syncPanel) setRightPanel(enabled ? "repair" : "candidates", {syncFixMode: false});
  renderTimeline();
  updateControls();
}

function clearDiagnostics() {
  state.diagnostics = null;
  el["diagnostic-timeline"].classList.remove("has-data");
  el["diagnostic-empty"].textContent = state.fixMode
    ? "Run Diagnose to display AKE and RKE curves."
    : "Enter Repair, then run Diagnose to display AKE and RKE curves.";
  const context = el["diagnostic-canvas"].getContext("2d");
  context?.clearRect(0, 0, el["diagnostic-canvas"].width, el["diagnostic-canvas"].height);
  renderDiagnosticLegend([]);
}

function setCompletionState(completed) {
  state.completed = Boolean(completed && state.timeline);
  if (!state.completed) {
    state.fixMode = false;
    el["fix-toggle"].setAttribute("aria-pressed", "false");
    clearDiagnostics();
    setRangeMode(false, {clearSelection: true});
    setRightPanel("candidates", {syncFixMode: false});
  }
  updateControls();
}

async function loadCurrentMotion() {
  state.currentMotion = await api(`/api/sessions/${state.sessionId}/motion/current`);
  if (!state.previewMotion) state.stage?.setMotion(state.currentMotion, {fitCamera: true});
}

async function fillMotion(item) {
  const targetSlotId = focusedSlot()?.slot_id || "unknown";
  try {
    const response = await api(`/api/sessions/${state.sessionId}/fill`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({clip_id: item.clip_id}),
    });
    state.timeline = response.timeline;
    setCompletionState(false);
    await loadCurrentMotion();
    stopPreview();
    renderTimeline();
    renderMotionList();
    addMessage(
      "system",
      "The selected candidate was applied to this four-second segment.",
      {key: `candidate-use-${targetSlotId}`},
    );
    toast("Candidate applied to the selected segment.", "success");
    return true;
  } catch (error) {
    toast(error.message, "error");
    return false;
  }
}

async function importMusic(file) {
  const form = new FormData();
  form.append("file", file);
  el["file-state"].textContent = "Uploading locally…";
  try {
    const response = await api("/api/sessions", {method: "POST", body: form});
    hideSlotDescription();
    stopPreview({restoreTransport: false});
    el["audio-player"].pause();
    state.sessionId = response.session_id;
    state.analysis = null;
    state.timeline = null;
    state.retrieval = [];
    state.favoriteIdsBySlot.clear();
    state.selectedJointGroups.clear();
    state.queryBySlot.clear();
    state.querySlotId = null;
    state.currentMotion = null;
    setCompletionState(false);
    state.musicFilename = response.original_filename;
    state.musicSizeBytes = response.size_bytes;
    state.tempoBpm = null;
    state.sampleRate = null;
    state.stage?.clearMotion();
    el["audio-player"].src = `${response.audio_url}?v=${Date.now()}`;
    renderAudioSummary();
    resetChatLog();
    addMessage(
      "system",
      `Imported ${response.original_filename}.`,
      {key: "music-import"},
    );
    renderTimeline();
    renderMotionList();
    updateControls();
  } catch (error) {
    el["file-state"].textContent = "Import failed";
    el["audio-meta"].textContent = error.message;
    toast(error.message, "error");
  }
}

function renderAudioSummary() {
  if (!state.musicFilename) {
    el["file-state"].textContent = "No music imported";
    el["audio-meta"].textContent = "WAV · 4–34 seconds · up to 20 MB";
    return;
  }
  const bpm = Number.isFinite(state.tempoBpm) ? ` · ${Math.round(state.tempoBpm)} BPM` : "";
  el["file-state"].textContent = `${state.musicFilename}${bpm}`;
  const metadata = [];
  const duration = projectDuration();
  if (duration > 0) metadata.push(`${duration.toFixed(1)} s`);
  if (Number.isFinite(state.sampleRate)) metadata.push(`${Math.round(state.sampleRate / 1000)} kHz`);
  if (state.musicSizeBytes > 0) metadata.push(`${(state.musicSizeBytes / 1024 / 1024).toFixed(2)} MB`);
  if (!Number.isFinite(state.tempoBpm)) metadata.push("BPM available after analysis");
  el["audio-meta"].textContent = metadata.join(" · ");
}

async function analyze() {
  if (state.analysis && !window.confirm("Re-analyzing resets every slot and filled motion. Continue?")) return;
  const payload = {
    mode: "openai",
    global_intent: el["global-intent"].value,
    confirm_reset: Boolean(state.analysis),
    consent_to_external_api: el["external-consent"].checked,
  };
  setBusy(el["analyze-button"], true, "Analyzing…");
  try {
    const response = await api(`/api/sessions/${state.sessionId}/analyze`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    state.analysis = response.analysis;
    state.timeline = response.timeline;
    state.retrieval = [];
    state.favoriteIdsBySlot.clear();
    state.queryBySlot.clear();
    state.querySlotId = null;
    setCompletionState(false);
    state.tempoBpm = Number.isFinite(response.tempo_bpm) ? response.tempo_bpm : state.tempoBpm;
    state.sampleRate = Number.isFinite(response.sample_rate) ? response.sample_rate : state.sampleRate;
    stopPreview({restoreTransport: false});
    await loadCurrentMotion();
    renderAudioSummary();
    renderTimeline();
    renderMotionList();
    addMessage(
      "system",
      `${response.analysis.summary} ${response.analysis.slots.length} slots ready.`,
      {key: "analysis-result"},
    );
    for (const [index, warning] of (response.warnings || []).entries()) {
      addMessage("system", `Analyze notice: ${warning}`, {key: `analysis-warning-${index}`});
    }
    toast("Analysis complete. Select a candidate slot.", "success");
  } catch (error) {
    addMessage("system", `Analyze failed: ${error.message}`, {key: "analysis-error"});
    toast(error.message, "error");
  } finally {
    setBusy(el["analyze-button"], false);
  }
}

async function retrieve() {
  if (!el["external-consent"].checked) {
    toast("Consent to API processing is required to interpret the current intent.", "error");
    return;
  }
  const query = el["motion-query"].value.trim();
  if (!query) return;
  setBusy(el["refresh-button"], true, "Matching music + text…");
  el["motion-list"].innerHTML = '<p class="empty-state">Matching music and text against the local index…</p>';
  try {
    const response = await api(`/api/sessions/${state.sessionId}/retrieve`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({query, top_k: 15}),
    });
    state.retrieval = response.items.map((item, index) => ({
      ...item,
      _candidateNumber: index + 1,
    }));
    renderMotionList();
    if (response.warning) {
      addMessage(
        "system",
        response.warning,
        {key: `retrieval-warning-${focusedSlot()?.slot_id || "unknown"}`},
      );
    }
  } catch (error) {
    addMessage(
      "system",
      `Motion retrieval failed: ${error.message}`,
      {key: `retrieval-error-${focusedSlot()?.slot_id || "unknown"}`},
    );
    toast(error.message, "error");
    renderMotionList();
  } finally {
    setBusy(el["refresh-button"], false);
  }
}

async function completeMotion() {
  setBusy(el["complete-button"], true, "Running Completer…");
  try {
    const response = await api(`/api/sessions/${state.sessionId}/complete`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({}),
    });
    state.timeline = response.timeline;
    setCompletionState(true);
    clearDiagnostics();
    await loadCurrentMotion();
    stopPreview({restoreTransport: false});
    renderTimeline();
    addMessage(
      "system",
      "Complete finished.",
      {key: "complete-result"},
    );
    toast("Complete finished.", "success");
  } catch (error) {
    toast(error.message, "error");
  } finally {
    setBusy(el["complete-button"], false);
  }
}

async function remakeMotion() {
  const selection = state.fixSelection;
  if (
    !canRepairMotion()
    || !state.fixMode
    || !selection
    || !state.currentMotion
    || selection.startFrame < 0
    || selection.endFrame <= selection.startFrame
    || selection.endFrame > state.currentMotion.frames
  ) {
    toast("Activate range selection and choose a non-empty frame range.", "error");
    return;
  }
  const {startFrame, endFrame} = selection;
  setBusy(el["remake-button"], true, "Remaking…");
  try {
    const response = await api(`/api/sessions/${state.sessionId}/remake`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({start_frame: startFrame, end_frame: endFrame}),
    });
    state.timeline = response.timeline;
    await loadCurrentMotion();
    stopPreview({restoreTransport: false});
    renderTimeline();
    clearDiagnostics();
    if (state.fixMode) await diagnoseMotion({announce: false});
    addMessage(
      "system",
      "Remake finished.",
      {key: `remake-result-${startFrame}-${endFrame}`},
    );
    toast("Remake finished.", "success");
  } catch (error) {
    toast(error.message, "error");
  } finally {
    setBusy(el["remake-button"], false);
  }
}

async function smoothMotion() {
  const selection = state.fixSelection;
  const targets = selectedSmoothTargets();
  if (!canRepairMotion() || !state.fixMode || !selection || !selectionHasSmoothAnchors()) {
    toast("Smooth needs a non-empty range with one anchor frame on both sides.", "error");
    return;
  }
  setBusy(el["smooth-button"], true, "Smoothing…");
  try {
    const response = await api(`/api/sessions/${state.sessionId}/smooth`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        start_frame: selection.startFrame,
        end_frame: selection.endFrame,
        smooth_translation: targets.smoothTranslation,
        joint_groups: targets.jointGroups,
      }),
    });
    state.timeline = response.timeline;
    await loadCurrentMotion();
    stopPreview({restoreTransport: false});
    renderTimeline();
    clearDiagnostics();
    if (state.fixMode) await diagnoseMotion({announce: false});
    addMessage(
      "system",
      "Smooth finished.",
      {key: `smooth-result-${selection.startFrame}-${selection.endFrame}`},
    );
    toast("Smooth finished.", "success");
  } catch (error) {
    toast(error.message, "error");
  } finally {
    setBusy(el["smooth-button"], false);
  }
}

function diagnosticSeries() {
  if (!state.diagnostics) return [];
  const peakFallback = (matches, length) => {
    const values = Array.from({length}, () => 0);
    for (const peak of state.diagnostics.peaks.filter(matches)) {
      const emphasized = TRANSPORT_MATH.clamp(
        (peak.robust_z - 2) / 10,
        0,
        1,
      );
      values[peak.frame_index] = Math.max(values[peak.frame_index], emphasized);
      if (peak.frame_index > 0) values[peak.frame_index - 1] = Math.max(values[peak.frame_index - 1], emphasized * 0.28);
      if (peak.frame_index + 1 < length) values[peak.frame_index + 1] = Math.max(values[peak.frame_index + 1], emphasized * 0.28);
    }
    return values;
  };
  const akeValues = state.diagnostics.ake_anomaly
    || peakFallback((peak) => peak.curve === "ake", state.diagnostics.frame_count);
  const rkeGroups = state.diagnostics.rke_anomaly
    || Object.fromEntries(Object.keys(state.diagnostics.rke).map((key) => [
      key,
      peakFallback((peak) => peak.curve === `rke.${key}`, state.diagnostics.frame_count),
    ]));
  const series = [];
  if (state.diagnosticSignals.has("ake")) {
    series.push({
      key: "ake",
      curve: "ake",
      values: akeValues,
      color: DIAGNOSTIC_COLORS.ake,
      label: "AKE",
      peakCurves: ["ake"],
    });
  }
  if (state.diagnosticSignals.has("rke")) {
    for (const item of DIAGNOSTIC_VIEW.rkeSeries(rkeGroups, [...state.selectedJointGroups])) {
      series.push({
        ...item,
        color: DIAGNOSTIC_COLORS[item.group || "rke"],
        label: item.group ? `RKE ${JOINT_GROUP_LABELS[item.group]}` : "RKE Overview",
      });
    }
  }
  return series;
}

function renderDiagnosticLegend(series = diagnosticSeries()) {
  const legend = el["diagnostic-legend"];
  legend.replaceChildren();
  legend.hidden = !state.diagnostics || series.length === 0;
  for (const item of series) {
    const label = document.createElement("span");
    label.style.setProperty("--series-color", item.color);
    label.textContent = item.label;
    legend.append(label);
  }
}

function updateDiagnosticControls() {
  document.querySelectorAll("[data-diagnostic-signal]").forEach((button) => {
    button.setAttribute("aria-pressed", String(state.diagnosticSignals.has(button.dataset.diagnosticSignal)));
  });
  document.querySelectorAll("[data-joint-group]").forEach((button) => {
    const selected = state.selectedJointGroups.has(button.dataset.jointGroup);
    const image = button.querySelector("img");
    button.setAttribute("aria-pressed", String(selected));
    image.src = selected ? image.dataset.selectedSrc : image.dataset.baseSrc;
  });
  if (state.diagnostics && state.diagnosticSignals.size === 0) {
    el["diagnostic-timeline"].classList.remove("has-data");
    el["diagnostic-empty"].textContent = "AKE and RKE curves are hidden from the Repair panel.";
  } else if (state.diagnostics) {
    el["diagnostic-timeline"].classList.add("has-data");
    el["diagnostic-empty"].textContent = "";
  }
  renderDiagnosticLegend();
  updateFixSelectionStatus();
  updateControls();
}

function drawDiagnostics() {
  if (!state.diagnostics) return;
  const canvas = el["diagnostic-canvas"];
  const {width, height, ratio} = sizeCanvas(canvas);
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, width, height);
  const margin = {left: 2 * ratio, right: 2 * ratio, top: 5 * ratio, bottom: 5 * ratio};
  const plotWidth = Math.max(1, width - margin.left - margin.right);
  const plotHeight = Math.max(1, height - margin.top - margin.bottom);
  const series = diagnosticSeries();
  renderDiagnosticLegend(series);
  const visiblePeakCurves = DIAGNOSTIC_VIEW.visiblePeakCurves(series);
  const peaks = state.diagnostics.peaks.filter((peak) => visiblePeakCurves.has(peak.curve));

  context.strokeStyle = "rgba(148,163,184,.16)";
  context.lineWidth = ratio;
  for (const normalized of [0, 0.5, 1]) {
    const y = margin.top + plotHeight * (1 - normalized);
    context.beginPath();
    context.moveTo(margin.left, y);
    context.lineTo(width - margin.right, y);
    context.stroke();
  }

  context.strokeStyle = "rgba(251,113,133,.28)";
  context.lineWidth = ratio;
  for (const frame of new Set(peaks.map((peak) => peak.frame_index))) {
    const x = margin.left + plotWidth * frame / Math.max(1, state.diagnostics.frame_count - 1);
    context.beginPath();
    context.moveTo(x, margin.top);
    context.lineTo(x, height - margin.bottom);
    context.stroke();
  }

  for (const item of series) {
    context.strokeStyle = item.color;
    context.lineWidth = (item.key === "ake" ? 2.2 : 1.55) * ratio;
    context.lineJoin = "round";
    context.lineCap = "round";
    context.beginPath();
    item.values.forEach((value, index) => {
      const x = margin.left + plotWidth * index / Math.max(1, item.values.length - 1);
      const y = margin.top + plotHeight * (1 - TRANSPORT_MATH.clamp(value, 0, 1));
      if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
    });
    context.stroke();
    context.fillStyle = item.color;
    for (const peak of peaks.filter((candidate) => item.peakCurves.includes(candidate.curve))) {
      const x = margin.left + plotWidth * peak.frame_index / Math.max(1, state.diagnostics.frame_count - 1);
      const value = item.values[peak.frame_index] || 0;
      const y = margin.top + plotHeight * (1 - TRANSPORT_MATH.clamp(value, 0, 1));
      context.beginPath();
      context.arc(x, y, 3.2 * ratio, 0, Math.PI * 2);
      context.fill();
    }
  }
}

async function diagnoseMotion({announce = true} = {}) {
  if (!canRepairMotion()) {
    if (announce) toast("Run Complete before diagnosing or repairing motion.", "error");
    return false;
  }
  if (!state.fixMode) return false;
  setBusy(el["diagnose-button"], true, "Computing AKE/RKE…");
  try {
    state.diagnostics = await api(`/api/sessions/${state.sessionId}/diagnose`, {method: "POST"});
    el["diagnostic-timeline"].classList.add("has-data");
    const count = state.diagnostics.peaks.length;
    updateDiagnosticControls();
    window.requestAnimationFrame(drawDiagnostics);
    if (announce) {
      addMessage(
        "system",
        `Diagnoser found ${count} discontinuity cue${count === 1 ? "" : "s"}.`,
        {key: "diagnostics-result"},
      );
      toast("Diagnostics ready.", "success");
    }
    return true;
  } catch (error) {
    toast(error.message, "error");
    return false;
  } finally {
    setBusy(el["diagnose-button"], false);
  }
}

async function downloadPkl(button, filename) {
  setBusy(button, true, "Writing PKL…");
  try {
    const response = await fetch(`/api/sessions/${state.sessionId}/exports/pkl`, {method: "POST"});
    if (!response.ok) {
      let detail = `${response.status} ${response.statusText}`;
      try { detail = (await response.json()).detail || detail; } catch { /* response was not JSON */ }
      throw new Error(detail);
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    const frames = response.headers.get("X-CustomDance-Frames");
    addMessage("system", `${filename} exported: ${frames} frames.`, {key: "export-pkl"});
    toast(`${filename} is ready.`, "success");
  } catch (error) {
    toast(error.message, "error");
  } finally {
    setBusy(button, false);
  }
}

function projectDuration() {
  if (Number.isFinite(state.analysis?.duration_sec)) return state.analysis.duration_sec;
  return Number.isFinite(el["audio-player"].duration) ? el["audio-player"].duration : 0;
}

function setPlaybackRate(value) {
  const policy = TRANSPORT_MATH.playbackPolicy(value);
  state.playbackRate = policy.rate;
  el["audio-player"].playbackRate = policy.rate;
  el["audio-player"].defaultPlaybackRate = policy.rate;
  el["audio-player"].muted = policy.muteAudio;
  el["playback-speed"].value = String(policy.rate);
  el["playback-audio-state"].textContent = policy.muteAudio ? "No music" : "Music";
  el["playback-audio-state"].dataset.muted = String(policy.muteAudio);
  el["playback-speed"].setAttribute(
    "aria-label",
    `Playback speed ${policy.label}${policy.muteAudio ? ", music muted" : ", music enabled"}`,
  );
}

function setAudioTime(time, {internal = false} = {}) {
  const duration = projectDuration();
  const target = TRANSPORT_MATH.clamp(Number(time) || 0, 0, Math.max(0, duration));
  if (internal) el["audio-player"].dataset.internalSeek = "true";
  try {
    el["audio-player"].currentTime = target;
  } catch {
    el["audio-player"].addEventListener("loadedmetadata", () => {
      el["audio-player"].currentTime = target;
      updatePlayhead();
    }, {once: true});
  }
  updatePlayhead();
}

function transportTick() {
  state.transportRaf = 0;
  const context = state.previewContext;
  if (context && !el["audio-player"].paused) {
    const tolerance = 1 / 60;
    if (
      el["audio-player"].currentTime >= context.endSec - tolerance
      || el["audio-player"].currentTime < context.startSec - tolerance
    ) {
      setAudioTime(context.startSec, {internal: true});
    }
  }
  updatePlayhead();
  if (!el["audio-player"].paused || state.seekPointerId !== null) {
    state.transportRaf = window.requestAnimationFrame(transportTick);
  }
}

function startTransportLoop() {
  if (!state.transportRaf) {
    state.transportRaf = window.requestAnimationFrame(transportTick);
  }
}

function seekTimelineTo(time, {focus = true} = {}) {
  if (!state.analysis) return;
  if (state.previewMotion) stopPreview({restoreTransport: false});
  setAudioTime(time, {internal: true});
  if (focus) el.playhead.focus({preventScroll: true});
  startTransportLoop();
}

function timelineTimeFromPointer(event) {
  const rect = el["timeline-content"].getBoundingClientRect();
  return TRANSPORT_MATH.timelineTimeFromClientX(
    event.clientX,
    rect.left,
    rect.width,
    projectDuration(),
  );
}

function timelineFrameFromPointer(event) {
  const rect = el["timeline-content"].getBoundingClientRect();
  return TRANSPORT_MATH.frameFromClientX(
    event.clientX,
    rect.left,
    rect.width,
    timelineFrameCount(),
  );
}

function setFixSelection(startFrame, endFrame) {
  const frameCount = timelineFrameCount();
  if (frameCount < 1) return;
  const start = TRANSPORT_MATH.clamp(Math.round(Number(startFrame) || 0), 0, frameCount - 1);
  const end = TRANSPORT_MATH.clamp(Math.round(Number(endFrame) || start + 1), start + 1, frameCount);
  state.fixSelection = {startFrame: start, endFrame: end};
  updateFixSelectionStatus();
  renderSelectionOverlay();
  updateControls();
}

function setRangeMode(active, {clearSelection = false} = {}) {
  const enabled = Boolean(active && state.fixMode && canRepairMotion());
  state.rangeMode = enabled;
  if (clearSelection) state.fixSelection = null;
  el["range-select-toggle"].setAttribute("aria-pressed", String(enabled));
  el["range-select-toggle"].setAttribute("aria-label", enabled ? "Deactivate range selection" : "Activate range selection");
  el["timeline-content"].classList.toggle("is-range-mode", enabled);
  updateFixSelectionStatus();
  renderSelectionOverlay();
  updateControls();
}

function syncFrameInputs() {
  const frameCount = timelineFrameCount();
  const maximumFrame = Math.max(0, frameCount - 1);
  el["current-frame"].max = String(maximumFrame);
  el["start-frame"].max = String(maximumFrame);
  el["end-frame"].max = String(Math.max(1, frameCount));
  if (state.fixSelection) {
    el["start-frame"].value = String(state.fixSelection.startFrame);
    el["end-frame"].value = String(state.fixSelection.endFrame);
  } else {
    const current = timelineFrameAtTime(el["audio-player"].currentTime || 0);
    el["start-frame"].value = String(current);
    el["end-frame"].value = String(Math.min(frameCount, current + 1));
  }
}

function handleTimelinePointerDown(event) {
  if (!state.analysis || event.button !== 0) return;
  if (state.rangeMode) {
    event.preventDefault();
    state.rangePointerId = event.pointerId;
    state.rangeAnchorFrame = timelineFrameFromPointer(event);
    event.currentTarget.setPointerCapture?.(event.pointerId);
    setFixSelection(state.rangeAnchorFrame, state.rangeAnchorFrame + 1);
    return;
  }
  if (event.target.closest?.(".slot-button")) return;
  event.preventDefault();
  state.seekPointerId = event.pointerId;
  state.seekSurface = event.currentTarget;
  event.currentTarget.setPointerCapture?.(event.pointerId);
  seekTimelineTo(timelineTimeFromPointer(event));
}

function handleTimelinePointerMove(event) {
  if (state.rangePointerId === event.pointerId) {
    event.preventDefault();
    const range = TRANSPORT_MATH.frameRange(
      state.rangeAnchorFrame,
      timelineFrameFromPointer(event),
      timelineFrameCount(),
    );
    setFixSelection(range.startFrame, range.endFrame);
    return;
  }
  if (state.seekPointerId !== event.pointerId) return;
  event.preventDefault();
  seekTimelineTo(timelineTimeFromPointer(event), {focus: false});
}

function handleTimelinePointerEnd(event) {
  if (state.rangePointerId === event.pointerId) {
    event.preventDefault();
    event.currentTarget.releasePointerCapture?.(event.pointerId);
    state.rangePointerId = null;
    state.rangeAnchorFrame = null;
    updateFixSelectionStatus();
    updateControls();
    return;
  }
  if (state.seekPointerId !== event.pointerId) return;
  state.seekSurface?.releasePointerCapture?.(event.pointerId);
  state.seekPointerId = null;
  state.seekSurface = null;
  updatePlayhead();
}

function commitFrameRangeInputs() {
  if (!state.rangeMode) return;
  setFixSelection(Number(el["start-frame"].value), Number(el["end-frame"].value));
}

function commitFrameRangeInputsWhileEditing() {
  if (!el["start-frame"].value || !el["end-frame"].value) return;
  commitFrameRangeInputs();
}

function commitCurrentFrame() {
  if (!state.timeline) return;
  const frame = TRANSPORT_MATH.clamp(
    Math.round(Number(el["current-frame"].value) || 0),
    0,
    timelineFrameCount() - 1,
  );
  seekTimelineTo(frame / timelineFps(), {focus: false});
}

function setTimelineZoom(index) {
  const normalized = TRANSPORT_MATH.clamp(Math.round(index), 0, TIMELINE_ZOOM_LEVELS.length - 1);
  const currentTime = el["audio-player"].currentTime || 0;
  state.zoomSeconds = TIMELINE_ZOOM_LEVELS[normalized];
  updateTimelineScale();
  if (state.analysis) renderRuler(projectDuration());
  window.requestAnimationFrame(() => {
    const duration = projectDuration();
    if (duration > 0) {
      const x = currentTime / duration * el["timeline-content"].clientWidth;
      el["timeline-shell"].scrollLeft = Math.max(0, x - el["timeline-shell"].clientWidth / 2);
    }
    drawDiagnostics();
  });
  updateControls();
}

function resetPlayback() {
  el["audio-player"].pause();
  if (state.previewMotion) stopPreview({restoreTransport: false});
  setAudioTime(0, {internal: true});
}

async function playMotion() {
  if (!state.timeline) return;
  if ((el["audio-player"].currentTime || 0) >= projectDuration() - 1 / timelineFps()) {
    setAudioTime(0, {internal: true});
  }
  try {
    await el["audio-player"].play();
    startTransportLoop();
  } catch (error) {
    toast(`Playback could not start: ${error.message}`, "error");
  }
}

function handlePlayheadKeydown(event) {
  if (!state.analysis) return;
  const current = el["audio-player"].currentTime || 0;
  const step = event.shiftKey ? 1 : 1 / timelineFps();
  let target = null;
  if (event.key === "ArrowLeft" || event.key === "ArrowDown") target = current - step;
  if (event.key === "ArrowRight" || event.key === "ArrowUp") target = current + step;
  if (event.key === "Home") target = 0;
  if (event.key === "End") target = projectDuration();
  if (target === null) return;
  event.preventDefault();
  seekTimelineTo(target);
}

function updatePlayhead() {
  const duration = projectDuration();
  if (!state.analysis || duration <= 0) {
    el["current-frame"].value = "0";
    syncFrameInputs();
    return;
  }
  const time = TRANSPORT_MATH.clamp(el["audio-player"].currentTime || 0, 0, duration);
  const frameCount = timelineFrameCount();
  const frame = timelineFrameAtTime(time);
  el.playhead.style.left = `${time / duration * 100}%`;
  el.playhead.setAttribute("aria-valuemax", String(frameCount - 1));
  el.playhead.setAttribute("aria-valuenow", String(frame));
  el.playhead.setAttribute("aria-valuetext", `Frame ${frame} of ${frameCount - 1}`);
  if (document.activeElement !== el["current-frame"]) {
    el["current-frame"].value = String(frame);
  }
  syncFrameInputs();
}

function sizeCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(rect.width * ratio));
  const height = Math.max(1, Math.round(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return {width, height, ratio};
}

function quantile(values, fraction) {
  if (!values.length) return 0;
  const ordered = [...values].sort((left, right) => left - right);
  const position = (ordered.length - 1) * fraction;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const weight = position - lower;
  return ordered[lower] * (1 - weight) + ordered[upper] * weight;
}

class ThreeStage {
  constructor(container, interactionSurface) {
    this.container = container;
    this.interactionSurface = interactionSurface;
    this.motion = null;
    this.mode = "smpl";
    this.groundHeight = 0;
    this.floorBaseM = 0;
    this.centerX = 0;
    this.centerZ = 0;
    this.bodyHeight = 1.7;
    this.motionHeight = 1.7;
    this.sceneSpan = 3;
    this.characterSources = new Map();
    this.smplAvailable = false;
    this.smplLoading = null;
    this.reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    this.basis = new THREE.Quaternion().setFromAxisAngle(
      new THREE.Vector3(1, 0, 0), -Math.PI / 2,
    );
    this.basisInverse = this.basis.clone().invert();
    const canonicalRestPose = new Array(SMPL_NAMES.length * 3).fill(0);
    // Canonical clips pre-multiply the source root by +90 degrees around X so
    // Y-up SMPL offsets become Z-up. Remove that fixed basis pose before
    // retargeting rotations onto an already-upright Three.js character.
    canonicalRestPose[0] = Math.PI / 2;
    this.smplRestWorld = this._smplWorldRotations(canonicalRestPose);
    this.up = new THREE.Vector3(0, 1, 0);
    this._initialize();
  }

  _initialize() {
    try {
      this.scene = new THREE.Scene();
      this.scene.fog = new THREE.Fog(0x030712, 9, 28);
      this.camera = new THREE.PerspectiveCamera(42, 1, 0.02, 120);
      this.renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true,
        powerPreference: "high-performance",
      });
      this.renderer.setClearColor(0x030712, 0);
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      if ("outputColorSpace" in this.renderer && THREE.SRGBColorSpace) {
        this.renderer.outputColorSpace = THREE.SRGBColorSpace;
      } else {
        this.renderer.outputEncoding = THREE.sRGBEncoding;
      }
      this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
      this.renderer.toneMappingExposure = 1.05;
      this.renderer.shadowMap.enabled = true;
      this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      this.container.append(this.renderer.domElement);

      this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = !this.reducedMotion;
      this.controls.dampingFactor = 0.075;
      this.controls.screenSpacePanning = true;
      this.controls.minDistance = 0.45;
      this.controls.maxDistance = 60;
      this.controls.maxPolarAngle = Math.PI * 0.495;
      this.controls.listenToKeyEvents(this.interactionSurface);
      this.renderer.domElement.addEventListener("wheel", (event) => {
        event.stopPropagation();
      }, {passive: true});

      this._createEnvironment();
      this._createCharacterSources();
      this._resize();
      this.resetCamera();
      this.resizeObserver = new ResizeObserver(() => this._resize());
      this.resizeObserver.observe(this.interactionSurface);
      this.interactionSurface.addEventListener("keydown", (event) => {
        if (event.key.toLowerCase() === "r") {
          event.preventDefault();
          this.resetCamera();
        }
      });
      this.renderer.setAnimationLoop((timestamp) => this._render(timestamp));
      this._checkSmplAvailability();
    } catch (error) {
      el["viewer-status"].dataset.kind = "error";
      el["viewer-status"].textContent = `WebGL unavailable: ${error.message}`;
      el["stage-empty"].textContent = "This browser could not create the Three.js WebGL renderer.";
    }
  }

  _createEnvironment() {
    const hemisphere = new THREE.HemisphereLight(0xb9d9ee, 0x080d17, 0.9);
    this.scene.add(hemisphere);
    const key = new THREE.DirectionalLight(0xffffff, 1.65);
    key.position.set(4.5, 8, 5.5);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.camera.left = -8;
    key.shadow.camera.right = 8;
    key.shadow.camera.top = 8;
    key.shadow.camera.bottom = -8;
    key.shadow.camera.near = 0.1;
    key.shadow.camera.far = 30;
    key.shadow.bias = -0.0004;
    this.scene.add(key, key.target);

    this.floor = new THREE.Mesh(
      new THREE.PlaneGeometry(20, 20),
      new THREE.MeshStandardMaterial({color: 0x0a1322, roughness: 0.96, metalness: 0.01}),
    );
    this.floor.rotation.x = -Math.PI / 2;
    this.floor.receiveShadow = true;
    this.scene.add(this.floor);
    this.grid = new THREE.GridHelper(20, 40, 0x36526f, 0x22324a);
    this.grid.material.transparent = true;
    this.grid.material.opacity = 0.48;
    this.scene.add(this.grid);
    this.worldForward = new THREE.ArrowHelper(
      new THREE.Vector3(0, 0, -1),
      new THREE.Vector3(0, 0, 0),
      0.6,
      0x22d3ee,
      0.12,
      0.07,
    );
    this.scene.add(this.worldForward);
    this._applyFloorHeight();
  }

  _applyFloorHeight() {
    this.floor.position.y = this.floorBaseM;
    this.grid.position.y = this.floorBaseM + 0.002;
    this.worldForward.position.y = this.floorBaseM + 0.008;
  }

  _createCharacterSources() {
    this.characterSources = new Map();
  }

  _frontCorrection() {
    const [x, y, z, w] = VIEWER_MATH.neutralSmplBindCorrectionQuaternion();
    return new THREE.Quaternion(x, y, z, w);
  }

  _orientModel(model, authoredModelQuaternion) {
    model.quaternion.copy(
      this._frontCorrection().multiply(authoredModelQuaternion),
    );
  }

  _setViewerStatus(message, kind = "ready") {
    el["viewer-status"].dataset.kind = kind;
    el["viewer-status"].textContent = message;
  }

  async _checkSmplAvailability() {
    try {
      const status = await api("/api/models/smpl/status");
      this.smplAvailable = Boolean(status.available);
      el["view-character"].disabled = !this.smplAvailable;
      el["view-character"].title = status.detail;
      if (this.smplAvailable) {
        this._setViewerStatus("3D ready · Neutral SMPL installed");
        await this.activateSmpl();
      } else {
        this._setViewerStatus("Neutral SMPL required · download and configure the model");
      }
    } catch (error) {
      this.smplAvailable = false;
      el["view-character"].disabled = true;
      this._setViewerStatus("SMPL status unavailable · check the configured resource", "error");
    }
  }

  _disposeObject(root) {
    root.traverse((object) => {
      object.geometry?.dispose?.();
      const materials = Array.isArray(object.material)
        ? object.material
        : object.material
          ? [object.material]
          : [];
      materials.forEach((material) => material.dispose?.());
    });
  }

  _disposeSource(id) {
    const source = this.characterSources.get(id);
    if (!source) return;
    this.scene.remove(source.root);
    this._disposeObject(source.root);
    this.characterSources.delete(id);
  }

  _registerRigSource({
    id,
    label,
    root,
    model,
    rigBones,
    mappedTargets,
    bodyBone,
  }) {
    this._disposeSource(id);
    root.name = label;
    root.visible = false;
    this.scene.add(root);
    root.updateMatrixWorld(true);
    let hasSkinnedMesh = false;
    model.traverse((object) => {
      if (object.isSkinnedMesh) hasSkinnedMesh = true;
      if (object.isMesh || object.isSkinnedMesh) {
        object.castShadow = true;
        object.receiveShadow = true;
        object.frustumCulled = false;
      }
    });
    const rejectSource = (message) => {
      this.scene.remove(root);
      this._disposeObject(root);
      throw new Error(message);
    };
    if (!hasSkinnedMesh) {
      rejectSource("Character source must contain at least one SkinnedMesh");
    }
    const restBoneState = new Map();
    const restWorldQuaternions = new Map();
    for (const bone of rigBones.values()) {
      restBoneState.set(bone, {
        position: bone.position.clone(),
        quaternion: bone.quaternion.clone(),
      });
      restWorldQuaternions.set(
        bone,
        bone.getWorldQuaternion(new THREE.Quaternion()),
      );
    }
    const orderedTargets = mappedTargets
      .filter(({bone}) => Boolean(bone))
      .sort((left, right) => this._objectDepth(left.bone) - this._objectDepth(right.bone));
    if (!bodyBone || orderedTargets.length < 12) {
      rejectSource("Rig needs a pelvis/root and at least 12 mapped humanoid bones");
    }
    const bodyWorld = bodyBone.getWorldPosition(new THREE.Vector3());
    const bodyRestLocal = root.worldToLocal(bodyWorld.clone());
    const bounds = new THREE.Box3().setFromObject(model);
    const modelRestHeight = bounds.max.y - bounds.min.y;
    if (!Number.isFinite(modelRestHeight) || modelRestHeight <= 1e-4) {
      rejectSource("Character source has no finite renderable body bounds");
    }
    const mappedBoneByJoint = new Map(
      orderedTargets.map(({bone, joint}) => [joint, bone]),
    );
    const restBoneLengths = new Array(SMPL_NAMES.length).fill(null);
    for (const {bone, joint} of orderedTargets) {
      const parentJoint = SMPL_PARENTS[joint];
      const parentBone = mappedBoneByJoint.get(parentJoint);
      if (parentJoint < 0 || !parentBone) continue;
      restBoneLengths[joint] = bone.getWorldPosition(new THREE.Vector3()).distanceTo(
        parentBone.getWorldPosition(new THREE.Vector3()),
      );
    }
    const footBoneHeights = FOOT_JOINTS
      .map((joint) => mappedBoneByJoint.get(joint))
      .filter(Boolean)
      .map((bone) => bone.getWorldPosition(new THREE.Vector3()).y);
    const soleBelowFootJointM = footBoneHeights.length
      ? Math.max(0, Math.min(...footBoneHeights) - bounds.min.y)
      : 0;
    const source = {
      id,
      label,
      root,
      model,
      rigBones,
      mappedTargets: orderedTargets,
      restBoneState,
      restWorldQuaternions,
      bodyBone,
      bodyRestLocal,
      modelRestHeight,
      restBoneLengths,
      soleBelowFootJointM,
      scale: 1,
    };
    this.characterSources.set(id, source);
    this._updateCharacterScale();
    this._syncCharacterVisibility();
    return source;
  }

  _buildSmplSource(payload) {
    if (
      payload.schema_version !== "1.0"
      || payload.coordinate_system !== "right-handed-y-up"
      || payload.parents?.length !== 24
    ) {
      throw new Error("Unsupported SMPL template response");
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(payload.vertices.flat(), 3),
    );
    geometry.setAttribute(
      "skinIndex",
      new THREE.Uint16BufferAttribute(payload.skin_indices.flat(), 4),
    );
    geometry.setAttribute(
      "skinWeight",
      new THREE.Float32BufferAttribute(payload.skin_weights.flat(), 4),
    );
    geometry.setIndex(
      new THREE.Uint16BufferAttribute(payload.faces.flat(), 1),
    );
    geometry.computeVertexNormals();
    geometry.computeBoundingSphere();

    const material = new THREE.MeshStandardMaterial({
      color: 0xc98f72,
      roughness: 0.76,
      metalness: 0.0,
      skinning: true,
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.SkinnedMesh(geometry, material);
    mesh.name = "Neutral SMPL Skin";
    const bones = payload.joint_names.map((name) => {
      const bone = new THREE.Bone();
      bone.name = name;
      return bone;
    });
    payload.parents.forEach((parent, index) => {
      const joint = payload.joints[index];
      if (parent < 0) {
        bones[index].position.set(joint[0], joint[1], joint[2]);
        mesh.add(bones[index]);
      } else {
        const parentJoint = payload.joints[parent];
        bones[index].position.set(
          joint[0] - parentJoint[0],
          joint[1] - parentJoint[1],
          joint[2] - parentJoint[2],
        );
        bones[parent].add(bones[index]);
      }
    });
    const skeleton = new THREE.Skeleton(bones);
    const model = new THREE.Group();
    const root = new THREE.Group();
    model.name = "Neutral SMPL";
    const authoredModelQuaternion = model.quaternion.clone();
    // SMPL's native front is +Z; editor forward is Three.js -Z.
    this._orientModel(model, authoredModelQuaternion);
    model.add(mesh);
    root.add(model);
    root.updateMatrixWorld(true);
    mesh.bind(skeleton);
    mesh.normalizeSkinWeights();

    const rigBones = new Map(bones.map((bone) => [bone.name, bone]));
    return this._registerRigSource({
      id: "smpl",
      label: "Neutral SMPL",
      root,
      model,
      rigBones,
      mappedTargets: bones.map((bone, joint) => ({bone, joint})),
      bodyBone: bones[0],
    });
  }

  async activateSmpl() {
    if (this.characterSources.has("smpl")) {
      this.setMode("smpl");
      return;
    }
    if (!this.smplAvailable) {
      toast(
        "Neutral SMPL is not installed. Download it under the official license and configure SMPL_MODEL_ROOT.",
        "error",
      );
      return;
    }
    if (this.smplLoading) {
      await this.smplLoading;
      this.setMode("smpl");
      return;
    }
    el["view-character"].disabled = true;
    el["view-character"].setAttribute("aria-busy", "true");
    this._setViewerStatus("Loading the local Neutral SMPL template…");
    this.smplLoading = (async () => {
      const payload = await api("/api/models/smpl/template");
      this._buildSmplSource(payload);
    })();
    try {
      await this.smplLoading;
      this.setMode("smpl");
      this._setViewerStatus("Neutral SMPL ready");
    } catch (error) {
      this._setViewerStatus("Neutral SMPL failed to load: " + error.message, "error");
      toast(error.message, "error");
    } finally {
      this.smplLoading = null;
      el["view-character"].disabled = !this.smplAvailable;
      el["view-character"].removeAttribute("aria-busy");
    }
  }

  _objectDepth(object) {
    let depth = 0;
    for (let current = object.parent; current; current = current.parent) depth += 1;
    return depth;
  }

  _resize() {
    if (!this.renderer) return;
    const rect = this.interactionSurface.getBoundingClientRect();
    const width = Math.max(1, Math.round(rect.width));
    const height = Math.max(1, Math.round(rect.height));
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  _point(point) {
    return new THREE.Vector3(...VIEWER_MATH.canonicalPointToEditor(
      point,
      this.centerX,
      this.groundHeight,
      this.centerZ,
    ));
  }

  _rotationFromAxisAngle(vector) {
    const axis = new THREE.Vector3(vector[0], vector[1], vector[2]);
    const angle = axis.length();
    const canonical = angle > 1e-10
      ? new THREE.Quaternion().setFromAxisAngle(axis.multiplyScalar(1 / angle), angle)
      : new THREE.Quaternion();
    return this.basis.clone().multiply(canonical).multiply(this.basisInverse);
  }

  setMotion(motion, {fitCamera = false} = {}) {
    if (!motion || motion.coordinate_system !== "right-handed-z-up") {
      throw new Error("Three.js viewer requires Canonical right-handed Z-up motion");
    }
    this.motion = motion;
    const footHeights = [];
    const frameHeights = [];
    let minimumX = Infinity;
    let maximumX = -Infinity;
    let minimumZ = Infinity;
    let maximumZ = -Infinity;
    let maximumHeight = -Infinity;
    for (const frame of motion.joint_positions) {
      const feet = FOOT_JOINTS.map((index) => frame[index][2]);
      footHeights.push(...feet);
      const vertical = frame.map((point) => point[2]);
      maximumHeight = Math.max(maximumHeight, ...vertical);
      frameHeights.push(Math.max(...vertical) - Math.min(...feet));
      for (const point of frame) {
        minimumX = Math.min(minimumX, point[0]);
        maximumX = Math.max(maximumX, point[0]);
        const threeZ = -point[1];
        minimumZ = Math.min(minimumZ, threeZ);
        maximumZ = Math.max(maximumZ, threeZ);
      }
    }
    // Use the lowest SMPL foot joint for both short previews and completed
    // motions so a four-second slice cannot visibly penetrate the floor.
    // The skinned mesh's sole thickness is compensated independently below.
    this.groundHeight = Math.min(...footHeights);
    this.centerX = (minimumX + maximumX) / 2;
    this.centerZ = (minimumZ + maximumZ) / 2;
    this.bodyHeight = Math.max(0.5, quantile(frameHeights, 0.5));
    this.motionHeight = Math.max(0.5, maximumHeight - this.groundHeight);
    this.sceneSpan = Math.max(
      this.motionHeight,
      maximumX - minimumX,
      maximumZ - minimumZ,
      1.2,
    );
    const floorScale = Math.max(1, this.sceneSpan * 2.4 / 20);
    this.floor.scale.setScalar(floorScale);
    this.grid.scale.setScalar(floorScale);
    el["stage-empty"].hidden = true;
    this._updateCharacterScale();
    this._syncCharacterVisibility();
    if (fitCamera) this.resetCamera();
  }

  clearMotion() {
    this.motion = null;
    for (const source of this.characterSources.values()) {
      source.root.visible = false;
    }
    el["stage-empty"].hidden = false;
    el["stage-time"].textContent = "00:00.000";
  }

  _updateCharacterScale() {
    let floorBaseM = 0;
    for (const source of this.characterSources.values()) {
      const rigScale = this.motion
        ? VIEWER_MATH.rigScaleFromBoneLengths(
          this.motion.joint_positions[0],
          SMPL_PARENTS,
          source.restBoneLengths,
        )
        : null;
      source.scale = rigScale ?? this.bodyHeight / source.modelRestHeight;
      source.root.scale.setScalar(source.scale);
      if (source.id === "smpl" && source.soleBelowFootJointM > 0) {
        floorBaseM = -source.soleBelowFootJointM * source.scale;
      }
    }
    this.floorBaseM = floorBaseM;
    if (this.floor) this._applyFloorHeight();
  }

  _syncCharacterVisibility() {
    for (const [id, source] of this.characterSources) {
      source.root.visible = Boolean(this.motion) && this.mode === id;
    }
  }

  setMode(mode) {
    if (mode !== "smpl" || !this.characterSources.has(mode)) return;
    this.mode = mode;
    el["view-character"].setAttribute("aria-pressed", "true");
    this._syncCharacterVisibility();
    this._setViewerStatus("Neutral SMPL active");
  }

  resetCamera() {
    if (!this.camera || !this.controls) return;
    const target = new THREE.Vector3(0, this.motionHeight * 0.48, 0);
    const verticalFov = THREE.MathUtils.degToRad(this.camera.fov);
    const distance = Math.max(2.2, this.sceneSpan / (2 * Math.tan(verticalFov / 2)) * 1.6);
    this.controls.target.copy(target);
    this.camera.position.set(distance * 0.68, target.y + distance * 0.34, -distance * 0.88);
    this.controls.minDistance = distance * 0.7;
    this.controls.maxDistance = distance * 1.8;
    this.camera.near = Math.max(0.01, distance / 200);
    this.camera.far = Math.max(40, distance * 12);
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  _smplWorldRotations(poses) {
    const local = SMPL_NAMES.map((_, index) => (
      this._rotationFromAxisAngle(poses.slice(index * 3, index * 3 + 3))
    ));
    const world = [];
    local.forEach((rotation, index) => {
      const parent = SMPL_PARENTS[index];
      world[index] = parent < 0 ? rotation : world[parent].clone().multiply(rotation);
    });
    return world;
  }

  _smplWorldDeltas(poses) {
    return this._smplWorldRotations(poses).map((rotation, index) => {
      const nativeDelta = rotation.clone().multiply(
        this.smplRestWorld[index].clone().invert(),
      );
      const corrected = VIEWER_MATH.editorConjugateQuaternion([
        nativeDelta.x,
        nativeDelta.y,
        nativeDelta.z,
        nativeDelta.w,
      ]);
      return new THREE.Quaternion(...corrected);
    });
  }

  _updateCharacter(source, joints, poses) {
    if (!source) return;
    source.root.scale.setScalar(source.scale);
    const rootTarget = this._point(joints[0]);
    source.root.position.copy(rootTarget).sub(
      source.bodyRestLocal.clone().multiplyScalar(source.scale),
    );
    for (const [bone, rest] of source.restBoneState) {
      bone.position.copy(rest.position);
      bone.quaternion.copy(rest.quaternion);
    }
    source.root.updateMatrixWorld(true);
    const smplWorld = this._smplWorldDeltas(poses);
    for (const {bone, joint} of source.mappedTargets) {
      bone.parent.updateWorldMatrix(true, false);
      const parentWorld = bone.parent.getWorldQuaternion(new THREE.Quaternion());
      const targetWorld = smplWorld[joint].clone().multiply(
        source.restWorldQuaternions.get(bone),
      );
      bone.quaternion.copy(parentWorld.invert().multiply(targetWorld));
      bone.updateMatrix();
      bone.updateWorldMatrix(false, true);
    }
  }

  _render() {
    if (!this.renderer) return;
    const motion = state.previewMotion || state.currentMotion;
    if (motion && this.motion === motion) {
      const audioTime = el["audio-player"].currentTime || 0;
      const time = state.previewMotion && state.previewContext
        ? TRANSPORT_MATH.previewTime(
          audioTime,
          state.previewContext.startSec,
          motion.duration_sec,
        )
        : audioTime;
      const frameIndex = TRANSPORT_MATH.frameIndex(time, motion.fps, motion.frames);
      const joints = motion.joint_positions[frameIndex];
      const poses = motion.smpl_poses[frameIndex];
      this._updateCharacter(this.characterSources.get(this.mode), joints, poses);
      el["stage-time"].textContent = new Date(time * 1000).toISOString().slice(14, 23);
    }
    this.controls?.update();
    this.renderer.render(this.scene, this.camera);
  }
}

function setViewerMaximized(active) {
  const panel = el["viewport-panel"];
  const maximized = Boolean(active);
  panel.classList.toggle("is-maximized", maximized);
  el["viewer-maximize"].setAttribute("aria-pressed", String(maximized));
  el["viewer-maximize"].setAttribute("aria-label", maximized ? "Restore 3D Viewer" : "Maximize 3D Viewer");
}

function toggleViewerMaximize() {
  setViewerMaximized(!el["viewport-panel"].classList.contains("is-maximized"));
}

state.stage = new ThreeStage(el["stage-canvas"], el["stage-shell"]);

el["music-file"].addEventListener("change", (event) => {
  const [file] = event.target.files;
  if (file) importMusic(file);
});
el["analyze-button"].addEventListener("click", analyze);
el["motion-query"].addEventListener("input", () => {
  saveFocusedQuery();
  updateQueryOrigin();
  updateControls();
});
el["refresh-button"].addEventListener("click", retrieve);
el["stop-preview"].addEventListener("click", () => stopPreview());
el["planning-chat-launcher"].addEventListener("click", () => setChatOpen(true));
el["planning-chat-close"].addEventListener("click", () => setChatOpen(false));
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && state.chatOpen) setChatOpen(false);
  if (event.key === "Escape") hideSlotDescription();
  if (event.key === "Escape" && el["viewport-panel"].classList.contains("is-maximized")) {
    setViewerMaximized(false);
  }
});
document.querySelectorAll("[data-diagnostic-signal]").forEach((button) => {
  button.addEventListener("click", () => {
    const key = button.dataset.diagnosticSignal;
    if (state.diagnosticSignals.has(key)) state.diagnosticSignals.delete(key);
    else state.diagnosticSignals.add(key);
    updateDiagnosticControls();
    window.requestAnimationFrame(drawDiagnostics);
  });
});
document.querySelectorAll("[data-joint-group]").forEach((button) => {
  button.addEventListener("click", () => {
    const key = button.dataset.jointGroup;
    if (state.selectedJointGroups.has(key)) state.selectedJointGroups.delete(key);
    else state.selectedJointGroups.add(key);
    updateDiagnosticControls();
    if (state.diagnostics) window.requestAnimationFrame(drawDiagnostics);
  });
});
for (const radio of document.querySelectorAll('input[name="top-k"]')) {
  radio.addEventListener("change", () => {
    if (!radio.checked) return;
    state.topK = Number(radio.value);
    renderMotionList();
  });
}
el["timeline-content"].addEventListener("pointerdown", handleTimelinePointerDown);
el["timeline-content"].addEventListener("pointermove", handleTimelinePointerMove);
el["timeline-content"].addEventListener("pointerup", handleTimelinePointerEnd);
el["timeline-content"].addEventListener("pointercancel", handleTimelinePointerEnd);
el.playhead.addEventListener("keydown", handlePlayheadKeydown);
el["audio-player"].addEventListener("play", startTransportLoop);
el["playback-speed"].addEventListener("change", () => setPlaybackRate(el["playback-speed"].value));
for (const eventName of ["pause", "timeupdate", "seeking", "seeked", "loadedmetadata", "durationchange", "ended"]) {
  el["audio-player"].addEventListener(eventName, updatePlayhead);
}
el["audio-player"].addEventListener("loadedmetadata", () => {
  setPlaybackRate(state.playbackRate);
  renderAudioSummary();
  updateTimelineScale();
});
el["fix-toggle"].addEventListener("click", () => {
  setFixMode(!state.fixMode);
});
el["candidates-tab"].addEventListener("click", () => setRightPanel("candidates"));
el["repair-tab"].addEventListener("click", () => setRightPanel("repair"));
el["complete-button"].addEventListener("click", completeMotion);
el["smooth-button"].addEventListener("click", smoothMotion);
el["remake-button"].addEventListener("click", remakeMotion);
el["diagnose-button"].addEventListener("click", () => diagnoseMotion());
el["view-character"].addEventListener("click", () => state.stage?.activateSmpl());
el["reset-camera"].addEventListener("click", () => state.stage?.resetCamera());
el["viewer-maximize"].addEventListener("click", toggleViewerMaximize);
el["timeline-reset"].addEventListener("click", resetPlayback);
el["play-button"].addEventListener("click", playMotion);
el["pause-button"].addEventListener("click", () => el["audio-player"].pause());
el["range-select-toggle"].addEventListener("click", () => setRangeMode(!state.rangeMode));
el["zoom-in"].addEventListener("click", () => setTimelineZoom(TIMELINE_ZOOM_LEVELS.indexOf(state.zoomSeconds) - 1));
el["zoom-out"].addEventListener("click", () => setTimelineZoom(TIMELINE_ZOOM_LEVELS.indexOf(state.zoomSeconds) + 1));
el["timeline-zoom-slider"].addEventListener("input", () => setTimelineZoom(Number(el["timeline-zoom-slider"].value)));
el["current-frame"].addEventListener("change", commitCurrentFrame);
el["start-frame"].addEventListener("change", commitFrameRangeInputs);
el["end-frame"].addEventListener("change", commitFrameRangeInputs);
el["start-frame"].addEventListener("input", commitFrameRangeInputsWhileEditing);
el["end-frame"].addEventListener("input", commitFrameRangeInputsWhileEditing);
el["export-pkl"].addEventListener("click", () => downloadPkl(el["export-pkl"], "customdance-motion.pkl"));

fetch("/api/health").then((response) => {
  if (!response.ok) throw new Error("health check failed");
}).catch(() => toast("Local backend is unavailable.", "error"));

window.addEventListener("resize", () => window.requestAnimationFrame(() => {
  updateTimelineScale();
  drawDiagnostics();
  if (state.analysis) renderRuler(projectDuration());
}));

setRightPanel("candidates", {syncFixMode: false});
setTimelineZoom(1);
setPlaybackRate(1);
updateChatBadge();
updateDiagnosticControls();
updateFixSelectionStatus();
renderAudioSummary();
updateControls();
