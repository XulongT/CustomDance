"use strict";

(function exposeCandidateState(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.CustomDanceCandidateState = api;
})(typeof globalThis !== "undefined" ? globalThis : this, () => {
  function validateStore(store) {
    if (!(store instanceof Map)) throw new Error("favorite store must be a Map");
  }

  function validateId(value, label) {
    if (typeof value !== "string" || !value.trim()) {
      throw new Error(`${label} must be a non-empty string`);
    }
    return value;
  }

  function favoritesForSlot(store, slotId, {create = true} = {}) {
    validateStore(store);
    const normalizedSlotId = validateId(slotId, "slot id");
    let favorites = store.get(normalizedSlotId);
    if (!favorites && create) {
      favorites = new Set();
      store.set(normalizedSlotId, favorites);
    }
    return favorites || new Set();
  }

  function toggleFavorite(store, slotId, clipId) {
    const favorites = favoritesForSlot(store, slotId);
    const normalizedClipId = validateId(clipId, "clip id");
    if (favorites.has(normalizedClipId)) {
      favorites.delete(normalizedClipId);
      return false;
    }
    favorites.add(normalizedClipId);
    return true;
  }

  return Object.freeze({favoritesForSlot, toggleFavorite});
});
