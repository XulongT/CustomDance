"use strict";

(function exposeViewerMath(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.CustomDanceViewerMath = api;
})(typeof globalThis !== "undefined" ? globalThis : this, () => {
  const EDITOR_YAW_QUATERNION = Object.freeze([0, 1, 0, 0]);
  const EDITOR_FORWARD = Object.freeze([0, 0, -1]);

  function multiplyQuaternion(left, right) {
    const [lx, ly, lz, lw] = left;
    const [rx, ry, rz, rw] = right;
    return [
      lw * rx + lx * rw + ly * rz - lz * ry,
      lw * ry - lx * rz + ly * rw + lz * rx,
      lw * rz + lx * ry - ly * rx + lz * rw,
      lw * rw - lx * rx - ly * ry - lz * rz,
    ];
  }

  function inverseUnitQuaternion(value) {
    return [-value[0], -value[1], -value[2], value[3]];
  }

  function editorConjugateQuaternion(value) {
    return multiplyQuaternion(
      multiplyQuaternion(EDITOR_YAW_QUATERNION, value),
      inverseUnitQuaternion(EDITOR_YAW_QUATERNION),
    );
  }

  function canonicalPointToEditor(point, centerX, groundHeight, legacyCenterZ) {
    return [
      centerX - point[0],
      point[2] - groundHeight,
      point[1] + legacyCenterZ,
    ];
  }

  function neutralSmplBindCorrectionQuaternion() {
    return [...EDITOR_YAW_QUATERNION];
  }

  function rigScaleFromBoneLengths(joints, parents, sourceBoneLengths) {
    if (!Array.isArray(joints) || joints.length !== parents.length) return null;
    if (!sourceBoneLengths || sourceBoneLengths.length !== parents.length) return null;
    const ratios = [];
    parents.forEach((parent, joint) => {
      if (parent < 0) return;
      const sourceLength = Number(sourceBoneLengths[joint]);
      const child = joints[joint];
      const parentPoint = joints[parent];
      if (
        !Number.isFinite(sourceLength)
        || sourceLength < 0.04
        || !Array.isArray(child)
        || !Array.isArray(parentPoint)
      ) return;
      const motionLength = Math.hypot(
        child[0] - parentPoint[0],
        child[1] - parentPoint[1],
        child[2] - parentPoint[2],
      );
      const ratio = motionLength / sourceLength;
      if (Number.isFinite(ratio) && motionLength >= 0.04 && ratio >= 0.5 && ratio <= 2) {
        ratios.push(ratio);
      }
    });
    if (ratios.length < 8) return null;
    ratios.sort((left, right) => left - right);
    const middle = Math.floor(ratios.length / 2);
    return ratios.length % 2
      ? ratios[middle]
      : (ratios[middle - 1] + ratios[middle]) / 2;
  }

  return Object.freeze({
    EDITOR_FORWARD,
    EDITOR_YAW_QUATERNION,
    canonicalPointToEditor,
    editorConjugateQuaternion,
    inverseUnitQuaternion,
    multiplyQuaternion,
    neutralSmplBindCorrectionQuaternion,
    rigScaleFromBoneLengths,
  });
});
