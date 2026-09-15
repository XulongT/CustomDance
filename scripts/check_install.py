#!/usr/bin/env python3
"""Report independent offline setup checks; never send API requests or require a key."""

import argparse
import contextlib
import importlib
import importlib.metadata
import io
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/default.yaml")
    parser.add_argument(
        "--models", action="store_true", help="load the real retrieval and completion models"
    )
    args = parser.parse_args()
    report = {
        "scope": "offline",
        "python": sys.version.split()[0],
        "api_key_configured": None,
        "checks": {},
    }
    checks = report["checks"]

    def check(name, action, hint, *, requires=(), enabled=True):
        if not enabled:
            checks[name] = {"status": "SKIPPED", "reason": hint}
            return None
        failed = [item for item in requires if checks[item]["status"] != "PASS"]
        if failed:
            checks[name] = {"status": "SKIPPED", "reason": "Requires passing: " + ", ".join(failed)}
            return None
        try:
            # Keep stdout valid JSON even if an imported dependency prints diagnostics.
            with contextlib.redirect_stdout(io.StringIO()):
                value = action()
        except Exception as error:
            # Validation/SDK exception text can contain configuration or key values.
            checks[name] = {"status": "FAIL", "error_type": type(error).__name__, "reason": hint}
            return None
        checks[name] = {"status": "PASS"}
        return value

    def python_version():
        if sys.version_info[:2] != (3, 10):
            raise RuntimeError("unsupported Python")

    def system_tools():
        report["missing_commands"] = [name for name in ("ffmpeg", "ffprobe") if not shutil.which(name)]
        if report["missing_commands"]:
            raise FileNotFoundError("missing audio tools")

    def packages():
        report["packages"], report["missing_packages"] = {}, []
        for name in ("mamba-ssm", "causal-conv1d", "transformers", "numpy", "librosa",
                     "fastapi", "openai", "smplx", "chumpy"):
            try:
                report["packages"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                report["missing_packages"].append(name)
        if report["missing_packages"]:
            raise ImportError("missing runtime packages")

    def configuration():
        settings = importlib.import_module("app.config").load_settings(args.config)
        report["api_key_configured"] = bool(settings.openai_api_key)
        return settings, settings.asset_paths()

    check("python", python_version, "Activate the Python 3.10 environment from README.")
    check("system_tools", system_tools, "Install ffmpeg (including ffprobe); see README.")
    check("packages", packages, "Install the CUDA extensions, then requirements.txt in README order.")
    torch = check("torch", lambda: importlib.import_module("torch"),
                  "Install the CUDA PyTorch wheel from README before installing extensions.")
    configured = check("configuration", configuration,
                       "Check --config, local environment settings and requirements.txt; a key is optional here.")
    settings, paths = configured if configured is not None else (None, {})

    def cuda():
        report["torch"], report["cuda_build"] = torch.__version__, torch.version.cuda
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        report["gpu"] = torch.cuda.get_device_name()
        report["compute_capability"] = list(torch.cuda.get_device_capability())

    def cuda_kernel():
        mamba = importlib.import_module("mamba_ssm").Mamba
        layer = mamba(d_model=16, d_state=16, d_conv=4, expand=2).cuda().eval()
        with torch.inference_mode():
            result = layer(torch.zeros(1, 32, 16, device="cuda"))
        torch.cuda.synchronize()
        if result.shape != (1, 32, 16) or not torch.isfinite(result).all():
            raise RuntimeError("invalid CUDA kernel result")

    check("cuda", cuda, "An NVIDIA CUDA GPU and compatible Windows driver/PyTorch are required.",
          requires=("torch",))
    check("cuda_kernel", cuda_kernel, "Install/rebuild Mamba and causal-conv1d in README order.",
          requires=("cuda",))
    report["cuda_kernel_smoke"] = checks["cuda_kernel"]["status"]

    def assets():
        report["assets"] = importlib.import_module("app.utils.assets").validation_report(settings=settings)
        if not report["assets"]["ready"]:
            raise FileNotFoundError("missing required assets")

    def smpl_resource():
        service = importlib.import_module("app.utils.smpl_template").NeutralSmplTemplateService(
            paths["smpl_model_root"]
        )
        report["smpl"] = {**service.status(), "loaded": False}
        if not report["smpl"]["available"]:
            raise FileNotFoundError("SMPL unavailable")
        return service

    check("assets", assets, "Place the required assets at the paths listed in the assets report.",
          requires=("configuration",))
    smpl = check("smpl_resource", smpl_resource, "Configure Neutral SMPL and install smplx; see README.",
                 requires=("configuration",))
    runtime = check("runtime", lambda: importlib.import_module("app.runtime").RuntimeServices(settings),
                    "Run with --models; if initialization fails, check requirements.txt and configuration.",
                    requires=("configuration", "torch"), enabled=args.models)

    def retriever():
        report["retrieval"] = runtime.retriever().status()

    def motion_library():
        report["motion_library"] = runtime.motion_library().status()

    def inpainting():
        completer = runtime.completer()
        completer._load()
        if runtime.remaker() is not completer:
            raise RuntimeError("Completer and Remaker must share one inpainting backend")
        report["completer"], report["remaker"] = "LOADED", "SHARED_WITH_COMPLETER"

    def smpl_model():
        smpl.template_json()
        report["smpl"]["loaded"] = True

    check("retriever_model", retriever,
          "Run with --models; check stage2_model.pt and its paired retriever/index files if loading fails.",
          requires=("runtime",), enabled=args.models)
    check("motion_library", motion_library,
          "Run with --models; prepare the matching Motion Library if loading fails.",
          requires=("runtime",), enabled=args.models)
    check("inpainting_model", inpainting,
          "Run with --models; check stage3_model.pt, condition_normalizer.npz and available GPU memory.",
          requires=("runtime", "cuda_kernel"), enabled=args.models)
    check("smpl_model", smpl_model,
          "Run with --models; use Neutral SMPL v1.1.0 and install chumpy==0.70 as shown in README.",
          requires=("smpl_resource",), enabled=args.models)
    checks["online_api"] = {
        "status": "SKIPPED",
        "reason": "No API requests are made. Key validity, model access and the online workflow are not tested.",
    }
    if report["api_key_configured"] is False:
        checks["online_api"]["reason"] += " Configure OPENAI_API_KEY locally before using the application."
    report["status"] = "FAIL" if any(item["status"] == "FAIL" for item in checks.values()) else "PASS"
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
