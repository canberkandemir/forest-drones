"""Verify a CUDA MMDetection 2.x environment for ForestPersons baselines.

Run this on a CUDA machine, not on the local Apple Silicon Mac.
"""

from __future__ import annotations

import json
import platform


def main() -> None:
    report: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        report["torch"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["torch_cuda"] = torch.version.cuda
        report["cuda_device_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            report["cuda_device_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            report["cuda_device_memory_gib"] = round(props.total_memory / (1024**3), 2)
    except Exception as exc:  # pragma: no cover - depends on external CUDA env.
        report["torch_error"] = repr(exc)

    for module_name in ("mmcv", "mmdet", "pycocotools"):
        try:
            module = __import__(module_name)
            report[module_name] = getattr(module, "__version__", "installed")
        except Exception as exc:  # pragma: no cover - depends on external env.
            report[f"{module_name}_error"] = repr(exc)

    print(json.dumps(report, indent=2))
    if not report.get("cuda_available"):
        raise SystemExit("CUDA is not available; do not run baseline training/inference.")


if __name__ == "__main__":
    main()
