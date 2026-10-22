"""Dump every run's config, summary, and scalar history from W&B to JSON.

.env provides auth via iwild.config. Output: reports/wandb_dump.json with per-run
{user, name, state, url, config, summary, history:{metric:[[step,value],...]}}.
History is sampled (max ~400 points per metric) to keep the dump small.
"""

import json
import os
import sys

import iwild.config  # noqa: F401  (loads .env)


def sample(series, max_points=400):
    if len(series) <= max_points:
        return series
    stride = len(series) / max_points
    return [series[int(i * stride)] for i in range(max_points)]


def main():
    import wandb

    project = os.environ.get("WANDB_PROJECT") or "team-support-vectors-idlf26/iwilds"
    if "/" not in project:
        project = f"{os.environ.get('WANDB_ENTITY', 'team-support-vectors-idlf26')}/{project}"
    api = wandb.Api()
    out = []
    for run in api.runs(project):
        user = getattr(run, "user", None)
        uname = (getattr(user, "name", "") or "") if user else ""
        summary = {k: v for k, v in run.summary.items() if not k.startswith("_")}
        config = {k: v for k, v in run.config.items() if not k.startswith("_")}
        entry = {
            "name": run.name,
            "user": uname,
            "state": run.state,
            "url": run.url,
            "created": str(run.created_at),
            "config": config,
            "summary": summary,
            "history": {},
        }
        try:
            hist = run.history(keys=None, samples=5000)
            for col in hist.columns:
                if col in ("_step", "_runtime", "_timestamp", "examples"):
                    continue
                vals = hist[[ "_step", col]].dropna()
                if len(vals) == 0:
                    continue
                pts = [[int(s), float(v)] for s, v in zip(vals["_step"], vals[col])
                       if isinstance(v, (int, float))]
                if pts:
                    entry["history"][col] = sample(pts)
        except Exception as exc:  # noqa: BLE001
            entry["history_error"] = str(exc)
        out.append(entry)
        print(f"dumped {uname[:18]:20s} {run.name[:50]}", file=sys.stderr)

    path = "reports/wandb_dump.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(f"wrote {path} ({len(out)} runs)", file=sys.stderr)


if __name__ == "__main__":
    main()
