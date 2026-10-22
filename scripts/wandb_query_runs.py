"""List W&B runs matching a user + name filter and dump their summary/config.

Reads WANDB_API_KEY / WANDB_BASE_URL / WANDB_ENTITY / WANDB_PROJECT from the
environment (`.env` is loaded via iwild.config). Never prints secret values.

Usage:
    python scripts/wandb_query_runs.py --user nnoukastephen --name-filter iwildcam
"""

import argparse
import json

import iwild.config  # noqa: F401  (loads .env: WANDB_API_KEY, WANDB_BASE_URL, ...)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default=None, help="username substring filter (default: all)")
    parser.add_argument("--name-filter", default=None, help="run-name substring filter")
    parser.add_argument("--project", default=None, help="e.g. team/iwilds (default: $WANDB_PROJECT)")
    args = parser.parse_args()

    import os

    import wandb

    project = args.project or os.environ.get("WANDB_PROJECT")
    if not project:
        # WANDB_PROJECT may hold just the project name; try entity/project
        entity = os.environ.get("WANDB_ENTITY")
        project = f"{entity}/iwilds" if entity else "iwilds"
        print(f"WANDB_PROJECT not set; querying '{project}'")

    api = wandb.Api()
    runs = api.runs(project)
    matched = []
    for run in runs:
        user = (getattr(run, "user", None) and (run.user.name or "")) or ""
        if args.user and args.user.lower() not in user.lower():
            continue
        if args.name_filter and args.name_filter.lower() not in run.name.lower():
            continue
        summary = {k: v for k, v in run.summary.items() if not k.startswith("_")}
        config = {k: v for k, v in run.config.items() if not k.startswith("_")}
        matched.append(
            {
                "name": run.name,
                "state": run.state,
                "user": user,
                "url": run.url,
                "created": str(run.created_at),
                "summary": summary,
                "config": {k: v for k, v in list(config.items())[:40]},
            }
        )

    print(json.dumps(matched, indent=2, default=str))
    print(f"\n{len(matched)} run(s) matched", file=__import__("sys").stderr)


if __name__ == "__main__":
    main()
