"""
generate the Kubernetes PrometheusRule manifest from the local alert rules.

Source:
    deploy/prometheus/rules/rag.yml

Output:
    deploy/k8s/45-alert-rules.yaml

vLLM rules are intentionally excluded
"""

from __future__ import annotations

import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "deploy" / "prometheus" / "rules" / "rag.yml"
TARGET = ROOT / "deploy" / "k8s" / "45-alert-rules.yaml"

HEADER = """\
# GENERATED FILE - DO NOT EDIT BY HAND.
#
# Written by scripts/make_prometheusrule.py from deploy/prometheus/rules/rag.yml,
# which is the single source of truth for these thresholds. Edit that file and
# re-run the script; a test fails if this copy is stale.
#
# The Prometheus Operator discovers this by LABEL, not by namespace: the
# `release: kube-prometheus-stack` label below must match the Helm release name
# you installed the stack with, and must match the operator's ruleSelector. If
# the alerts never appear in the Prometheus UI, that label is the first thing to
# check (`kubectl get prometheus -A -o yaml | grep -A5 ruleSelector`).
"""


def render() -> str:
    source = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    doc = {
        "apiVersion": "monitoring.coreos.com/v1",
        "kind": "PrometheusRule",
        "metadata": {
            "name": "fdr-rag-rules",
            "namespace": "fdr",
            "labels": {"release": "kube-prometheus-stack", "app": "fdr"},
        },
        "spec": {"groups": source["groups"]},
    }
    return HEADER + yaml.safe_dump(doc, sort_keys=False, width=100, default_flow_style=False)


def main(argv: list[str]) -> int:
    rendered = render()
    if "--check" in argv:
        current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
        if current != rendered:
            print(f"{TARGET} is STALE: re-run scripts/make_prometheusrule.py", file=sys.stderr)
            return 1
        print(f"{TARGET} is up to date")
        return 0
    TARGET.write_text(rendered, encoding="utf-8")
    groups = yaml.safe_load(rendered)["spec"]["groups"]
    n = sum(len(g["rules"]) for g in groups)
    print(f"wrote {TARGET} ({len(groups)} groups, {n} alert rules)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
