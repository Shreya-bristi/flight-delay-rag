"""
validate Kubernetes deployment configuration before resources are applied.

Checks repository-level deployment invariants such as:
- shared conversation secrets across API replicas
- vLLM context-window compatibility
- StorageClass references
- NLB configuration and deployment ordering
- Postgres NetworkPolicy access
- IRSA and ServiceAccount usage
- ConfigMap ordering and .env parity
- Alertmanager SNS delivery
- unresolved deployment placeholders

Usage:
    .venv/Scripts/python.exe scripts/k8s_preflight.py
    .venv/Scripts/python.exe scripts/k8s_preflight.py --target monitoring
    .venv/Scripts/python.exe scripts/k8s_preflight.py --target vllm
    .venv/Scripts/python.exe scripts/k8s_preflight.py --static-only

Exit 0 means all checks passed; exit 1 reports configuration problems.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
K8S = ROOT / "deploy" / "k8s"
HELM = ROOT / "deploy" / "helm"
MAKEFILE = ROOT / "Makefile"
# Single source of the fdr-config ConfigMap
CONFIG_FILE = "18-config.yaml"

# files used by the default EKS deployment
DEPLOY_FILES = [
    "eksctl-cluster.yaml", "00-namespace.yaml", "05-storageclass.yaml", "10-postgres.yaml",
    "15-networkpolicy.yaml", "18-config.yaml", "50-index-job.yaml", "19-api-service.yaml",
    "20-api.yaml", "60-db-backup.yaml",
]
# workloads allowed to use IRSA in the fdr namespace
AWS_WORKLOADS = {"db-backup": "fdr-backup"}
TARGETS = {
    "deploy": [K8S / f for f in DEPLOY_FILES],
    "monitoring": [HELM / "kube-prometheus-stack-values.yaml"],
    "vllm": [K8S / "30-vllm.yaml"],
}
PLACEHOLDER_RE = re.compile(r"\bACCOUNT_ID\b|\bREPLACE_WITH_[A-Z0-9_]+\b")


def _docs(name: str) -> list[dict]:
    text = (K8S / name).read_text(encoding="utf-8")
    return [d for d in yaml.safe_load_all(text) if d]


def _find(name: str, kind: str, meta_name: str | None = None) -> dict:
    for doc in _docs(name):
        if doc.get("kind") == kind and (meta_name is None or doc["metadata"]["name"] == meta_name):
            return doc
    raise LookupError(f"{name}: no {kind} {meta_name or ''}".strip())


def _pod_spec(workload: dict) -> dict:
    spec = workload["spec"]
    if workload["kind"] == "CronJob":
        spec = spec["jobTemplate"]["spec"]
    return spec["template"]


def _all_containers(pod_spec: dict) -> list[dict]:
    return list(pod_spec.get("initContainers") or []) + list(pod_spec.get("containers") or [])


def _secret_env(container: dict) -> dict[str, str]:
    out = {}
    for env in container.get("env") or []:
        ref = (env.get("valueFrom") or {}).get("secretKeyRef")
        if ref:
            out[env["name"]] = ref["key"]
    return out


def _workloads() -> list[tuple[str, dict]]:
    found = []
    for path in sorted(K8S.glob("*.yaml")):
        if path.name == "eksctl-cluster.yaml":
            continue
        for doc in _docs(path.name):
            if doc.get("kind") in {"Deployment", "StatefulSet", "Job", "CronJob"}:
                found.append((path.name, doc))
    return found


def _config() -> dict:
    return _find(CONFIG_FILE, "ConfigMap", "fdr-config")["data"]


def _config_refs(pod_spec: dict) -> set[str]:
    refs = set()
    for c in _all_containers(pod_spec):
        refs |= {e["configMapRef"]["name"] for e in c.get("envFrom") or [] if "configMapRef" in e}
    return refs


def _eks_deploy_applies() -> list[str]:
    """return manifests applied by make eks-deploy in deployment"""
    lines = MAKEFILE.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("eks-deploy:"))
    order: list[str] = []
    for line in lines[start + 1:]:
        if line and not line.startswith(("\t", " ", "#")):
            break   # the next target
        m = re.search(r"kubectl apply -f deploy/k8s/(\S+\.yaml)", line)
        if m:
            order.append(m.group(1))
        elif "targetgroupbindings" in line and "<wait-tgb>" not in order:
            order.append("<wait-tgb>")
    return order


def _read_env(path: pathlib.Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key.strip().upper()] = value
    return values


# ------------------------------------------------------------------ checks
def check_conversation_secret() -> list[str]:
    api = _find("20-api.yaml", "Deployment", "api")
    if int(api["spec"].get("replicas", 1)) <= 1:
        return []
    container = _pod_spec(api)["spec"]["containers"][0]
    if "CONVERSATION_SECRET" not in _secret_env(container):
        return [f"20-api.yaml: {api['spec']['replicas']} replicas but CONVERSATION_SECRET is not read "
                "from a Secret - each pod would sign conversations with its own random key (403 across pods)"]
    return []


def check_vllm_window() -> list[str]:
    data = _config()
    need = int(data["LLM_CONTEXT_WINDOW"]) + int(data["LLM_MAX_COMPLETION_TOKENS"])
    args = _pod_spec(_find("30-vllm.yaml", "Deployment", "vllm"))["spec"]["containers"][0]["args"]
    have = int(args[args.index("--max-model-len") + 1])
    if have < need:
        return [f"30-vllm.yaml: --max-model-len {have} < LLM_CONTEXT_WINDOW + LLM_MAX_COMPLETION_TOKENS "
                f"= {need} ({CONFIG_FILE}): a maximum-size request would be rejected"]
    return []


def check_same_image() -> list[str]:
    api = _pod_spec(_find("20-api.yaml", "Deployment", "api"))["spec"]["containers"][0]["image"]
    job = _pod_spec(_find("50-index-job.yaml", "Job", "index-corpus"))["spec"]["containers"][0]["image"]
    if api != job:
        return [f"the index Job image ({job}) differs from the API image ({api}): "
                "/ready would refuse an index built by another build"]
    return []


def check_storage_classes() -> list[str]:
    defined = {d["metadata"]["name"] for d in _docs("05-storageclass.yaml") if d["kind"] == "StorageClass"}
    problems = []
    for path in sorted(K8S.glob("*.yaml")):
        if path.name == "eksctl-cluster.yaml":
            continue
        for doc in _docs(path.name):
            claims = []
            if doc.get("kind") == "PersistentVolumeClaim":
                claims.append(doc["spec"])
            if doc.get("kind") == "StatefulSet":
                claims += [t["spec"] for t in doc["spec"].get("volumeClaimTemplates") or []]
            for spec in claims:
                name = spec.get("storageClassName")
                if name not in defined:
                    problems.append(f"{path.name}: storageClassName {name!r} is not defined in "
                                    f"05-storageclass.yaml ({sorted(defined)}) - the claim would stay Pending")
    return problems


def check_load_balancer() -> list[str]:
    problems = []
    svc = _find("19-api-service.yaml", "Service", "api")
    for name in sorted(p.name for p in K8S.glob("*.yaml") if p.name != "19-api-service.yaml"):
        if name != "eksctl-cluster.yaml" and any(d.get("kind") == "Service" and d["spec"].get("type") == "LoadBalancer"
                                                 for d in _docs(name)):
            problems.append(f"{name}: a LoadBalancer Service outside 19-api-service.yaml - it must be applied "
                            "before its pods, or they miss the controller's readiness gate")
    if svc["spec"].get("type") == "LoadBalancer":
        ann = svc["metadata"].get("annotations") or {}
        if ann.get("service.beta.kubernetes.io/aws-load-balancer-type") != "external":
            problems.append("19-api-service.yaml: LoadBalancer Service without aws-load-balancer-type: external - "
                            "the in-tree controller would create a Classic Load Balancer")
        preserve = ann.get("service.beta.kubernetes.io/aws-load-balancer-target-group-attributes", "")
        if (ann.get("service.beta.kubernetes.io/aws-load-balancer-nlb-target-type") == "ip"
                and "preserve_client_ip.enabled=true" not in preserve):
            problems.append("19-api-service.yaml: ip targets without preserve_client_ip - every passenger would "
                            "share the NLB's address in the per-client rate limit")
    data = _config()
    if str(data.get("TRUST_FORWARDED_FOR", "false")).lower() == "true":
        problems.append(f"{CONFIG_FILE}: TRUST_FORWARDED_FOR=true behind an NLB, which sets no "
                        "X-Forwarded-For: clients could choose their own rate-limit key")
    return problems


def check_postgres_clients() -> list[str]:
    policy = _find("15-networkpolicy.yaml", "NetworkPolicy", "postgres-clients-only")
    allowed: set[str] = set()
    for rule in policy["spec"].get("ingress") or []:
        for peer in rule.get("from") or []:
            selector = peer.get("podSelector") or {}
            allowed |= {v for k, v in (selector.get("matchLabels") or {}).items() if k == "app"}
            for expr in selector.get("matchExpressions") or []:
                if expr["key"] == "app" and expr["operator"] == "In":
                    allowed |= set(expr["values"])
    problems = []
    for name, workload in _workloads():
        pod = _pod_spec(workload)
        if any("PG_DSN" in _secret_env(c) for c in _all_containers(pod["spec"])):
            app = ((pod.get("metadata") or {}).get("labels") or {}).get("app")
            if app not in allowed:
                problems.append(f"{name}: {workload['metadata']['name']} reads PG_DSN but its pods "
                                f"(app={app!r}) are not admitted by the postgres NetworkPolicy {sorted(allowed)}")
    return problems


def check_aws_identities() -> list[str]:
    """Only the backup job may run as an IRSA ServiceAccount, every other
    workload runs with no ServiceAccount token at all."""
    cluster = yaml.safe_load((K8S / "eksctl-cluster.yaml").read_text(encoding="utf-8"))
    irsa = {s["metadata"]["name"] for s in cluster["iam"]["serviceAccounts"]
            if s["metadata"]["namespace"] == "fdr"}
    problems = []
    if irsa != set(AWS_WORKLOADS.values()):
        problems.append(f"eksctl-cluster.yaml: IRSA ServiceAccounts in fdr are {sorted(irsa)}, "
                        f"expected only {sorted(AWS_WORKLOADS.values())}")
    for name, workload in _workloads():
        spec = _pod_spec(workload)["spec"]
        wl = workload["metadata"]["name"]
        sa = spec.get("serviceAccountName")
        if wl in AWS_WORKLOADS:
            if sa != AWS_WORKLOADS[wl]:
                problems.append(f"{name}: {wl} must run as {AWS_WORKLOADS[wl]}, not {sa!r}")
        elif sa in irsa:
            problems.append(f"{name}: {wl} runs as IRSA account {sa!r} but calls no AWS API")
        elif spec.get("automountServiceAccountToken") is not False:
            problems.append(f"{name}: {wl} mounts a ServiceAccount token it never uses "
                            "(set automountServiceAccountToken: false)")
    return problems


def check_alert_delivery() -> list[str]:
    """Validate Alertmanager SNS delivery through the IRSA ServiceAccount"""
    values = yaml.safe_load((HELM / "kube-prometheus-stack-values.yaml").read_text(encoding="utf-8"))
    am = values["alertmanager"]
    cluster = yaml.safe_load((K8S / "eksctl-cluster.yaml").read_text(encoding="utf-8"))
    monitoring = {s["metadata"]["name"]: s for s in cluster["iam"]["serviceAccounts"]
                  if s["metadata"]["namespace"] == "monitoring"}
    problems = []
    sa = am.get("serviceAccount") or {}
    if sa.get("create") is not False or sa.get("name") not in monitoring:
        problems.append("kube-prometheus-stack-values.yaml: Alertmanager must use the eksctl IRSA "
                        f"ServiceAccount (create: false, name in {sorted(monitoring)}), got {sa}")
    else:
        # Accept the repository placeholder or a resolved 12-digit AWS account ID
        arns = monitoring[sa["name"]].get("attachPolicyARNs") or []
        if len(arns) != 1 or not re.fullmatch(r"arn:aws:iam::(ACCOUNT_ID|\d{12}):policy/fdr-alerts-publish", arns[0]):
            problems.append(f"eksctl-cluster.yaml: {sa['name']} must carry only the fdr-alerts-publish policy")
    cfg = am.get("config") or {}
    if any(k.startswith("smtp_") for k in cfg.get("global") or {}) or any(
            r.get("email_configs") for r in cfg.get("receivers") or []):
        problems.append("kube-prometheus-stack-values.yaml: an SMTP/email receiver is back - alerts go "
                        "through SNS; no mail password belongs in the cluster")
    topics = [c.get("topic_arn", "") for r in cfg.get("receivers") or [] for c in r.get("sns_configs") or []]
    if not topics or not all(t.endswith(":fdr-alerts") for t in topics):
        problems.append(f"kube-prometheus-stack-values.yaml: no SNS receiver for the fdr-alerts topic ({topics})")
    elif (cfg.get("route") or {}).get("receiver") not in {r["name"] for r in cfg["receivers"] if r.get("sns_configs")}:
        problems.append("kube-prometheus-stack-values.yaml: the default route does not go to the SNS receiver")
    return problems


def check_config_order() -> list[str]:
    """ensure fdr-config is defined once and applied before its consumers"""
    problems = []
    defined: dict[str, list[str]] = {}
    for path in sorted(K8S.glob("*.yaml")):
        if path.name == "eksctl-cluster.yaml":
            continue
        for doc in _docs(path.name):
            if doc.get("kind") == "ConfigMap":
                defined.setdefault(doc["metadata"]["name"], []).append(path.name)
    if defined.get("fdr-config") != [CONFIG_FILE]:
        problems.append(f"fdr-config must be defined exactly once, in {CONFIG_FILE}; "
                        f"found in {defined.get('fdr-config', [])}")
    index = _pod_spec(_find("50-index-job.yaml", "Job", "index-corpus"))["spec"]
    if "fdr-config" not in _config_refs(index):
        problems.append("50-index-job.yaml: the index Job no longer reads fdr-config - it would index "
                        "with settings the API does not serve")

    order = _eks_deploy_applies()
    applied = {name: i for i, name in enumerate(order)}
    for name, workload in _workloads():
        if name not in applied:
            continue
        for ref in sorted(_config_refs(_pod_spec(workload)["spec"])):
            homes = [f for f in defined.get(ref, []) if f in applied]
            if not homes:
                problems.append(f"Makefile eks-deploy applies {name} but never the ConfigMap {ref!r} it reads")
            elif min(applied[f] for f in homes) > applied[name]:
                problems.append(f"Makefile eks-deploy applies {name} BEFORE {homes[0]}, which defines the "
                                f"ConfigMap {ref!r} it reads - its pods fail with CreateContainerConfigError")
    for before, after, why in [
        ("50-index-job.yaml", "20-api.yaml", "the index must be complete before the API rolls out"),
        ("19-api-service.yaml", "20-api.yaml", "the Service must exist before the API pods (readiness gate)"),
        ("19-api-service.yaml", "<wait-tgb>", "the TargetGroupBinding wait belongs after the Service"),
        ("<wait-tgb>", "20-api.yaml", "wait for the TargetGroupBinding before creating the API pods"),
    ]:
        if before not in applied or after not in applied or applied[before] > applied[after]:
            problems.append(f"Makefile eks-deploy: {before} must come before {after} ({why})")
    return problems


def check_env_matches(env_path: pathlib.Path) -> list[str]:
    """compare shared non-secret settings between .env and fdr-config"""
    if not env_path.exists():
        return []
    env = _read_env(env_path)
    data = _config()
    differ = sorted(k for k, v in data.items() if k.upper() in env and env[k.upper()] != str(v))
    if differ:
        return [f"{CONFIG_FILE} ConfigMap disagrees with {env_path.name} on: {', '.join(differ)} "
                "(CLAUDE.md: the cluster copy must match .env)"]
    return []


STATIC_CHECKS = [check_conversation_secret, check_vllm_window, check_same_image,
                 check_storage_classes, check_load_balancer, check_postgres_clients,
                 check_aws_identities, check_config_order, check_alert_delivery]


def static_problems() -> list[str]:
    problems = []
    for check in STATIC_CHECKS:
        problems += check()
    return problems


def placeholder_problems(target: str) -> list[str]:
    """Find unresolved placeholders in files used by the selected target"""
    problems = []
    for path in TARGETS[target]:
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            for marker in sorted(set(PLACEHOLDER_RE.findall(line))):
                problems.append(f"{path.relative_to(ROOT).as_posix()}:{n}: placeholder {marker} not replaced")
    return problems


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--target", choices=sorted(TARGETS), default="deploy")
    parser.add_argument("--static-only", action="store_true",
                        help="skip the placeholder scan (the repository ships the markers)")
    parser.add_argument("--env", type=pathlib.Path, default=ROOT / ".env",
                        help="compare the ConfigMap against this file (skipped, with a warning, if absent)")
    parser.add_argument("--require-env", action="store_true",
                        help="FAIL if the .env file is absent (the machine that deploys uses this)")
    args = parser.parse_args(argv)

    problems = static_problems() + check_env_matches(args.env)
    if not args.env.exists():
        if args.require_env:
            problems.append(f"{args.env} not found: the {CONFIG_FILE} <-> .env parity check cannot run. "
                            "Run the preflight from the real repository, where .env exists.")
        else:
            print(f"WARNING: {args.env.name} not found; .env <-> fdr-config parity check skipped. "
                  "This result does NOT show that the cluster configuration matches .env.")
    if not args.static_only:
        problems += placeholder_problems(args.target)
    for p in problems:
        print(f"FAIL  {p}")
    if problems:
        print(f"\n{len(problems)} problem(s): fix them before applying anything.", file=sys.stderr)
        return 1
    print(f"preflight OK ({'static checks' if args.static_only else 'target ' + args.target})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
