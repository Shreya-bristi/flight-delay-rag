# Deploying to AWS (EKS)

I deployed the chatbot end to end on Amazon EKS, ran the acceptance checks, and tore it down
again; the notes at the bottom are what that run taught me.

The generator openai/gpt-oss-20b is hosted Groq, so the default deployment needs no GPU.

## What gets built

```text
Terraform (long-lived, cents/month)      eksctl + Helm + kubectl (per deployment)
  S3 bucket   eval runs, db-backups/        EKS 1.35, 2 x m7i-flex.large
  ECR         fdr-api (immutable tags)      AWS Load Balancer Controller -> NLB
  IAM         backup + alert policies       Postgres + pgvector (StatefulSet, gp3)
  SNS         fdr-alerts -> email           index Job, API x2, backup CronJob
  Budget      $40/month                     kube-prometheus-stack
```

Terraform owns the resources worth keeping between deployments. The cluster is disposable:
I create it for a session and delete it after.

Design choices:

- **Only one workload has an AWS identity.**  Only the backup CronJob has an AWS identity. It uses the fdr-backup IRSA role, which is limited to writing backups under db-backups/. It cannot list, read, or delete objects. The API, index Job, and Postgres do not have AWS roles and do not receive ServiceAccount tokens.
- **No mail password anywhere.** Alertmanager publishes alerts to SNS using its own IRSA role, and SNS handles email delivery.
- **NLB with IP targets and preserved client IPs**, traffic goes directly to pod IPs while preserving the original client address. So API rate limiting is based on the passenger's IP rather than the load balancer. NLB health checks use the /ready endpoint.
- **NetworkPolicies** restrict Postgres to the pods that need it (API, index Job, backup).
- **Immutable image tags.** So that the image I tested and the image running can't drift apart.
- **Pinned Helm chart versions** Helm chart versions are fixed in the Makefile to keep deployments reproducible.
- **One config source.** `deploy/k8s/18-config.yaml`is in sync with `.env`, and a preflight script refuses to deploy if the values differ.

## Prerequisites

AWS CLI, Docker, kubectl, eksctl, Helm, Terraform, make, openssl and a GNU shell (I use
Git Bash on Windows). You also need a Groq key(or any API key of your choice) and an AirLabs key.

Region is `us-east-2` for me.

On Windows python may may not be isn't on PATH, so pass the venv interpreter to make:
`PYTHON=.venv/Scripts/python.exe`.

## Account checks

```bash
aws sts get-caller-identity
aws eks describe-cluster-versions --region us-east-2 --output table
```

The cluster file pins Kubernetes 1.35. Make sure it's still in standard support: extended
support bills the control plane at $0.60/hr instead of $0.10. 

## Terraform

```bash
cd deploy/terraform
cp terraform.tfvars.example terraform.tfvars   # set bucket_suffix and alert_email
terraform init && terraform apply
cd ../..
```

This creates the S3 bucket (versioned, encrypted, backups expire after 14 days), the ECR
repo, the two IAM policies(fdr-db-backup, fdr-alerts-publish), the SNS topic with an email subscription, and the budget
(alerts that given by my rate of usage I wills spend above 60% of my budget in the month and when I spent already more than 90% ).

Confirm the SNS subscription from the email AWS sends, otherwise no alert is ever
delivered:

```bash
aws sns list-subscriptions-by-topic --region us-east-2 \
  --topic-arn "$(terraform -chdir=deploy/terraform output -raw alerts_topic_arn)"
# SubscriptionArn must be a real ARN, not PendingConfirmation
```

terraform.tfstate is local and is in .gitignore. The bucket has `prevent_destroy`, so the storage bucket is intentionally protected from accidental deletion.
## 3. Image and placeholders

```bash
make ecr-push      # builds, pushes, prints a timestamp tag
```

The manifests ship with placeholders on purpose. I fill them in a copy of the working
folder(with `.env` and the Terraform state in it), never in the repo:

```bash
TAG=<tag from ecr-push>
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
BUCKET=$(terraform -chdir=deploy/terraform output -raw s3_bucket)

sed -i "s/REPLACE_WITH_IMMUTABLE_TAG/$TAG/" deploy/k8s/20-api.yaml deploy/k8s/50-index-job.yaml
sed -i "s/ACCOUNT_ID/$ACCOUNT/g" deploy/k8s/*.yaml deploy/helm/*.yaml
sed -i "s/REPLACE_WITH_S3_BUCKET/$BUCKET/" deploy/k8s/60-db-backup.yaml
```

This has to happen before cluster creation, because `eksctl-cluster.yaml` contains the account ID too.

## Cluster (~20 min)

```bash
make eks-up
```

This creates the cluster, node group, OIDC provider, the three IRSA service accounts
(`fdr-backup`, `fdr-alertmanager`, `aws-load-balancer-controller`), the EBS CSI driver,
and the VPC CNI with network policy enforcement on.

```bash
kubectl get nodes
kubectl get pods -n kube-system -l app.kubernetes.io/name=aws-ebs-csi-driver
kubectl get pods -n kube-system -l k8s-app=aws-node     # 2/2 = policy agent running
```

If the node group fails, eksctl stops before it writes the kubeconfig and before it installs
the add-ons. Don't re-run `eks-up`, since the control plane already exists. Instead:

```bash
eksctl delete nodegroup --cluster fdr --region us-east-2 --name cpu --wait
eksctl create nodegroup -f deploy/k8s/eksctl-cluster.yaml --include=cpu
aws eks update-kubeconfig --region us-east-2 --name fdr
eksctl create addon -f deploy/k8s/eksctl-cluster.yaml
eksctl create iamserviceaccount -f deploy/k8s/eksctl-cluster.yaml --approve
```

## Load balancer controller

Install it before the API Service exists. Without it, EKS makes a Classic Load Balancer.

```bash
make eks-lb-controller
```

## Secrets

```bash
kubectl create namespace fdr --dry-run=client -o yaml | kubectl apply -f -

PGPASS=$(openssl rand -hex 24)    # hex, so it's safe inside the DSN
kubectl -n fdr create secret generic fdr-secrets \
  --from-literal=postgres-password="$PGPASS" \
  --from-literal=pg-dsn="postgresql://fdr:${PGPASS}@postgres-0.postgres.fdr.svc.cluster.local:5432/fdr" \
  --from-literal=llm-api-key='<groq key>' \
  --from-literal=airlabs-key='<airlabs key>' \
  --from-literal=conversation-secret="$(openssl rand -hex 32)"
unset PGPASS
```

Secrets never go in the image, Git or Terraform state. The API replicas share a `conversation-secret` used to sign and verify conversation tokens. Both replicas must use the same value; otherwise, a token created by one pod cannot be verified by the other and the request may return 403 Forbidden.

## Deploy

```bash
.venv/Scripts/python.exe scripts/k8s_preflight.py --require-env
make eks-deploy PYTHON=.venv/Scripts/python.exe
```

The preflight must print `preflight OK` **with no WARNING line**. It checks for leftover
placeholders, ConfigMap/`.env` parity, apply order, StorageClasses, NetworkPolicy coverage,
IRSA, and the load balancer annotations.

`eks-deploy` applies, in order:

1. Namespace, StorageClass and NetworkPolicies.
2. Postgres. It waits for it.
3. The `fdr-config` ConfigMap.
4. The index Job. It waits about 23 minutes.
5. It asks how many AirLabs calls are left this month. Press Enter to skip.
6. The API Service. It then waits for the TargetGroupBinding.
7. The API Deployment.
8. The backup CronJob.

The Service goes in before the Deployment because the controller only injects its readiness
gate into pods created after the TargetGroupBinding exists.

```bash
HOST=$(kubectl -n fdr get svc api -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
curl http://$HOST/health
curl http://$HOST/ready        # 503 until the index matches the running settings
```

API pods download about 2.5 GB of model weights on first start. A startup probe gives them
10 minutes. Rollouts replace one pod at a time with no surge pod, a PDB keeps one replica up
during drains, and the two replicas are spread across nodes.

## Monitoring

```bash
make eks-monitoring GRAFANA_PASSWORD='<pick one>' PYTHON=.venv/Scripts/python.exe
```

This installs kube-prometheus-stack from `deploy/helm/kube-prometheus-stack-values.yaml`,
plus the ServiceMonitor, the alert rules, and both Grafana dashboards as ConfigMaps.

The alert path is Prometheus -> Alertmanager -> SNS -> email. Components EKS doesn't expose
(scheduler, etcd, and so on) are disabled so they don't fire `...Down` alerts forever.

```bash
kubectl -n monitoring port-forward svc/kube-prometheus-stack-prometheus 9090:9090
kubectl -n monitoring port-forward svc/kube-prometheus-stack-grafana 3000:80
```

Alert rules are edited in `deploy/prometheus/rules/rag.yml`. Then
`scripts/make_prometheusrule.py` regenerates `deploy/k8s/45-alert-rules.yaml`, and a test
fails if the two drift.

Optional: `make eks-logs` ships container logs to CloudWatch, and `make eks-logs-retention`
sets them to expire after a day. It's off by default because Container Insights is billed.

## Backups

`pg_dump` runs nightly at 03:00 UTC to `s3://<bucket>/db-backups/`. To run one now and
restore it:

```bash
make db-backup

kubectl -n fdr scale deployment api --replicas=0
aws s3 cp s3://<bucket>/db-backups/<file>.dump - \
  | kubectl -n fdr exec -i postgres-0 -- pg_restore --clean --if-exists --no-owner -U fdr -d fdr
kubectl -n fdr scale deployment api --replicas=2
```

Postgres is a single instance on one EBS volume. The backup limits data loss to a day; it
isn't high availability. For real traffic I'd move to RDS, which only changes `pg-dsn`.


## HTTPS ( TO do)

The Service listens on plain HTTP. To enable HTTPS with a domain I have to request an ACM certificate in `us-east-2` and validate it by DNS.I have to update the DNS so that it points to NLB.

(din't get enough time to do this a the complexity of the project became exponential in a short span. But will make this change)

## Teardown

```bash
make eks-down
make audit
```

`eks-down` does things in an order that matters:

a. A final backup. It stops if the backup fails; `SKIP_BACKUP=1` overrides that.
b. Delete the Services while the controller still runs, so the NLB is removed.
c. Delete the StatefulSet and PVCs while the CSI driver still runs, so the EBS volume goes.
d. Delete the cluster.

Getting either of the first two wrong and I'm left with an orphaned NLB or volume that keeps
billing.

`make audit` returns `CLEAR`, `FOUND` (lists what's still running) or `UNKNOWN` (a query
failed, which is not a clean bill). Billing data lags, so I check in AWS next morning.
## Cost

| item | rate |
|---|---|
| EKS control plane | $0.10/hr |
| 2 x m7i-flex.large | ~$0.19/hr |
| NLB | ~$0.023/hr + LCU |
| 10 GiB gp3 | ~$0.001/hr |
| S3, ECR, SNS, budget | cents/month |

That comes to about $0.32/hr while the cluster is up. The real risk is forgetting to tear it
down, not the hourly rate.

## What the first deploy taught me

Everything above passed offline checks before I deployed. The live run still found four
problems:

- **Free-plan accounts refuse non-free-tier instances.** The node group failed with
  `AsgInstanceLaunchFailures` on `t3.medium`. I switched to `m7i-flex.large`, which is on
  the eligible list, and wrote down the node group recovery steps (step 4).
- **The index Job crashed at startup.** The app refuses to start with `LLM_PROVIDER=groq`
  and no key, and the Job never calls the LLM. It now gets a placeholder key instead of the
  real one.
- **Grafana was OOMKilled** three times at 384Mi. It's now 768Mi.
- **`KubeCPUOvercommit` fires on two nodes.** CPU requests are 2.31 of 3.86 cores, so one
  node can't hold everything if the other dies. I accepted that for a two-node demo. A
  re-index doesn't fit beside the running stack by CPU either: scale the API to 1 replica,
  or the node group to 3, for the duration.

## Known limitations

- HTTP only until a domain and certificate are added.
- Single-instance Postgres.
- No autoscaling. `maxSize: 3` is headroom for a manual scale-up, and an autoscaler on a
  demo budget mostly produces surprise bills.
- Unrestricted API egress. Groq, AirLabs and Hugging Face are DNS names on shifting IP
  ranges, so this would need an egress proxy or an FQDN-aware policy.
- Local Terraform state. If I lose the file terraform loses track of everything. I should use a remote backend and store the state in an S3 bucket with versioning and encryption. (Again i plan to do it in future, this time it's timee constarined)
- No self-hosted model. I tried Qwen locally through Ollama and it was too slow on my 4 GB
  GPU, so I didn't go ahead with vLLM and use a hosted model (Groq) instead.