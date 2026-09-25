# AWS Deployment Runbook

This runbook covers deployment, verification, monitoring, backup, and teardown of the flight-delay RAG application on Amazon EKS.

The default deployment uses a hosted Groq generator and does **not** require a GPU.

## Architecture

```text
EKS
├── API Deployment (2 replicas)
├── PostgreSQL + pgvector
├── corpus indexing Job
├── Network Load Balancer
├── nightly Postgres backup CronJob
└── kube-prometheus-stack
    ├── Prometheus
    ├── Grafana
    └── Alertmanager
```

Supporting AWS resources are created with Terraform. The EKS cluster itself is created with `eksctl`.

## Prerequisites

Install and configure:

- AWS CLI
- Docker
- kubectl
- eksctl
- Helm
- Terraform
- make
- Python virtual environment at `.venv`

Required configuration:

- AWS authentication
- `.env`
- Groq API key
- AirLabs API key

Default region:

```text
us-east-2
```

> **Warning:** AWS resources in this runbook incur charges. Always run the audit after teardown.

## 1. Verify AWS access

```bash
aws sts get-caller-identity
```

Check available EKS versions:

```bash
aws eks describe-cluster-versions --region us-east-2 --output table
```

The current cluster configuration pins EKS `1.35`.

## 2. Create supporting AWS resources

```bash
cd deploy/terraform
cp terraform.tfvars.example terraform.tfvars
```

Set:

```hcl
bucket_suffix = "your-unique-suffix"
alert_email   = "you@example.com"
```

Then:

```bash
terraform init
terraform apply
```

Terraform creates supporting resources including:

- S3 artifacts/backup bucket
- ECR repository
- backup IAM policy
- SNS alert topic
- Alertmanager publish policy
- AWS budget
- email subscriptions

Record the outputs:

```bash
terraform output
```

### Confirm SNS email

Confirm the AWS subscription email before relying on monitoring alerts.

```bash
aws sns list-subscriptions-by-topic \
  --region us-east-2 \
  --topic-arn "$(terraform output -raw alerts_topic_arn)"
```

The subscription should not remain `PendingConfirmation`.

> **Note:** Terraform state is local in this project. Keep `terraform.tfstate` safe and out of Git.

## 3. Build and push the application image

From the repository root:

```bash
make ecr-push
```

Use the immutable tag printed by the command.

Replace deployment placeholders in the deployment copy:

```bash
TAG=<immutable-tag>

sed -i "s/REPLACE_WITH_IMMUTABLE_TAG/$TAG/" \
  deploy/k8s/20-api.yaml \
  deploy/k8s/50-index-job.yaml

sed -i "s/ACCOUNT_ID/$(aws sts get-caller-identity --query Account --output text)/g" \
  deploy/k8s/*.yaml \
  deploy/helm/*.yaml

sed -i "s/REPLACE_WITH_S3_BUCKET/$(terraform -chdir=deploy/terraform output -raw s3_bucket)/" \
  deploy/k8s/60-db-backup.yaml
```

> **Important:** Make placeholder replacements in a deployment copy, not in the clean repository.

## 4. Create the EKS cluster

```bash
make eks-up
```

Cluster configuration:

```text
deploy/k8s/eksctl-cluster.yaml
```

Verify:

```bash
kubectl get nodes

kubectl get pods -n kube-system \
  -l app.kubernetes.io/name=aws-ebs-csi-driver

kubectl get pods -n kube-system \
  -l k8s-app=aws-node
```

## 5. Install the AWS Load Balancer Controller

Install it before creating the public API Service:

```bash
make eks-lb-controller
```

Verify:

```bash
kubectl -n kube-system get deployment aws-load-balancer-controller
```

The controller provisions the Network Load Balancer and participates in the API pod readiness gate.

## 6. Create application secrets

Ensure the namespace exists:

```bash
kubectl create namespace fdr --dry-run=client -o yaml | kubectl apply -f -
```

Generate a URI-safe PostgreSQL password:

```bash
PGPASS=$(openssl rand -hex 24)
```

Create the secret:

```bash
kubectl -n fdr create secret generic fdr-secrets \
  --from-literal=postgres-password="$PGPASS" \
  --from-literal=pg-dsn="postgresql://fdr:${PGPASS}@postgres-0.postgres.fdr.svc.cluster.local:5432/fdr" \
  --from-literal=llm-api-key='<Groq key>' \
  --from-literal=airlabs-key='<AirLabs key>' \
  --from-literal=conversation-secret="$(openssl rand -hex 32)"

unset PGPASS
```

The `conversation-secret` must be shared across replicas so conversation tokens remain valid after load balancing and restarts.

## 7. Run preflight checks

```bash
test -f .env && echo ".env present"

.venv/Scripts/python.exe scripts/k8s_preflight.py --require-env
```

Continue only when it reports:

```text
preflight OK
```

The preflight checks deployment relationships including secrets, ConfigMap ordering, `.env` parity, image consistency, StorageClasses, NetworkPolicies, IRSA, load balancer settings, and unresolved placeholders.

## 8. Deploy workloads

```bash
make eks-deploy PYTHON=.venv/Scripts/python.exe
```

Deployment order:

```text
namespace
→ StorageClass
→ NetworkPolicy
→ PostgreSQL
→ ConfigMap
→ index Job
→ API Service
→ TargetGroupBinding wait
→ API Deployment
→ backup CronJob
```

The Service is created before the API Deployment so the readiness gate can be attached before API pods start.

## 9. Verify the deployment

```bash
kubectl -n fdr get pods,svc,pdb,networkpolicy
```

Get the NLB hostname:

```bash
HOST=$(kubectl -n fdr get svc api \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
```

Check:

```bash
curl http://$HOST/health
curl http://$HOST/ready
```

DNS and target registration may take a few minutes.

## 10. Monitoring

Install:

```bash
make eks-monitoring \
  GRAFANA_PASSWORD='<choose-a-password>' \
  PYTHON=.venv/Scripts/python.exe
```

The deployment uses the `kube-prometheus-stack` Helm chart.

Application monitoring resources:

```text
deploy/k8s/40-monitoring.yaml
deploy/k8s/45-alert-rules.yaml
```

Source alert rules:

```text
deploy/prometheus/rules/rag.yml
```

After changing them:

```bash
.venv/Scripts/python.exe scripts/make_prometheusrule.py
```

Verify:

```bash
kubectl -n monitoring get pods
kubectl -n fdr get servicemonitor,prometheusrule
```

Prometheus:

```bash
kubectl -n monitoring port-forward \
  svc/kube-prometheus-stack-prometheus 9090:9090
```

Grafana:

```bash
kubectl -n monitoring port-forward \
  svc/kube-prometheus-stack-grafana 3000:80
```

Alert path:

```text
Prometheus
→ Alertmanager
→ Amazon SNS
→ email
```

Alertmanager uses IRSA; no SMTP/Gmail password is required.

## 11. Optional CloudWatch application logs

```bash
make eks-logs
make eks-logs-retention
```

Application logs are written under:

```text
/aws/containerinsights/fdr/application
```

This is optional because CloudWatch ingestion and Container Insights add cost.

## 12. Backups

The backup CronJob writes nightly `pg_dump` files under:

```text
s3://<bucket>/db-backups/
```

Run manually:

```bash
make db-backup
```

Verify:

```bash
aws s3 ls s3://<bucket>/db-backups/
```

### Restore

```bash
kubectl -n fdr scale deployment api --replicas=0

aws s3 cp s3://<bucket>/db-backups/<file>.dump - \
  | kubectl -n fdr exec -i postgres-0 -- \
      pg_restore --clean --if-exists --no-owner -U fdr -d fdr

kubectl -n fdr scale deployment api --replicas=2
```

> **Note:** The current Postgres deployment is single-instance. Backups provide recovery, not high availability.

## 13. HTTPS

The repository defaults to HTTP for demo deployment.

For a public deployment:

1. Use a domain you control.
2. Request an ACM certificate in `us-east-2`.
3. Add the certificate ARN to `deploy/k8s/19-api-service.yaml`.
4. Enable the HTTPS Service port.
5. Point DNS at the NLB.
6. Remove the HTTP port after HTTPS is verified.

## 14. Acceptance checks

After the first deployment, verify properties static checks cannot prove.

### Readiness gate

```bash
kubectl -n fdr get targetgroupbindings

kubectl -n fdr get pods -l app=api \
  -o jsonpath='{range .items[*]}{.metadata.name}{"  "}{.spec.readinessGates[*].conditionType}{"\n"}{end}'
```

### Load balancer

Confirm the NLB is internet-facing, uses IP targets, checks `/ready`, and has healthy targets.

### NetworkPolicy

Confirm an unapproved pod cannot reach Postgres while an approved workload can.

### IAM

Confirm:

- API has no AWS role
- index Job has no AWS role
- backup workload assumes `fdr-backup`
- backup IAM can write only to the intended S3 prefix

### Backup restore

Create a backup, restore it into a scratch database, and compare row counts.

### Alert delivery

Send a test Alertmanager alert and verify the SNS email arrives.

## 15. Optional self-hosted vLLM profile

The default deployment does not use vLLM.

Optional manifest:

```text
deploy/k8s/30-vllm.yaml
```

To enable it:

```bash
make eks-deploy-vllm
```

This requires the optional GPU node group and enough VRAM for the selected model and configured context window.

## 16. Teardown

```bash
make eks-down
```

Teardown order:

1. Run a final backup.
2. Delete LoadBalancer Services while the controller still exists.
3. Remove StatefulSet/PVC resources while the CSI driver still exists.
4. Delete the EKS cluster.

Then:

```bash
make audit
```

Expected verdict:

```text
VERDICT: CLEAR
```

Run the audit again later because AWS billing data can lag.

## 17. Cost protection

Useful commands:

```bash
make gpu-sleep
make eks-down
make audit
```

Treat `UNKNOWN` from the audit as unresolved, not as a clean account.

## 18. Known limitations

Current accepted limitations:

- HTTP until TLS/domain configuration is added
- single-instance PostgreSQL
- no automatic cluster or API autoscaling
- local Terraform state
- unrestricted application egress
- optional vLLM path not part of the default deployment
