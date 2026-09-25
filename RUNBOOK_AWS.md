# AWS Deployment Runbook

**Executed once, end to end, on 2026-09-21** (PROGRESS.md Session 37): steps 1-6 and
the 8b acceptance checks except 4c and 5, then torn down (`make audit` CLEAR). That
run found four problems no offline check could, all fixed here since: Free-plan
accounts refuse `t3.medium` (step 3), the index Job had no LLM key (step 5), Grafana
was OOMKilled at 384Mi (step 6), and `KubeCPUOvercommit` fires on two nodes
(section 9). Nothing is running now. Deploy from a fresh COPY of the repository -
the placeholder `sed` edits in step 2 must not be made in the repository itself.

`make` targets call `python`, which is not on PATH on this Windows machine. Pass
`PYTHON=.venv/Scripts/python.exe` to make, or run the commands by hand.

What changed on 2026-09-20 (PROGRESS.md Session 33): a shared
`conversation-secret`, EKS 1.35 instead of an extended-support 1.31, the AWS Load
Balancer Controller (so the Service really is an NLB, with client IPs preserved),
an explicit encrypted gp3 StorageClass, NetworkPolicies, a nightly Postgres
backup, alert email via Amazon SNS (no mail password), the dashboard as a ConfigMap, and a preflight check.

---

## 0. Before you spend anything

| check | command | why |
|---|---|---|
| Identity | `aws sts get-caller-identity` | everything below silently targets the wrong account otherwise |
| EKS version | `aws eks describe-cluster-versions --region us-east-2 --output table` | the file pins **1.35** (standard until 2027-03-26). A version in extended support bills the control plane at **$0.60/hr instead of $0.10** |
| G-quota (only for the optional GPU) | Service Quotas -> EC2 -> "Running On-Demand G and VT instances" >= 4 | a new account is often 0; approval takes hours to days |
| Budget | step 1 creates it | create the budget *before* the cluster, not after |

**The default deployment needs no GPU**: the generator is hosted (Groq).

## 1. Data-plane resources (cents/month)

```bash
cd deploy/terraform
cp terraform.tfvars.example terraform.tfvars   # set bucket_suffix and alert_email (git-ignored)
terraform init && terraform apply              # = make tf-apply
```

Neither variable has a default any more: `changeme` gave a globally-unique
bucket name somebody else may own, and `you@example.com` sent the budget alarm
to nobody. Both are validated. `alert_email` is used twice: the budget alarm
emails it directly, and the SNS topic `fdr-alerts` (Prometheus alerts) has an
email subscription to it.

**Click the link in "AWS Notification - Subscription Confirmation"** that
arrives at `alert_email` right after `apply` (check spam). Until you do, the
SNS subscription stays `PendingConfirmation` and **no monitoring alert is ever
delivered**. Check it:
`aws sns list-subscriptions-by-topic --region us-east-2 --topic-arn $(terraform output -raw alerts_topic_arn)`
- the `SubscriptionArn` must be a real ARN, not `PendingConfirmation`.

Creates S3 (artifacts + database backups, versioned, encrypted, `db-backups/`
expiring after 14 days), ECR (images), two IRSA IAM policies (`fdr-db-backup`:
PutObject under `db-backups/`; `fdr-alerts-publish`: sns:Publish on
`fdr-alerts`), the SNS topic + email subscription, and a $40 monthly budget
with alerts at 60% forecast and 90% actual. Note `terraform output`: you need
`account_id` and `s3_bucket`.

The bucket is `force_destroy = false` with `prevent_destroy`, so `terraform
destroy` **fails** rather than deleting eval runs. To really remove it: empty it
yourself, then drop the `lifecycle` block.

**State is local** (`terraform.tfstate`, git-ignored) - fine for one person and
four cheap resources. For a team, `main.tf` has a commented `backend "s3"` block
and the one-time steps.

**Nothing uploads eval runs to S3 automatically**; push them deliberately:

```bash
make artifacts-push S3_BUCKET=$(terraform -chdir=deploy/terraform output -raw s3_bucket)
```

## 2. Image and placeholders

```bash
make ecr-push            # builds, pushes, prints an immutable timestamp tag
```

Replace the markers the repository ships on purpose:

```bash
TAG=<the tag ecr-push printed>
sed -i "s/REPLACE_WITH_IMMUTABLE_TAG/$TAG/" deploy/k8s/20-api.yaml deploy/k8s/50-index-job.yaml
sed -i "s/ACCOUNT_ID/$(aws sts get-caller-identity --query Account --output text)/g" deploy/k8s/*.yaml deploy/helm/*.yaml
sed -i "s/REPLACE_WITH_S3_BUCKET/$(terraform -chdir=deploy/terraform output -raw s3_bucket)/" deploy/k8s/60-db-backup.yaml
```

The ECR repository is **IMMUTABLE**: re-pushing a tag fails by design, so "the
image we tested" and "the image running" cannot drift apart.

## 3. Cluster (~20 min, $0.10/hr control plane + 2 x m7i-flex.large ~$0.19/hr)

```bash
eksctl create cluster -f deploy/k8s/eksctl-cluster.yaml    # = make eks-up
```

It creates, besides the node group: OIDC, the `fdr-backup` IRSA account (the
ONLY application workload with AWS permissions: `s3:PutObject` on `db-backups/*`),
the `fdr-alertmanager` IRSA account in `monitoring` (`sns:Publish` on `fdr-alerts`
only), the `aws-load-balancer-controller` IRSA account (AWS's controller policy), the EBS
CSI driver, and the VPC CNI **with network policy enforcement on**. Verify
before going on:

```bash
kubectl get pods -n kube-system -l app.kubernetes.io/name=aws-ebs-csi-driver   # Running
kubectl get pods -n kube-system -l k8s-app=aws-node                           # 2/2: policy agent present
```

**If the node group fails** (first real deploy, Session 37: `AsgInstanceLaunchFailures`,
"not eligible for Free Tier"), the control plane stays up and eksctl stops BEFORE it
writes your kubeconfig and before it creates the add-ons, so `kubectl` talks to
`localhost:8080` and the EBS CSI driver is missing. Do not re-run `make eks-up` (the
cluster exists). Fix the cause, then:

```bash
aws cloudformation describe-stack-events --region us-east-2 --stack-name eksctl-fdr-nodegroup-cpu \
  --query "StackEvents[?ResourceStatus=='CREATE_FAILED'].[LogicalResourceId,ResourceStatusReason]" --output table
eksctl delete nodegroup --cluster fdr --region us-east-2 --name cpu --wait
eksctl create nodegroup -f deploy/k8s/eksctl-cluster.yaml --include=cpu
aws eks update-kubeconfig --region us-east-2 --name fdr
eksctl get addon --cluster fdr --region us-east-2              # aws-ebs-csi-driver must be listed
eksctl create addon -f deploy/k8s/eksctl-cluster.yaml          # creates only what is missing
eksctl create iamserviceaccount -f deploy/k8s/eksctl-cluster.yaml --approve
```

Then the two checks above.

Then install the load balancer controller - **before** the API Service exists:

```bash
make eks-lb-controller
kubectl -n kube-system get deployment aws-load-balancer-controller   # 2/2
```

Without it, EKS's legacy in-tree controller turns `type: LoadBalancer` into a
**Classic** Load Balancer. With it, `19-api-service.yaml`'s annotations give an NLB with
`ip` targets, client IP preservation (so the per-client rate limit sees
passengers, not the NLB) and a health check on `/ready`. `TRUST_FORWARDED_FOR`
stays **false**: an NLB sets no `X-Forwarded-For`, and trusting the header
would let clients pick their own rate-limit key.

## 4. Secrets

None of these are in the image, in Git, or in Terraform state.

```bash
# The namespace may ALREADY EXIST: eksctl creates `fdr` when it creates the
# fdr-backup IRSA ServiceAccount in step 3. This form succeeds either way
# (a bare `kubectl create namespace fdr` fails with AlreadyExists).
kubectl create namespace fdr --dry-run=client -o yaml | kubectl apply -f -

# Hex is URI-safe: 48 random hex characters (192 bits) can go into the DSN as-is.
PGPASS=$(openssl rand -hex 24)
kubectl -n fdr create secret generic fdr-secrets \
  --from-literal=postgres-password="$PGPASS" \
  --from-literal=pg-dsn="postgresql://fdr:${PGPASS}@postgres-0.postgres.fdr.svc.cluster.local:5432/fdr" \
  --from-literal=llm-api-key='<Groq key>' \
  --from-literal=airlabs-key='<AirLabs key>' \
  --from-literal=conversation-secret="$(openssl rand -hex 32)"
unset PGPASS
```

- **The namespace's labels are NOT set here.** This command only guarantees the
  namespace exists. `deploy/k8s/00-namespace.yaml`, which `make eks-deploy`
  applies first, adds the `elbv2.k8s.aws/pod-readiness-gate-inject` label the
  load balancer controller needs - on a namespace that already exists, `apply`
  adds it. Do not skip it because the namespace is already there.
- **Password in a URI.** `pg-dsn` is a URI, so characters such as `@ : / ? # %`
  (and spaces) in the password break it - `@` ends the user-info part, `/`
  starts the path - and the API and backup job cannot connect. Either generate a
  strong URI-safe password as above (recommended for the first deployment), or
  keep your own password raw in `postgres-password` and **percent-encode** it
  only inside `pg-dsn`:
  `.venv/Scripts/python.exe -c "import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=''))" '<password>'`
  (libpq decodes it). Never weaken the password to avoid encoding.
- The API **refuses to start** with `LLM_PROVIDER=groq` and no key.
- **`conversation-secret` is not optional with 2 replicas.** It signs
  conversation tokens. Without it each pod invents its own random key, so a
  conversation started on pod A gets **403** from pod B, and every restart
  orphans every open conversation. Rotating it ends every open conversation
  (clients start a new one); that is the only cost.

## 5. Workloads

**First, prove the cluster configuration equals `.env`.** `deploy/k8s/18-config.yaml`
(the `fdr-config` ConfigMap) is the cluster's copy of `.env`, and `.env` is never
in Git or in an archive. A preflight run in a copy without `.env` prints
`WARNING: .env not found; .env <-> fdr-config parity check skipped.` - that
`preflight OK` proves nothing about parity. So, on the machine you deploy from:

```bash
test -f .env && echo ".env present"                       # the REAL repository, not an archive copy
.venv/Scripts/python.exe scripts/k8s_preflight.py --require-env
```

It must print `preflight OK` with **no WARNING line**. A mismatch names the keys
(never the values); fix `18-config.yaml` or `.env` until they agree.
`make eks-deploy` repeats this with `--require-env`, so a missing `.env` stops it.

```bash
make eks-deploy PYTHON=.venv/Scripts/python.exe
```

It first runs `scripts/k8s_preflight.py` and stops on any problem: a marker still
in a file, a missing `conversation-secret`, vLLM's window too small, a claim on
an undefined StorageClass, a Service not handed to the controller, a pod reading
`PG_DSN` that the Postgres NetworkPolicy would block, a workload other than the
backup running with an AWS role (or mounting a token it never uses), a workload
applied before the ConfigMap it reads, a missing `.env`, or a ConfigMap value
that disagrees with `.env`. It also checks the controller and the Secret exist.

Then, in order: namespace labels (`00-namespace.yaml`), StorageClass,
NetworkPolicies, Postgres (wait), **`fdr-config`** (`18-config.yaml`), the
**index Job** (wait, ~23 min), the API **Service** (`19-api-service.yaml`), a
wait for the controller's TargetGroupBinding, the API Deployment (wait), the
backup CronJob.

Why `fdr-config` is its own early file: the index Job reads it (`envFrom`), and
the Job runs to completion before the API exists. While the ConfigMap lived in
`20-api.yaml`, applied after the Job, a new cluster's index pod stopped at
`CreateContainerConfigError: configmap "fdr-config" not found`.

Why the Service goes first, on its own: the controller adds its readiness gate
(`target-health.elbv2.k8s.aws/...`) only to pods created **after** the
TargetGroupBinding exists, and creates that binding asynchronously. Applied
together, the first pods usually won the race and ran without the gate.

**AWS identities:** the API, the index Job, Postgres and vLLM run with no IAM
role and no ServiceAccount token (`automountServiceAccountToken: false`); none
of them calls an AWS or Kubernetes API. Only the backup job (`fdr-backup`) has
a role, and it can only write new objects under `db-backups/`.

- The index Job is **safe to re-run** (staging table, validated swap). Re-run with
  `kubectl delete job index-corpus -n fdr --ignore-not-found && kubectl apply -f deploy/k8s/50-index-job.yaml`
  - but see the capacity table: with the API and monitoring running there is no
  room for it on two nodes.
- API pods download ~2.5 GB of model weights on first start; a `startupProbe`
  with a 10-minute budget keeps the liveness probe from killing them mid-download.
- `/ready` returns 503 until the index matches the running settings; the NLB
  health check asks `/ready` too, and the readiness gate keeps a pod out of the
  rollout count until its NLB target is healthy.
- Rollouts replace **one pod at a time with no surge pod** (a third 1536Mi pod
  does not fit); a PodDisruptionBudget keeps one replica through node drains;
  the two replicas are required to be on different nodes.

Check it:

```bash
kubectl -n fdr get pods,svc,pdb,networkpolicy
HOST=$(kubectl -n fdr get svc api -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
curl http://$HOST/health && curl http://$HOST/ready      # DNS can take a few minutes
```

## 5b. HTTPS (required before real passengers use it)

The default Service listens on plain HTTP :80. For anything beyond a demo:

1. A domain in Route 53 (or anywhere you can add a CNAME).
2. `aws acm request-certificate --region us-east-2 --domain-name chat.<domain> --validation-method DNS`,
   add the validation record, wait for `ISSUED`.
3. In `deploy/k8s/19-api-service.yaml` (the Service - it is no longer in the
   API's Deployment file), uncomment the `aws-load-balancer-ssl-cert` (put the
   certificate ARN in) and `aws-load-balancer-ssl-ports` annotations and the
   `https` port, then `kubectl apply -f deploy/k8s/19-api-service.yaml`. The NLB
   terminates TLS and forwards to :8000 inside the VPC.
4. Point `chat.<domain>` at the NLB: a Route 53 **alias** A record (or a CNAME
   to the `$HOST` above).
5. Once HTTPS works, remove the `http` port so nothing is served in clear text.

Not done in the repository because it needs a domain you own.

## 6. Monitoring

```bash
kubectl -n monitoring get serviceaccount fdr-alertmanager   # made by eksctl in step 3 (IRSA)
make eks-monitoring GRAFANA_PASSWORD='<choose one>' PYTHON=.venv/Scripts/python.exe
```

No mail password, no Secret to create: the `monitoring` namespace and the
`fdr-alertmanager` ServiceAccount already exist (eksctl made them), and the
alert address is the `alert_email` you set in `terraform.tfvars`.

`deploy/helm/kube-prometheus-stack-values.yaml` now holds everything:

- **Alerts reach a person, through SNS.**
  `Prometheus -> Alertmanager -> SNS topic fdr-alerts -> email to alert_email`.
  Alertmanager publishes warning/critical alerts (and their resolution) with
  its IRSA role (`sns:Publish` on that topic only); `Watchdog`,
  `InfoInhibitor` and `info` alerts go nowhere. The emails come **from AWS
  Notifications** (`no-reply@sns.amazonaws.com`), subject like
  `[FIRING] fdr: HighLatency`. Nothing is delivered until the SNS subscription
  is confirmed (step 1).
- **EKS-unscrapable components are off** (scheduler, controller manager, etcd,
  kube-proxy); left on, they fire `...Down` alerts forever on a healthy cluster.
- **Resource requests are explicit** (~1 GiB in total; see section 9).
- **The dashboard is deployed**: `rag.json` becomes a ConfigMap labelled
  `grafana_dashboard=1`, which Grafana's sidecar loads. No manual import, and no
  second copy of the JSON to drift.

Verify:

```bash
kubectl -n monitoring get pods
kubectl -n fdr get servicemonitor,prometheusrule
kubectl -n monitoring port-forward svc/kube-prometheus-stack-prometheus 9090:9090
#  /targets -> fdr-api UP     /alerts -> the fdr rules listed
kubectl -n monitoring port-forward svc/kube-prometheus-stack-grafana 3000:80
#  the "Flight Delay RAG" dashboard is present
```

Edit alert thresholds in `deploy/prometheus/rules/rag.yml` and re-run
`.venv/Scripts/python.exe scripts/make_prometheusrule.py`.

## 7. Application logs in CloudWatch (opt-in)

`cloudWatch.clusterLogging` in the cluster file ships **control-plane** logs
only. The API's own stdout lives in `kubectl logs` and dies with the pod. To
centralise it:

```bash
make eks-logs              # IRSA role + amazon-cloudwatch-observability add-on
make eks-logs-retention    # after the groups appear: 1-day retention (default is NEVER expire)
```

Logs land in `/aws/containerinsights/fdr/application` (Logs Insights:
`fields @timestamp, log | filter kubernetes.namespace_name = "fdr"`).
**Opt-in because it costs money**: the add-on also enables Container Insights
metrics and Application Signals, billed per observation/request, on top of log
ingestion. Check Billing a day after turning it on. `eksctl delete cluster`
removes the add-on; the log groups expire on their own with 1-day retention.

## 8. Backups and restore

`60-db-backup.yaml` runs `pg_dump` nightly (03:00 UTC) to
`s3://<bucket>/db-backups/fdr-<timestamp>.dump`, with the database's own image
(client = server version) and the `fdr-backup` IRSA role - PutObject under
`db-backups/` only, so it can write a dump but not read, list or delete one (no AWS key in the
cluster). Run one now:

```bash
make db-backup
aws s3 ls s3://<bucket>/db-backups/
```

Restore (into a fresh or damaged database; the API is scaled down so nothing
writes meanwhile):

```bash
kubectl -n fdr scale deployment api --replicas=0
aws s3 cp s3://<bucket>/db-backups/<file>.dump - \
  | kubectl -n fdr exec -i postgres-0 -- pg_restore --clean --if-exists --no-owner -U fdr -d fdr
kubectl -n fdr scale deployment api --replicas=2
```

Postgres is still **one replica on one EBS volume in one AZ**: the backup bounds
the loss to a day, it does not make the database highly available. For real
passengers use RDS for PostgreSQL (pgvector supported, automated backups,
Multi-AZ); only `pg-dsn` changes.

## 8b. Acceptance checks (the first real deployment)

Everything above was validated **offline**. These are the properties no offline
check can prove. Run each once after the first deploy; record the outcome in
PROGRESS.md. A check that has never been run is an assumption, not a property.

```bash
HOST=$(kubectl -n fdr get svc api -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
BUCKET=$(terraform -chdir=deploy/terraform output -raw s3_bucket)
```

**1. Readiness gate on the API pods** - expect a `target-health.elbv2.k8s.aws/...` condition on every pod:

```bash
kubectl -n fdr get targetgroupbindings
kubectl -n fdr get pods -l app=api -o jsonpath='{range .items[*]}{.metadata.name}{"  "}{.spec.readinessGates[*].conditionType}{"\n"}{end}'
```

**2. The load balancer is what the manifest says** - expect `network internet-facing`,
then `ip HTTP /ready`, then `healthy healthy`, then `true`:

```bash
LB=$(aws elbv2 describe-load-balancers --region us-east-2 --query "LoadBalancers[?DNSName=='$HOST'].LoadBalancerArn" --output text)
aws elbv2 describe-load-balancers --region us-east-2 --load-balancer-arns $LB --query 'LoadBalancers[].[Type,Scheme]' --output text
TG=$(aws elbv2 describe-target-groups --region us-east-2 --load-balancer-arn $LB --query 'TargetGroups[0].TargetGroupArn' --output text)
aws elbv2 describe-target-groups --region us-east-2 --target-group-arns $TG --query 'TargetGroups[].[TargetType,HealthCheckProtocol,HealthCheckPath]' --output text
aws elbv2 describe-target-health --region us-east-2 --target-group-arn $TG --query 'TargetHealthDescriptions[].TargetHealth.State' --output text
aws elbv2 describe-target-group-attributes --region us-east-2 --target-group-arn $TG --query "Attributes[?Key=='preserve_client_ip.enabled'].Value" --output text
```

Then `curl http://$HOST/health` and check the API's access log shows **your
public address** (`curl -s https://checkip.amazonaws.com`), not a `10.x`
address - that is what the per-client rate limit keys on.

**3. NetworkPolicy is enforced** - an unlisted pod must NOT reach Postgres
(expect `no response`), an allowed label must (expect `accepting connections`):

```bash
kubectl -n fdr run np-deny --rm -i --restart=Never --labels=app=np-test \
  --image=pgvector/pgvector:0.8.1-pg16 -- pg_isready -h postgres-0.postgres.fdr.svc.cluster.local -t 5
kubectl -n fdr run np-allow --rm -i --restart=Never --labels=app=db-backup \
  --image=pgvector/pgvector:0.8.1-pg16 -- pg_isready -h postgres-0.postgres.fdr.svc.cluster.local -t 5
```

If the first one connects, enforcement is off: check `aws-node` pods are 2/2
and the vpc-cni add-on carries `enableNetworkPolicy`.

**4. AWS identities are as narrow as intended.**

The API and the index Job carry **no AWS credentials** - expect `0`, then
`|false` (no ServiceAccount, no token), then `0` or no output if the Job's
pod is already gone (`ttlSecondsAfterFinished` 1 h):

```bash
kubectl -n fdr exec deploy/api -- sh -c 'env | grep -c "^AWS_"'
kubectl -n fdr get job index-corpus -o jsonpath='{.spec.template.spec.serviceAccountName}{"|"}{.spec.template.spec.automountServiceAccountToken}{"\n"}'
kubectl -n fdr get pods -l job-name=index-corpus -o jsonpath='{.items[*].spec.containers[*].env[*].name}' | tr ' ' '\n' | grep -c '^AWS_'
```

The backup identity **assumes the fdr-backup role** - the assumed-role name in
the first line must be the role named in the second:

```bash
kubectl -n fdr run iam-who --rm -i --restart=Never --image=amazon/aws-cli:2.36.49 \
  --overrides='{"spec":{"serviceAccountName":"fdr-backup"}}' -- sts get-caller-identity --query Arn --output text
kubectl -n fdr get serviceaccount fdr-backup -o jsonpath='{.metadata.annotations.eks\.amazonaws\.com/role-arn}{"\n"}'
```

...and it can **only write under `db-backups/`** - expect `AccessDenied` for the
list, success for the first upload, `AccessDenied` for the second:

```bash
BACKUP_POD='{"spec":{"serviceAccountName":"fdr-backup"}}'
kubectl -n fdr run iam-list --rm -i --restart=Never --image=amazon/aws-cli:2.36.49 \
  --overrides="$BACKUP_POD" -- s3 ls "s3://$BUCKET/db-backups/"
kubectl -n fdr run iam-put-in --rm -i --restart=Never --image=amazon/aws-cli:2.36.49 \
  --overrides="$BACKUP_POD" --command -- sh -c "echo acceptance | aws s3 cp - s3://$BUCKET/db-backups/acceptance-test.txt"
kubectl -n fdr run iam-put-out --rm -i --restart=Never --image=amazon/aws-cli:2.36.49 \
  --overrides="$BACKUP_POD" --command -- sh -c "echo acceptance | aws s3 cp - s3://$BUCKET/evals/acceptance-test.txt"
aws s3 rm "s3://$BUCKET/db-backups/acceptance-test.txt"     # clean up with YOUR credentials
```

**5. A backup exists and restores** - the object is listed in S3 (with your
credentials: the backup role itself cannot list), then restored into a scratch
database; the two counts must match:

```bash
make db-backup
F=$(aws s3 ls "s3://$BUCKET/db-backups/" | sort | tail -1 | awk '{print $4}')
kubectl -n fdr exec postgres-0 -- createdb -U fdr restore_check
aws s3 cp "s3://$BUCKET/db-backups/$F" - | kubectl -n fdr exec -i postgres-0 -- pg_restore --no-owner -U fdr -d restore_check
for db in fdr restore_check; do kubectl -n fdr exec postgres-0 -- psql -U fdr -d $db -tAc "select count(*) from chunks"; done
kubectl -n fdr exec postgres-0 -- dropdb -U fdr restore_check
```

**6. An alert reaches a person, through SNS** - first check Alertmanager has
its AWS identity (expect an `AWS_ROLE_ARN` line), then send a test alert and
expect an email from AWS Notifications within about a minute (`group_wait` 30s)
and a "RESOLVED" one about five minutes later:

```bash
kubectl -n monitoring exec alertmanager-kube-prometheus-stack-alertmanager-0 -c alertmanager -- env | grep AWS_ROLE_ARN
kubectl -n monitoring exec alertmanager-kube-prometheus-stack-alertmanager-0 -c alertmanager -- \
  amtool alert add AcceptanceTest severity=warning --annotation=summary="acceptance test - ignore" \
  --alertmanager.url=http://localhost:9093
```

No email? In order: the subscription is still `PendingConfirmation` (step 1);
`kubectl -n monitoring logs alertmanager-kube-prometheus-stack-alertmanager-0 -c alertmanager | grep -i sns`
shows `AccessDenied` (role/policy) or a wrong topic ARN (ACCOUNT_ID not replaced);
the email is in spam.

**7. Teardown leaves nothing billing** - after `make eks-down`, `make audit`
must say CLEAR: no load balancer, no unattached EBS volume, no instances.

## 9. Capacity (two m7i-flex.large, ~6.9 GiB and 1.93 CPU allocatable each)

| pod | memory request | where |
|---|---|---|
| api x2 | 1536Mi each | one per node (required) |
| postgres | 512Mi | either |
| prometheus | 512Mi | either |
| grafana + operator + alertmanager + kube-state-metrics | ~510Mi | either |
| node-exporter, coredns, CSI, LB controller, aws-node | ~300Mi total | both |
| **total** | **~4.9 GiB of ~13.9 GiB** | |

Memory is not the limit. The nodes were `t3.medium` (4 GiB) until the first real
deploy (Session 37): an AWS account on the **Free plan** refuses any instance type
outside its free-tier-eligible list, and `m7i-flex.large` is on it. Measured on
that deploy: CPU **requests** are 2.31 of 3.86 allocatable, so the chart's own
`KubeCPUOvercommit` warning fires (and emails you) - it means "one node could
not hold everything if the other died", which a two-node demo accepts. The
index Job (1 CPU, 2Gi) fits beside the running stack by memory, but not by CPU
request on one node: if it stays Pending, scale the API to 1 replica for the
re-index, or `eksctl scale nodegroup --cluster fdr --name cpu --nodes 3`.

**No autoscaling, deliberately.** `maxSize: 3` is headroom for that manual
scale, not elasticity: there is no Cluster Autoscaler/Karpenter and no
HorizontalPodAutoscaler. Adding elasticity means metrics-server + an HPA on the
API, and Karpenter or the Cluster Autoscaler (with its own IRSA role) to add the
node the third replica needs. On a credit-funded demo an autoscaler's failure
mode is an unexpected bill, so it is left out.

## 10. Optional: self-hosted generator (only if you want a GPU)

Not part of the deployment. It needs the GPU node group (commented out in
`eksctl-cluster.yaml`; eksctl picks the accelerated AMI and **installs the NVIDIA
device plugin itself** - verify the node reports `nvidia.com/gpu: 1`, do not
install a second copy), a pinned vLLM image, and `--max-model-len >= LLM_CONTEXT_WINDOW +
LLM_MAX_COMPLETION_TOKENS` (now 11216 >= 10312 + 900; the preflight enforces it).
A 16 GB T4 cannot hold the hosted candidate, so this means a different, smaller
model: the generation quality Stage 2 measured does not carry over.

```bash
make eks-deploy-vllm   # after switching LLM_* in deploy/k8s/18-config.yaml
```

## 11. COST PROTECTION

```bash
make gpu-sleep   # scale the GPU nodegroup to 0 - only exists if you created it
make eks-down    # final backup, delete Services (NLB) and PVCs (EBS), then the cluster
make audit       # what is still billing; READ THE VERDICT LINE
```

`eks-down` order matters and is now enforced:

1. `make db-backup` - it **stops** if the backup fails (`SKIP_BACKUP=1` to
   accept losing the data).
2. Services, waited on, while the load balancer controller still runs: it is
   what deletes the NLB. Delete the cluster first and the NLB is orphaned and
   keeps billing.
3. StatefulSet and PVCs, while the EBS CSI driver still runs: otherwise the
   database volume outlives the cluster, unattached and billed.
4. The cluster.

`make audit` distinguishes three outcomes and exits accordingly:

| verdict | exit | meaning |
|---|---|---|
| CLEAR | 0 | every check answered and found nothing billable |
| FOUND | 1 | something is still running - it is listed |
| **UNKNOWN** | 2 | a query failed (credentials, permission, region). **Not** a clean bill |

**Run `make audit` again the next morning.** Billing data lags several hours.

## Expected spend

| item | rate |
|---|---|
| EKS control plane | $0.10/hr (standard support; 1.35) |
| 2 x m7i-flex.large | ~$0.19/hr (list price - confirm) |
| NLB | ~$0.023/hr + LCU |
| EBS: 10 GiB gp3 (Postgres) | ~$0.001/hr |
| CloudWatch logs + Container Insights, **opt-in** | usage-based - check Billing |
| 1 x g4dn.xlarge, **optional** | $0.526/hr |
| S3 + ECR + budget | cents/month |
| SNS alert emails | free (first 1,000 emails a month) |

~$0.32/hr without the GPU, ~$0.82/hr with it. A week left running is more than
the credit. The price is not the risk; forgetting is. Set a calendar alarm.

## Still not production (known, accepted for the demo)

HTTPS needs your domain (5b); Postgres is single-instance (8); no autoscaling
(9); API **egress** is unrestricted (NetworkPolicies match IPs, and Groq,
AirLabs and Hugging Face are DNS names on shifting ranges - an egress proxy or
FQDN-aware policy engine would be needed); Terraform state is local (1).
