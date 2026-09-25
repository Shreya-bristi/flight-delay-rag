SHELL := /bin/bash
export PYTHONPATH := src

# Overrides on Windows if python not on PATH
PYTHON ?= python

.DEFAULT_GOAL := help

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	 awk 'BEGIN{FS=":.*?## "};{printf "  \033[36m%-18s\033[0m %s\n",$$1,$$2}'

install:  ## install core dependencies from uv.lock
	uv sync --frozen

install-ml:  ## install real embedder and reranker from uv.lock 
	uv sync --frozen --extra ml

up:  ## start postgres + api + prometheus + grafana 
	docker compose up -d --build
	@echo "API      http://localhost:$${API_PORT:-8000}"
	@echo "Grafana  http://localhost:3000"
	@echo "Prom     http://localhost:9090"
	@echo
	@echo "The API serves 503 on /ready until the index exists: run 'make index-container'."
	@echo "Port 8000 already in use (scripts/run_local.ps1 on the host)? API_PORT=8001 make up"

index-container:  ## build the index INSIDE API container
	docker compose run --rm --no-deps -T api python scripts/index_corpus.py

prometheus-rule:  ## regenerate kubernetes alert rules
	$(PYTHON) scripts/make_prometheusrule.py

down:  ## stops everything
	docker compose down

clean:  ## stops everything and delete the database volume
	docker compose down -v

db:  ## starts postgres and wait until its healthcheck 
	docker compose up -d --wait postgres
	@echo "postgres ready on 127.0.0.1:5432"

index:  ## parse corpus, embed docs and load into pgvector
	$(PYTHON) scripts/index_corpus.py

index-hash:  ## build a smoke-test index using the hash embedder
	ALLOW_TEST_DOUBLES=true EMBEDDER=hash $(PYTHON) scripts/index_corpus.py

run:  ## starts the API on the host after running make db
	$(PYTHON) -m uvicorn flight_delay.api:app --reload --port 8000

golden:  ## rebuild golden eval set
	$(PYTHON) evals/build_golden_set.py

golden-check:  ## validate the golden set without writing it
	$(PYTHON) evals/build_golden_set.py --check

eval-retrieval:  ## run retreival evaluation
	$(PYTHON) evals/retrieval_eval.py

eval-calibrate:  ## run routing and confidence gate caliberation
	$(PYTHON) evals/generation_eval.py --generator none

eval-generation:  ## run generation evaluation and llm-as -a judge
	$(PYTHON) evals/generation_eval.py $(GENERATORS)

eval:  ## both evaluation stages in order
	$(PYTHON) evals/retrieval_eval.py
	$(PYTHON) evals/generation_eval.py $(GENERATORS)

test:  ## run the test suite
	$(PYTHON) -m pytest -q

lint:  ## run ruff lint and format check
	$(PYTHON) -m ruff check . && $(PYTHON) -m ruff format --check .

smoke:  ## run smoke test without API call
	$(MAKE) index-hash
	$(PYTHON) evals/retrieval_eval.py --backend memory --embedder hash --reranker none --sizes 256,512 --limit 10
	$(PYTHON) evals/generation_eval.py --backend memory --embedder hash --reranker none --generator echo --judge fake --limit 10 --allow-smoke-selection --retrieval-result evals/runs/retrieval/latest-smoke.json
	$(PYTHON) -m pytest -q
	@echo "SMOKE PASSED (stand-in embedder, generator and judge: not measurements)"

# ------------------------------------------------------------------ AWS
tf-apply:  ## create S3 ,ECR, IAM and budget alarm
	cd deploy/terraform && terraform init && terraform apply

ecr-push:  ## build and push an immutable timestamp-tagged API image
	ECR="$$ACCOUNT.dkr.ecr.us-east-2.amazonaws.com/fdr-api"; \
	TAG=$${TAG:-$$(date -u +%Y%m%dT%H%M%S)}; \
	aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin $$ECR; \
	docker build -t $$ECR:$$TAG . && docker push $$ECR:$$TAG; \
	echo; echo "PUSHED $$ECR:$$TAG"; \
	echo "now put $$TAG in deploy/k8s/20-api.yaml and deploy/k8s/50-index-job.yaml"; \
	echo "(the ECR repo is IMMUTABLE: re-pushing an existing tag fails by design)"

eks-up:  ## create EKS cluster 
	eksctl create cluster -f deploy/k8s/eksctl-cluster.yaml

preflight:  ## validate deployment files and required local configuration
	# --require-env: on the machine that deploys, a missing .env must FAIL -
	# otherwise the fdr-config <-> .env parity check is silently skipped.
	$(PYTHON) scripts/k8s_preflight.py --target deploy --require-env



LBC_CHART_VERSION ?= 3.5.0 # AWS load balancer controller helm chart
KPS_CHART_VERSION ?= 91.4.1 # kube-prometheus stackhelm chart

eks-lb-controller:  ## install the AWS Load Balancer Controller 
	helm repo add eks https://aws.github.io/eks-charts
	helm repo update eks
	@VPC=$$(aws eks describe-cluster --name fdr --region us-east-2 --query cluster.resourcesVpcConfig.vpcId --output text); \
	helm upgrade --install aws-load-balancer-controller eks/aws-load-balancer-controller \
	  --version $(LBC_CHART_VERSION) \
	  -n kube-system \
	  --set clusterName=fdr \
	  --set serviceAccount.create=false \
	  --set serviceAccount.name=aws-load-balancer-controller \
	  --set region=us-east-2 \
	  --set vpcId=$$VPC
	kubectl -n kube-system rollout status deployment/aws-load-balancer-controller --timeout=300s

eks-deploy: preflight  ## deploy default hosted-generator stack
	@echo ">> the fdr-secrets Secret must exist first (see RUNBOOK_AWS.md)"
	@kubectl -n kube-system get deployment aws-load-balancer-controller >/dev/null 2>&1 || \
	  { echo "the AWS Load Balancer Controller is not installed: run make eks-lb-controller first"; exit 1; }
	@kubectl -n fdr get secret fdr-secrets -o jsonpath='{.data.conversation-secret}' 2>/dev/null | grep -q . || \
	  { echo "fdr-secrets is missing or has no conversation-secret key (RUNBOOK_AWS.md section 4)"; exit 1; }

	kubectl apply -f deploy/k8s/00-namespace.yaml
	kubectl apply -f deploy/k8s/05-storageclass.yaml
	kubectl apply -f deploy/k8s/15-networkpolicy.yaml
	kubectl apply -f deploy/k8s/10-postgres.yaml
	kubectl rollout status statefulset/postgres -n fdr --timeout=300s

	# Config must exist before the index Job starts
	kubectl apply -f deploy/k8s/18-config.yaml
	kubectl apply -f deploy/k8s/50-index-job.yaml
	kubectl wait --for=condition=complete job/index-corpus -n fdr --timeout=2700s

	# Asks how many AirLabs calls are left , BEFORE the API pods start
	bash scripts/set_airlabs_quota.sh
	
	# Create the Service before API pods so the load-balancer readiness gate exists
	kubectl apply -f deploy/k8s/19-api-service.yaml
	@for i in $$(seq 1 60); do \
	  kubectl -n fdr get targetgroupbindings -o name 2>/dev/null | grep -q . && { echo "TargetGroupBinding present"; exit 0; }; \
	  sleep 5; \
	done; \
	echo "no TargetGroupBinding after 5 min: kubectl -n kube-system logs deploy/aws-load-balancer-controller"; exit 1
	kubectl apply -f deploy/k8s/20-api.yaml
	kubectl rollout status deployment/api -n fdr --timeout=900s
	kubectl apply -f deploy/k8s/60-db-backup.yaml
	@echo "30-vllm.yaml is NOT applied: the generator is hosted. See eks-deploy-vllm."
	@echo "public address: kubectl -n fdr get svc api -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'"

eks-monitoring:  ## install prometheus, grafana, dsahboards and alert rules
	@test -n "$(GRAFANA_PASSWORD)" || { echo "usage: make eks-monitoring GRAFANA_PASSWORD=<choose one>"; exit 1; }
	$(PYTHON) scripts/k8s_preflight.py --target monitoring
	@kubectl -n monitoring get serviceaccount fdr-alertmanager -o jsonpath='{.metadata.annotations.eks\.amazonaws\.com/role-arn}' 2>/dev/null | grep -q . || \
	  { echo "no IRSA ServiceAccount monitoring/fdr-alertmanager: it comes from eksctl-cluster.yaml (make eks-up) - alerts could not reach SNS"; exit 1; }
	helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
	helm repo update prometheus-community
	helm upgrade --install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
	  --version $(KPS_CHART_VERSION) \
	  --namespace monitoring --create-namespace \
	  -f deploy/helm/kube-prometheus-stack-values.yaml \
	  --set grafana.adminPassword='$(GRAFANA_PASSWORD)'

	# load grfana dashboard from configmaps
	kubectl -n monitoring create configmap fdr-grafana-dashboard \
	  --from-file=rag.json=deploy/grafana/dashboards/rag.json --dry-run=client -o yaml \
	  | kubectl label --local -f - grafana_dashboard=1 -o yaml \
	  | kubectl apply --server-side -f -
	
	#cluster only dashboard, seperate from local docker compose dashboards
	kubectl -n monitoring create configmap fdr-grafana-dashboard-overview \
	  --from-file=overview.json=deploy/grafana/dashboards-cluster/overview.json --dry-run=client -o yaml \
	  | kubectl label --local -f - grafana_dashboard=1 -o yaml \
	  | kubectl apply --server-side -f -
	kubectl apply -f deploy/k8s/40-monitoring.yaml
	kubectl apply -f deploy/k8s/45-alert-rules.yaml
	@echo "verify: kubectl -n fdr get servicemonitor,prometheusrule, then the /targets and /alerts pages"

eks-logs:  ## enable opt in cloudwatch conatiner login
	@ACCOUNT=$$(aws sts get-caller-identity --query Account --output text); \
	eksctl create iamserviceaccount --cluster fdr --region us-east-2 \
	  --namespace amazon-cloudwatch --name cloudwatch-agent \
	  --role-name fdr-cloudwatch-agent --role-only \
	  --attach-policy-arn arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy --approve; \
	aws eks create-addon --cluster-name fdr --region us-east-2 \
	  --addon-name amazon-cloudwatch-observability \
	  --service-account-role-arn arn:aws:iam::$$ACCOUNT:role/fdr-cloudwatch-agent
	@echo "log groups appear once Fluent Bit starts; then cap their retention: make eks-logs-retention"

eks-logs-retention:  ## set conatainer insights log retention to one day
	@for g in $$(aws logs describe-log-groups --region us-east-2 \
	    --log-group-name-prefix /aws/containerinsights/fdr/ --query 'logGroups[].logGroupName' --output text); do \
	  aws logs put-retention-policy --region us-east-2 --log-group-name $$g --retention-in-days 1 && echo "1 day: $$g"; \
	done

airlabs-quota:  ## set how many AirLabs calls are left this month 

db-backup:  ## run the Postgres backup immediately and wait for completion
	kubectl -n fdr delete job db-backup-manual --ignore-not-found
	kubectl -n fdr create job db-backup-manual --from=cronjob/db-backup
	kubectl -n fdr wait --for=condition=complete job/db-backup-manual --timeout=900s
	kubectl -n fdr logs job/db-backup-manual -c upload

eks-deploy-vllm:  ## deploy the self hosted VLLM generator  which I wil explore later
	@echo ">> also switch LLM_PROVIDER/LLM_BASE_URL/LLM_MODEL in deploy/k8s/18-config.yaml"
	kubectl apply -f deploy/k8s/30-vllm.yaml

artifacts-push:  ## upload local evaluation run to S3
	aws s3 sync evals/runs "s3://$(S3_BUCKET)/evals/" --exclude '*/latest*.json'

artifacts-pull:  ## fetch eval runs from the S3  into evals/runs
	aws s3 sync "s3://$(S3_BUCKET)/evals/" evals/runs

gpu-sleep:  ## scale GPU nodegroup to 0 
	eksctl scale nodegroup --cluster fdr --name gpu --nodes 0

gpu-wake:  ## scale GPU nodegroup to 1
	eksctl scale nodegroup --cluster fdr --name gpu --nodes 1

audit:  ## list everything in AWS currently billing
	bash scripts/aws_audit.sh

eks-down:  ## back up Postgres and delete the EKS cluster
	# 1. Backup before destroying cluster storage
	@if [ "$(SKIP_BACKUP)" = "1" ]; then echo "SKIP_BACKUP=1: no final backup"; \
	 else $(MAKE) db-backup || { echo "BACKUP FAILED - stopping. Fix it, or re-run with SKIP_BACKUP=1 to lose the data."; exit 1; }; fi

	# 2. delete Services while the LB controller is running so NLBs are cleaned up
	-kubectl delete svc --all -n fdr --wait=true --timeout=300s

	# 3. delete PVCs before the cluster so EBS volumes are removed
	-kubectl delete statefulset --all -n fdr --wait=true
	-kubectl delete pvc --all -n fdr --wait=true --timeout=300s
	eksctl delete cluster -f deploy/k8s/eksctl-cluster.yaml --wait
	@echo "now run: make audit   (and read its VERDICT line - an UNKNOWN is not a CLEAR)"

.PHONY: help install install-ml up down clean db index index-hash index-container run \
        prometheus-rule golden golden-check eval-retrieval eval-calibrate eval-generation \
        eval test lint smoke tf-apply ecr-push eks-up preflight eks-lb-controller eks-deploy \
        eks-monitoring eks-logs eks-logs-retention airlabs-quota db-backup eks-deploy-vllm \
        artifacts-push artifacts-pull gpu-sleep gpu-wake audit eks-down
