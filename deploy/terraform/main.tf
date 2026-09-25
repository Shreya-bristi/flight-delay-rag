# AWS resources for the flight-delay RAG deployment
# EKS cluster is managed seperately with eksctl

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  
  #Local state is sufficient for this project, for shared envs, migrate to S3 backend

  # backend "s3" {
  #   bucket       = "fdr-tfstate-<your suffix>"
  #   key          = "flight-delay-rag/terraform.tfstate"
  #   region       = "us-east-2"
  #   encrypt      = true
  #   use_lockfile = true
  # }
}

provider "aws" {
  region = var.region
  default_tags {
    # Apply project tags to all terrraform-managed resources

    tags = {
      Project   = "flight-delay-rag"
      ManagedBy = "terraform"
      Ephemeral = "true"
    }
  }
}

data "aws_caller_identity" "current" {}

variable "region" {
  type    = string
  default = "us-east-2"
}

# required values supplied through terraform.tfvars or interaactively
variable "bucket_suffix" {
  description = "S3 bucket names are globally unique. Put something personal here (lowercase letters, digits, hyphens)."
  type        = string

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9-]{2,40}[a-z0-9]$", var.bucket_suffix)) && var.bucket_suffix != "changeme"
    error_message = "bucket_suffix must be 4-42 lowercase letters, digits or hyphens, and not the old placeholder \"changeme\"."
  }
}

# ---------------------------------------------------------------------------
# S3 - evaluation artifacts and database backups
# ---------------------------------------------------------------------------
resource "aws_s3_bucket" "artifacts" {
  bucket = "fdr-artifacts-${var.bucket_suffix}"
  # protect evaluation artifacts and backups from accidental Terraform destruction
  force_destroy = false

  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration {
    # preserve artifact versions after accidental overwrites
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# explicitly enforce server-side encryption for artifacts and backups
resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    id     = "expire-old-versions"
    status = "Enabled"
    filter {}
    noncurrent_version_expiration {
      noncurrent_days = 7
    }
  }
  # database backups expire after 14 days and evaluation artifacts are retrained
  rule {
    id     = "expire-db-backups"
    status = "Enabled"
    filter {
      prefix = "db-backups/"
    }
    expiration {
      days = 14
    }
  }
}

# ---------------------------------------------------------------------------
# ECR - container images
# ---------------------------------------------------------------------------
resource "aws_ecr_repository" "api" {
  name         = "fdr-api"
  force_delete = true
  # immutable tags make deployment and rollbacks reproducible
  image_tag_mutability = "IMMUTABLE"
  image_scanning_configuration {
    # Free vulnerability scanning on push. One line, and it is a real security
    # control you can point at.
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name
  # remove untagged images after one day
  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "expire untagged images after 1 day"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 1
        }
        action = { type = "expire" }
      },
      {
        rulePriority = 2
        description  = "keep only the 10 most recent tagged images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 10
        }
        action = { type = "expire" }
      },
    ]
  })
}

# ---------------------------------------------------------------------------
# IAM policy for IRSA - database backup
#
# the backup job may only write objects under db-backups/
# ---------------------------------------------------------------------------
resource "aws_iam_policy" "db_backup" {
  name        = "fdr-db-backup"
  description = "Write new Postgres dumps under db-backups/ in the artifacts bucket; nothing else"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:PutObject"]
        Resource = ["${aws_s3_bucket.artifacts.arn}/db-backups/*"]
      }
    ]
  })
}

# ---------------------------------------------------------------------------
# Alert delivery: Prometheus -> Alertmanager -> SNS -> email
# Alertmanager publishes to SNS using IRSA; no SMTP credentials are stored.
# the SNS email subscription must be confirmed before alerts are delivered
# ---------------------------------------------------------------------------
resource "aws_sns_topic" "alerts" {
  name = "fdr-alerts"
}

resource "aws_sns_topic_subscription" "alerts_email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# # Allow Alertmanager to publish only to the project alert topic
resource "aws_iam_policy" "alerts_publish" {
  name        = "fdr-alerts-publish"
  description = "Publish Alertmanager notifications to the fdr-alerts SNS topic only"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["sns:Publish"]
        Resource = [aws_sns_topic.alerts.arn]
      }
    ]
  })
}

# ---------------------------------------------------------------------------
# Budget alarm 
# ---------------------------------------------------------------------------
resource "aws_budgets_budget" "monthly" {
  name         = "fdr-monthly"
  budget_type  = "COST"
  limit_amount = "40"
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  # forecast cost warning
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 60
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.alert_email]
  }
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 90
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.alert_email]
  }
}

variable "alert_email" {
  description = "Where the budget alarms AND the Alertmanager alerts (via SNS) go. Confirm the SNS subscription email AWS sends, or no alert is delivered."
  type        = string

  validation {
    condition     = can(regex("^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$", var.alert_email)) && !endswith(var.alert_email, "@example.com")
    error_message = "alert_email must be a real address you read (not the old you@example.com placeholder)."
  }
}

# ---------------------------------------------------------------------------
output "s3_bucket" { value = aws_s3_bucket.artifacts.id }
output "ecr_repo_url" { value = aws_ecr_repository.api.repository_url }
output "backup_policy_arn" { value = aws_iam_policy.db_backup.arn }
output "alerts_topic_arn" { value = aws_sns_topic.alerts.arn }
output "alerts_policy_arn" { value = aws_iam_policy.alerts_publish.arn }
output "account_id" { value = data.aws_caller_identity.current.account_id }

output "next_steps" {
  value = <<-EOT

    1. Put the account id and the bucket into the manifests:
         sed -i 's/ACCOUNT_ID/${data.aws_caller_identity.current.account_id}/g' ../k8s/*.yaml ../helm/*.yaml
         sed -i 's/REPLACE_WITH_S3_BUCKET/${aws_s3_bucket.artifacts.id}/' ../k8s/60-db-backup.yaml

    2. Build and push the image under an IMMUTABLE tag, then put that tag into
       20-api.yaml and 50-index-job.yaml (they ship REPLACE_WITH_IMMUTABLE_TAG
       on purpose - :latest makes a rollout unreproducible and `rollout undo`
       useless):
         TAG=$(date -u +%Y%m%dT%H%M%S)
         aws ecr get-login-password --region ${var.region} | \
           docker login --username AWS --password-stdin ${aws_ecr_repository.api.repository_url}
         docker build -t ${aws_ecr_repository.api.repository_url}:$TAG ../..
         docker push ${aws_ecr_repository.api.repository_url}:$TAG

    3. Create the cluster (~20 min), after checking the version and the addons:
         eksctl create cluster -f ../k8s/eksctl-cluster.yaml

    4. Create the SECRETS the pods read (none of them are in the image or in
       this state file). The API refuses to start without the generator key,
       and conversation-secret MUST be one value shared by every replica:
         kubectl create namespace fdr
         kubectl -n fdr create secret generic fdr-secrets \
           --from-literal=postgres-password='<choose one>' \
           --from-literal=pg-dsn='postgresql://fdr:<same password>@postgres-0.postgres.fdr.svc.cluster.local:5432/fdr' \
           --from-literal=llm-api-key='<Groq key>' \
           --from-literal=airlabs-key='<AirLabs key>' \
           --from-literal=conversation-secret="$(openssl rand -hex 32)"

    5. Check, then deploy:  python scripts/k8s_preflight.py  ->  make eks-deploy

    See ../../RUNBOOK_AWS.md for the ordered procedure and the teardown.
  EOT
}
