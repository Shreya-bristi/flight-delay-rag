#!/usr/bin/env bash

# Audit AWS resources that may still incur charges.
#
# Exit codes:
#   0 = clear
#   1 = billable resources found
#   2 = one or more checks could not be verified
set -uo pipefail
REGION="${AWS_REGION:-us-east-2}"
FOUND=0
UNKNOWN=0

echo "=== cost audit: $REGION @ $(date -u +%FT%TZ) ==="

# stop if AWS credentials cannot be verified
if ! IDENT=$(aws sts get-caller-identity --query 'Account' --output text 2>&1); then
  echo "FATAL: cannot call AWS. This audit proves NOTHING about your bill."
  echo "       $IDENT"
  exit 2
fi
echo "account: $IDENT"

# Run an AWS check and distinguish CLEAR, FOUND, and UNKNOWN
check() {
  local title="$1" note="$2"; shift 2
  local out status
  out=$(aws "$@" --output text 2>&1); status=$?
  echo
  echo "--- $title  ($note)"
  if [ $status -ne 0 ]; then
    UNKNOWN=$((UNKNOWN + 1))
    echo "    UNKNOWN: the query failed. This is NOT 'nothing is running'."
    echo "    ${out//$'\n'/$'\n'    }"
  elif [ -z "${out//[[:space:]]/}" ] || [ "$out" = "None" ]; then
    echo "    CLEAR: none."
  else
    FOUND=$((FOUND + 1))
    echo "    FOUND - still billing:"
    echo "    ${out//$'\n'/$'\n'    }"
  fi
}

check "Running EC2 instances" "g4dn.xlarge = \$0.526/hr = \$12.60/day" \
  ec2 describe-instances --region "$REGION" \
  --filters "Name=instance-state-name,Values=running,pending" \
  --query 'Reservations[].Instances[].[InstanceId,InstanceType,LaunchTime]'

check "EKS clusters" "\$0.10/hr EACH, billed even when idle" \
  eks list-clusters --region "$REGION" --query 'clusters'

check "Load balancers (v2)" "~\$0.023/hr each; delete k8s Services BEFORE the cluster or these orphan" \
  elbv2 describe-load-balancers --region "$REGION" \
  --query 'LoadBalancers[].[LoadBalancerName,Type]'

check "Load balancers (classic)" "created by a LoadBalancer Service on older setups" \
  elb describe-load-balancers --region "$REGION" \
  --query 'LoadBalancerDescriptions[].LoadBalancerName'

check "UNATTACHED EBS volumes" "\$0.08/GB-month FOREVER; survives instance termination" \
  ec2 describe-volumes --region "$REGION" \
  --filters Name=status,Values=available \
  --query 'Volumes[].[VolumeId,Size,CreateTime]'

check "NAT gateways" "\$32.85/MONTH each at zero traffic - there should be ZERO" \
  ec2 describe-nat-gateways --region "$REGION" \
  --filter Name=state,Values=available \
  --query 'NatGateways[].[NatGatewayId,State]'

check "Unassociated Elastic IPs" "\$0.005/hr while sitting idle" \
  ec2 describe-addresses --region "$REGION" \
  --query 'Addresses[?AssociationId==null].[PublicIp]'

# check a second region for accidentally created resources
OTHER=$([ "$REGION" = "us-east-2" ] && echo "us-east-1" || echo "us-east-2")
check "Cross-check: instances in $OTHER" "should be CLEAR - you only use $REGION" \
  ec2 describe-instances --region "$OTHER" \
  --filters "Name=instance-state-name,Values=running,pending" \
  --query 'Reservations[].Instances[].[InstanceId,InstanceType]'

echo
echo "--- Month-to-date spend"
END=$(date -u -d '+1 day' +%F 2>/dev/null || date -u -v+1d +%F)
if COST=$(aws ce get-cost-and-usage \
      --time-period Start="$(date -u +%Y-%m-01)",End="$END" \
      --granularity MONTHLY --metrics UnblendedCost \
      --query 'ResultsByTime[0].Total.UnblendedCost.[Amount,Unit]' \
      --output text 2>&1); then
  echo "    $COST"
else
  UNKNOWN=$((UNKNOWN + 1))
  echo "    UNKNOWN: Cost Explorer query failed (it must be enabled once in the"
  echo "    Billing console, and it lags several hours behind reality anyway)."
  echo "    ${COST//$'\n'/$'\n'    }"
fi

echo
if [ "$UNKNOWN" -gt 0 ]; then
  echo "VERDICT: UNKNOWN - $UNKNOWN check(s) could not be answered. Fix those and re-run;"
  echo "         do not read the empty sections as 'nothing is running'."
  exit 2
elif [ "$FOUND" -gt 0 ]; then
  echo "VERDICT: $FOUND section(s) found resources that are still billing. See above."
  exit 1
fi
echo "VERDICT: CLEAR - every check answered, and none found billable compute in $REGION."
echo "         Billing data lags several hours: re-run tomorrow morning."
exit 0
