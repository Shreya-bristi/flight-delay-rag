<#
.SYNOPSIS
    Audit AWS resources that may still incur charges.

.DESCRIPTION
    Reports each check as CLEAR, FOUND, or UNKNOWN.

    Exit codes:
      0 = clear
      1 = billable resources found
      2 = one or more checks could not be verified

.EXAMPLE
    .\scripts\aws_audit.ps1
#>

$ErrorActionPreference = "Continue"
$region = if ($env:AWS_REGION) { $env:AWS_REGION } else { "us-east-2" }
$script:Found = 0
$script:Unknown = 0

Write-Host "=== AWS cost audit : $region : $(Get-Date -Format 'yyyy-MM-dd HH:mm')" -ForegroundColor Yellow

# stop immediately if AWS credentials are unavailable
$who = aws sts get-caller-identity --query "Account" --output text 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "FATAL: cannot call AWS. This audit proves NOTHING about your bill." -ForegroundColor Red
    Write-Host "       $who" -ForegroundColor Red
    exit 2
}
Write-Host "account: $who" -ForegroundColor DarkGray

function Invoke-Check {
    param(
        [string]$Title,
        [string]$Note,
        [string[]]$AwsArgs
    )
    Write-Host ""
    Write-Host "--- $Title" -ForegroundColor Cyan
    if ($Note) { Write-Host "    $Note" -ForegroundColor DarkGray }

    # distinguish an empty successful query from a failed AWS CLI call
    $out = & aws @AwsArgs --output text 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0) {
        $script:Unknown++
        Write-Host "    UNKNOWN: the query failed. This is NOT 'nothing is running'." -ForegroundColor Magenta
        Write-Host ("    " + $out.Trim()) -ForegroundColor Magenta
    }
    elseif ([string]::IsNullOrWhiteSpace($out) -or $out.Trim() -eq "None") {
        Write-Host "    CLEAR: none." -ForegroundColor Green
    }
    else {
        $script:Found++
        Write-Host "    FOUND - still billing:" -ForegroundColor Red
        Write-Host ("    " + $out.Trim()) -ForegroundColor Red
    }
}

Invoke-Check "Running EC2 instances" "g4dn.xlarge = `$0.526/hr = `$12.60/day" @(
    "ec2", "describe-instances", "--region", $region,
    "--filters", "Name=instance-state-name,Values=running,pending",
    "--query", "Reservations[].Instances[].[InstanceId,InstanceType,LaunchTime]")

Invoke-Check "EKS clusters" "`$0.10/hr EACH, billed even when idle" @(
    "eks", "list-clusters", "--region", $region, "--query", "clusters")

Invoke-Check "Load balancers (v2)" "~`$0.023/hr each. Delete k8s Services BEFORE the cluster or these orphan." @(
    "elbv2", "describe-load-balancers", "--region", $region,
    "--query", "LoadBalancers[].[LoadBalancerName,Type]")

Invoke-Check "Load balancers (classic)" "created by a LoadBalancer Service on older setups" @(
    "elb", "describe-load-balancers", "--region", $region,
    "--query", "LoadBalancerDescriptions[].LoadBalancerName")

Invoke-Check "UNATTACHED EBS volumes" "`$0.08/GB-month FOREVER. Survives instance termination. Easy to miss." @(
    "ec2", "describe-volumes", "--region", $region,
    "--filters", "Name=status,Values=available",
    "--query", "Volumes[].[VolumeId,Size,CreateTime]")

Invoke-Check "NAT gateways" "`$32.85/MONTH each at zero traffic. There should be ZERO." @(
    "ec2", "describe-nat-gateways", "--region", $region,
    "--filter", "Name=state,Values=available",
    "--query", "NatGateways[].[NatGatewayId,State]")

Invoke-Check "Unassociated Elastic IPs" "`$0.005/hr while sitting idle" @(
    "ec2", "describe-addresses", "--region", $region,
    "--query", "Addresses[?AssociationId==null].[PublicIp]")

# check a second region for accidentally created resources.
$other = if ($region -eq "us-east-2") { "us-east-1" } else { "us-east-2" }
Invoke-Check "Cross-check: instances in $other" "should be CLEAR - you are only using $region" @(
    "ec2", "describe-instances", "--region", $other,
    "--filters", "Name=instance-state-name,Values=running,pending",
    "--query", "Reservations[].Instances[].[InstanceId,InstanceType]")

Write-Host ""
Write-Host "--- Month-to-date spend" -ForegroundColor Cyan
$start = Get-Date -Format "yyyy-MM-01"
$end = (Get-Date).AddDays(1).ToString("yyyy-MM-dd")
$cost = & aws ce get-cost-and-usage `
    --time-period "Start=$start,End=$end" `
    --granularity MONTHLY --metrics UnblendedCost `
    --query "ResultsByTime[0].Total.UnblendedCost.[Amount,Unit]" `
    --output text 2>&1 | Out-String
if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($cost)) {
    Write-Host ("    " + $cost.Trim())
}
else {
    $script:Unknown++
    Write-Host "    UNKNOWN: Cost Explorer query failed. It must be enabled once in the" -ForegroundColor Magenta
    Write-Host "    Billing console, and it lags several hours behind reality anyway." -ForegroundColor Magenta
    Write-Host ("    " + $cost.Trim()) -ForegroundColor Magenta
}

Write-Host ""
if ($script:Unknown -gt 0) {
    Write-Host "VERDICT: UNKNOWN - $($script:Unknown) check(s) could not be answered." -ForegroundColor Magenta
    Write-Host "         Fix those and re-run. Do not read an empty section as 'nothing is running'." -ForegroundColor Magenta
    exit 2
}
if ($script:Found -gt 0) {
    Write-Host "VERDICT: $($script:Found) section(s) found resources that are still billing. See above." -ForegroundColor Red
    exit 1
}
Write-Host "VERDICT: CLEAR - every check answered, and none found billable compute in $region." -ForegroundColor Green
Write-Host "         Billing data lags several hours: re-run tomorrow morning." -ForegroundColor DarkGray
exit 0
