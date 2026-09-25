#!/usr/bin/env bash

# initialize the AirLabs monthly quota counter from the current dashboard balance.
# This keeps the app's quota tracking and Grafana metric aligned with the real account balance.

# need to run before API pods start so the initial quota metric is correct

set -euo pipefail

QUOTA=${AIRLABS_MONTHLY_QUOTA:-1000}   # the app's default
PERIOD=$(date -u +%Y-%m)              # the app counts per UTC calendar month
LEFT=${AIRLABS_LEFT:-}

if [ -z "$LEFT" ]; then
  read -r -p "AirLabs calls left this month, from the AirLabs dashboard (0-$QUOTA, Enter = skip): " LEFT
fi
if [ -z "$LEFT" ]; then
  echo "skipped: the AirLabs counter is unchanged"
  exit 0
fi
if ! [[ "$LEFT" =~ ^[0-9]+$ ]] || [ "$LEFT" -gt "$QUOTA" ]; then
  echo "not a whole number between 0 and $QUOTA: $LEFT" >&2
  exit 1
fi

USED=$((QUOTA - LEFT))
kubectl -n fdr exec -i postgres-0 -- psql -q -v ON_ERROR_STOP=1 -U fdr -d fdr <<SQL
CREATE TABLE IF NOT EXISTS airlabs_quota (period TEXT PRIMARY KEY, used INTEGER NOT NULL);
INSERT INTO airlabs_quota (period, used) VALUES ('$PERIOD', $USED)
  ON CONFLICT (period) DO UPDATE SET used = EXCLUDED.used;
SQL
echo "recorded for $PERIOD: $LEFT of $QUOTA left ($USED used)"
