#!/usr/bin/env bash
# Starts compose.prod.yaml from an empty checkout on the domain localhost, where Caddy signs its
# own certificate, writes two Enron mailboxes as a folder of files with an access file, and runs
# e2e/test_prod.py against the stack. Run from a checkout without a .env: bash e2e/run_prod.sh

set -euo pipefail
cd "$(dirname "$0")/.."

if [ -e .env ] || [ -e documents ]; then
    echo "run this from a checkout without .env and documents/" >&2
    exit 1
fi

export COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-needtoknow-prod}"
compose="docker compose -f compose.prod.yaml"

cp .env.example .env
sed -i \
    -e 's/^NEEDTOKNOW_DOMAIN=.*/NEEDTOKNOW_DOMAIN=localhost/' \
    -e 's/^KC_BOOTSTRAP_ADMIN_USERNAME=.*/KC_BOOTSTRAP_ADMIN_USERNAME=admin/' .env
for name in NEEDTOKNOW_ADMIN_PASSWORD NEEDTOKNOW_OWNER_PASSWORD NEEDTOKNOW_APP_PASSWORD \
    KC_DB_PASSWORD KC_BOOTSTRAP_ADMIN_PASSWORD; do
    sed -i "s/^${name}=.*/${name}=$(openssl rand -hex 16)/" .env
done
set -a
. ./.env
set +a

mkdir -p documents/files

started=$(date +%s)
$compose up -d --wait
echo "stack up in $(($(date +%s) - started)) seconds"

certificates=$(mktemp -d)
$compose cp caddy:/data/caddy/pki/authorities/local/root.crt "$certificates/root.crt"
timeout 60 bash -c "until curl -sf --cacert $certificates/root.crt https://localhost/health; do sleep 2; done"

mailbox=tests/data/enron/maildir
for custodian in dasovich-j kaminski-v; do
    mkdir "documents/files/$custodian"
    find "$mailbox/$custodian" -type f | while read -r path; do
        name=$(echo "${path#"$mailbox/$custodian/"}" | tr / -)
        cp "$path" "documents/files/$custodian/$name.txt"
    done
    printf '{"path": "%s/*", "groups": ["%s"]}\n' "$custodian" "$custodian" >> documents/access.jsonl
done

E2E_PROD_DIR="$PWD" E2E_CA="$certificates/root.crt" "${PYTHON:-python}" -m pytest e2e/test_prod.py
