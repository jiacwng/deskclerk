#!/usr/bin/env bash
# Starts compose.prod.yaml from an empty checkout on the domain localhost, where Caddy signs its
# own certificate, writes two Enron mailboxes as a folder of files with an access file, sets up a
# second database as a company's own with one table whose policy gives each mailbox's group its
# own row, and runs e2e/test_prod.py against the stack. Run from a checkout without a .env:
# bash e2e/run_prod.sh

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
# The company database is on the same server, where needtoknow_app has one password.
app_password=$(sed -n 's/^NEEDTOKNOW_APP_PASSWORD=//p' .env)
sed -i \
    -e 's/^# NEEDTOKNOW_DATA_HOST=.*/NEEDTOKNOW_DATA_HOST=db/' \
    -e 's/^# NEEDTOKNOW_DATA_NAME=.*/NEEDTOKNOW_DATA_NAME=company/' \
    -e "s/^# NEEDTOKNOW_DATA_PASSWORD=.*/NEEDTOKNOW_DATA_PASSWORD=${app_password}/" .env
set -a
. ./.env
set +a

mkdir -p documents/files

started=$(date +%s)
$compose up -d --wait db
psql="$compose exec -T db psql -v ON_ERROR_STOP=1 -U needtoknow"
$psql -d needtoknow -c "CREATE DATABASE company"
$compose run --rm load data-setup
$psql -d company <<'SQL'
CREATE TABLE sales (team TEXT NOT NULL, amount INTEGER NOT NULL);
INSERT INTO sales VALUES ('group:dasovich-j', 100), ('group:kaminski-v', 200);
ALTER TABLE sales ENABLE ROW LEVEL SECURITY;
CREATE POLICY sales_read ON sales FOR SELECT TO needtoknow_app
USING (team = ANY (string_to_array(current_setting('app.principals', true), ',')));
GRANT SELECT ON sales TO needtoknow_app;
SQL
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
