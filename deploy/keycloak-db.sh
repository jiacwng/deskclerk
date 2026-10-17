# Gives Keycloak a role and a database of its own in the same server, on the first start of an
# empty data volume.
set -eu

psql -v ON_ERROR_STOP=1 -v password="$KC_DB_PASSWORD" \
	--username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<'SQL'
CREATE ROLE keycloak LOGIN PASSWORD :'password';
CREATE DATABASE keycloak OWNER keycloak;
SQL
