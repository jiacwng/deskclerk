-- Tables, grants and row-level security policies for the documents. Run by needtoknow_owner;
-- running it again drops the tables and starts empty. A change here raises
-- schema.SCHEMA_VERSION.

DROP TABLE IF EXISTS schema_info, chunks, doc_access, documents;
DROP FUNCTION IF EXISTS set_identity(TEXT);

-- The only way the app and reader roles can set who is asking: provision.prepare_database takes
-- set_config away from them. It works once per transaction, so SQL that runs after db.as_user
-- cannot call it again to become someone else.
CREATE FUNCTION set_identity(principals TEXT) RETURNS VOID
LANGUAGE plpgsql VOLATILE SECURITY DEFINER SET search_path = pg_catalog, public
AS $$
BEGIN
    IF current_setting('app.identity_set', true) = 'on' THEN
        RAISE EXCEPTION 'the identity is already set in this transaction'
            USING ERRCODE = 'insufficient_privilege';
    END IF;
    PERFORM set_config('app.principals', principals, true);
    PERFORM set_config('app.identity_set', 'on', true);
END
$$;

REVOKE EXECUTE ON FUNCTION set_identity(TEXT) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION set_identity(TEXT) TO needtoknow_app, needtoknow_reader;

CREATE TABLE documents (
    id    TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    body  TEXT NOT NULL
);

CREATE TABLE doc_access (
    doc_id    TEXT NOT NULL REFERENCES documents (id),
    principal TEXT NOT NULL,
    PRIMARY KEY (doc_id, principal)
);

CREATE INDEX doc_access_principal_doc_id ON doc_access (principal, doc_id);

CREATE TABLE chunks (
    id        BIGSERIAL PRIMARY KEY,
    doc_id    TEXT NOT NULL REFERENCES documents (id),
    position  INTEGER NOT NULL,
    text      TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL
);

CREATE INDEX chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);

-- One row: the schema version and the embedding model and chunk size of the stored chunks.
-- reindexed_through is the last document a running or stopped reindex has rebuilt.
CREATE TABLE schema_info (
    id                BOOLEAN PRIMARY KEY DEFAULT true CHECK (id),
    version           INTEGER NOT NULL,
    embedding_model   TEXT NOT NULL,
    chunk_chars       INTEGER NOT NULL,
    reindexed_through TEXT
);

GRANT SELECT ON documents, doc_access, chunks TO needtoknow_app, needtoknow_reader;
GRANT SELECT ON schema_info TO needtoknow_app;

ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE doc_access ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE schema_info ENABLE ROW LEVEL SECURITY;

-- The API and the MCP server read the record at startup, before anyone signs in.
CREATE POLICY schema_info_read ON schema_info FOR SELECT TO needtoknow_app USING (true);

-- app.principals is set per transaction by db.as_user through set_identity. When it is unset the
-- list is NULL or empty, so no row matches.
CREATE POLICY doc_access_read ON doc_access FOR SELECT TO needtoknow_app
USING (
    principal = ANY (string_to_array(current_setting('app.principals', true), ','))
);

CREATE POLICY documents_read ON documents FOR SELECT TO needtoknow_app
USING (
    EXISTS (
        SELECT 1
        FROM doc_access
        WHERE doc_access.doc_id = documents.id
          AND doc_access.principal = ANY (string_to_array(current_setting('app.principals', true), ','))
    )
);

CREATE POLICY chunks_read ON chunks FOR SELECT TO needtoknow_app
USING (
    EXISTS (
        SELECT 1
        FROM doc_access
        WHERE doc_access.doc_id = chunks.doc_id
          AND doc_access.principal = ANY (string_to_array(current_setting('app.principals', true), ','))
    )
);

-- needtoknow_reader is the evaluation's model of an application the database does not protect:
-- it reads every row, and its own SQL or code must drop what the asker may not see.
CREATE POLICY documents_reader ON documents FOR SELECT TO needtoknow_reader USING (true);
CREATE POLICY doc_access_reader ON doc_access FOR SELECT TO needtoknow_reader USING (true);
CREATE POLICY chunks_reader ON chunks FOR SELECT TO needtoknow_reader USING (true);
