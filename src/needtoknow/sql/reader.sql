-- needtoknow_reader is the evaluation's model of an application the database does not protect:
-- it reads every row, and its own SQL or code must drop what the asker may not see. Run by
-- needtoknow_owner after schema.sql, only for the evaluation; running it again changes nothing.

GRANT SELECT ON documents, doc_access, chunks TO needtoknow_reader;

DROP POLICY IF EXISTS documents_reader ON documents;
DROP POLICY IF EXISTS doc_access_reader ON doc_access;
DROP POLICY IF EXISTS chunks_reader ON chunks;
CREATE POLICY documents_reader ON documents FOR SELECT TO needtoknow_reader USING (true);
CREATE POLICY doc_access_reader ON doc_access FOR SELECT TO needtoknow_reader USING (true);
CREATE POLICY chunks_reader ON chunks FOR SELECT TO needtoknow_reader USING (true);
