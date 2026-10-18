-- The only way the app role can set who is asking: provisioning takes set_config away from it.
-- It works once per transaction, so SQL that runs after db.as_user cannot call it again to become
-- someone else. Created with the tables of the needtoknow database, and in a company database by
-- python -m needtoknow.ingest data-setup.

CREATE OR REPLACE FUNCTION public.set_identity(principals TEXT) RETURNS VOID
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

REVOKE EXECUTE ON FUNCTION public.set_identity(TEXT) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.set_identity(TEXT) TO needtoknow_app;
