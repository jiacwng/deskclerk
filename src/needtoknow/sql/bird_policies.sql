-- Row-level security for two BIRD databases, run by needtoknow_owner after the dump is loaded.
-- This is configuration: each rule reads a column the data already has. In financial a row
-- belongs to the district of its account, in california_schools to the county of its school.
-- group:analyst reads every school row and every financial row that joins to an account, so a
-- client with no disp row stays hidden even from the analyst. financial.district, the list of
-- districts with their demographics, is readable by everyone on purpose, as are the other nine
-- BIRD databases.

ALTER TABLE financial.account ENABLE ROW LEVEL SECURITY;
ALTER TABLE financial.loan ENABLE ROW LEVEL SECURITY;
ALTER TABLE financial.trans ENABLE ROW LEVEL SECURITY;
ALTER TABLE financial."order" ENABLE ROW LEVEL SECURITY;
ALTER TABLE financial.disp ENABLE ROW LEVEL SECURITY;
ALTER TABLE financial.card ENABLE ROW LEVEL SECURITY;
ALTER TABLE financial.client ENABLE ROW LEVEL SECURITY;
ALTER TABLE california_schools.schools ENABLE ROW LEVEL SECURITY;
ALTER TABLE california_schools.frpm ENABLE ROW LEVEL SECURITY;
ALTER TABLE california_schools.satscores ENABLE ROW LEVEL SECURITY;

CREATE POLICY account_read ON financial.account FOR SELECT TO needtoknow_app
USING (
    string_to_array(current_setting('app.principals', true), ',')
        && ARRAY['group:analyst', 'group:district:' || district_id]
);

-- A policy's subquery reads account under account_read, so these rows follow their account.
CREATE POLICY loan_read ON financial.loan FOR SELECT TO needtoknow_app
USING (EXISTS (SELECT 1 FROM financial.account WHERE account.account_id = loan.account_id));

CREATE POLICY trans_read ON financial.trans FOR SELECT TO needtoknow_app
USING (EXISTS (SELECT 1 FROM financial.account WHERE account.account_id = trans.account_id));

CREATE POLICY order_read ON financial."order" FOR SELECT TO needtoknow_app
USING (EXISTS (SELECT 1 FROM financial.account WHERE account.account_id = "order".account_id));

CREATE POLICY disp_read ON financial.disp FOR SELECT TO needtoknow_app
USING (EXISTS (SELECT 1 FROM financial.account WHERE account.account_id = disp.account_id));

CREATE POLICY card_read ON financial.card FOR SELECT TO needtoknow_app
USING (EXISTS (SELECT 1 FROM financial.disp WHERE disp.disp_id = card.disp_id));

CREATE POLICY client_read ON financial.client FOR SELECT TO needtoknow_app
USING (EXISTS (SELECT 1 FROM financial.disp WHERE disp.client_id = client.client_id));

CREATE POLICY schools_read ON california_schools.schools FOR SELECT TO needtoknow_app
USING (
    string_to_array(current_setting('app.principals', true), ',')
        && ARRAY['group:analyst', 'group:county:' || county]
);

CREATE POLICY frpm_read ON california_schools.frpm FOR SELECT TO needtoknow_app
USING (
    string_to_array(current_setting('app.principals', true), ',')
        && ARRAY['group:analyst', 'group:county:' || "County Name"]
);

CREATE POLICY satscores_read ON california_schools.satscores FOR SELECT TO needtoknow_app
USING (
    string_to_array(current_setting('app.principals', true), ',')
        && ARRAY['group:analyst', 'group:county:' || cname]
);
