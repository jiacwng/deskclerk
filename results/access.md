# Access checks on Enron, without a model

Every email held by at least one custodian is opened as one custodian who does not hold it (254,544 emails), and 10,000 of them are searched for by their subject, top 5, as that custodian. Counts carry their 95% Wilson score interval. The holder rows show the same tools do return the email to a custodian allowed to read it.

| Check | Non-holder got the email | Holder got the email |
|---|---|---|
| open_document (of 254,544) | 0 (0.0% to 0.0015%) | 254544 (100.0% to 100.0%) |
| search, top 5 (of 10,000) | 0 (0.0% to 0.038%) | 3703 (36.1% to 38.0%) |
