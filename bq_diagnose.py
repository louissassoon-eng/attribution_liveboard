"""Diagnose BigQuery service account access.

Tries: auth, listing datasets, table metadata, count query, table listing.
Pinpoints whether the issue is auth, project, dataset, table-level access,
or a wrong table path.
"""
from google.cloud import bigquery
from google.oauth2 import service_account
import google.api_core.exceptions as gae
import os

KEY = os.path.expanduser("~/.config/mrq/bq-service-account.json")
TARGET_PROJECT = "mrq-data"
TARGET_DATASET = "dbt"
TARGET_TABLE = "attribution_spend_metrics"

creds = service_account.Credentials.from_service_account_file(
    KEY, scopes=["https://www.googleapis.com/auth/cloud-platform"]
)
client = bigquery.Client(credentials=creds, project=creds.project_id)

print(f"Authenticated as: {creds.service_account_email}")
print(f"Project from key:  {creds.project_id}")
print()

print(f"=== Datasets this service account can list in {TARGET_PROJECT} ===")
try:
    datasets = list(client.list_datasets(TARGET_PROJECT))
    if not datasets:
        print("  (none visible — service account has no dataset read access)")
    for ds in datasets:
        print(f"  - {ds.project}.{ds.dataset_id}")
except Exception as e:
    print(f"  ERROR listing datasets: {type(e).__name__}: {str(e)[:300]}")

print()
full_path = f"{TARGET_PROJECT}.{TARGET_DATASET}.{TARGET_TABLE}"
print(f"=== Metadata read: {full_path} ===")
try:
    table = client.get_table(full_path)
    print(f"  OK — type: {table.table_type}, rows: {table.num_rows:,}, modified: {table.modified}")
    print(f"  Columns: {len(table.schema)}")
except gae.Forbidden as e:
    print(f"  403 FORBIDDEN — table exists but no read access")
    print(f"  Detail: {str(e)[:400]}")
except gae.NotFound as e:
    print(f"  404 NOT FOUND — table does not exist at this exact path")
    print(f"  Detail: {str(e)[:400]}")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {str(e)[:400]}")

print()
print(f"=== Count query against {full_path} ===")
try:
    query = f"SELECT COUNT(*) AS n FROM `{full_path}`"
    rows = list(client.query(query).result())
    print(f"  OK — row count: {rows[0].n:,}")
except Exception as e:
    print(f"  {type(e).__name__}: {str(e)[:400]}")

print()
print(f"=== Tables in {TARGET_PROJECT}.{TARGET_DATASET} (if listable) ===")
try:
    tables = list(client.list_tables(f"{TARGET_PROJECT}.{TARGET_DATASET}"))
    print(f"  Total tables visible: {len(tables)}")
    matched = [t for t in tables if "attribution" in t.table_id.lower() or "spend" in t.table_id.lower()]
    if matched:
        print("  Matching attribution/spend:")
        for t in matched[:20]:
            print(f"    - {t.project}.{t.dataset_id}.{t.table_id}")
    elif tables:
        print("  First 15 tables in dataset:")
        for t in tables[:15]:
            print(f"    - {t.table_id}")
except Exception as e:
    print(f"  Cannot list tables: {type(e).__name__}: {str(e)[:300]}")
