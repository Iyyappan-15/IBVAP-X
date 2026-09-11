# IBVAP-X REST API Endpoint Documentation

FastAPI base URL: `http://127.0.0.1:8000`

---

## System Endpoints

### GET `/health`
Returns system health, version, and active database mode (`postgresql` or `sqlite`).

### GET `/cameras`
Lists all active configured surveillance cameras.

### GET `/alerts`
Lists active priority alerts.

### POST `/alerts/{alert_id}/acknowledge`
Records operator ACKNOWLEDGED action for an alert.

### GET `/evidence/{evidence_id}`
Retrieves sealed evidence record metadata and SHA-256 hash.

### GET `/coverage`
Computes 3-tier camera coverage and returns polygon lists.
