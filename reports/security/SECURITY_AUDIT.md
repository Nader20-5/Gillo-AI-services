# Comprehensive Security Audit Report: Gillo / Taskove AI Service

**Audit Date:** November 14, 2026  
**Auditor:** Antigravity AI Security Specialist (`security-audit` skill)  
**Scope:** Repository roots (`app/`, `scripts/`, `config`, `requirements`, `docker-compose.yml`, `.env*`)  
**Execution Context:** Local offline test verification; zero external network probing.  
**Raw Test Evidence:** [`reports/security/pytest_security_audit_raw.txt`](file:///d:/Gillo%20AI%20Service/reports/security/pytest_security_audit_raw.txt), [`reports/security/pytest_full_suite_raw.txt`](file:///d:/Gillo%20AI%20Service/reports/security/pytest_full_suite_raw.txt)  
**Dependency Manifest:** [`reports/security/pip_list_raw.txt`](file:///d:/Gillo%20AI%20Service/reports/security/pip_list_raw.txt)  

---

## 1. Executive Summary

A comprehensive source-level and execution-based security audit of the Gillo AI Service was performed using the `security-audit` methodology. The review focused on five core risk surfaces:
1. **Authentication & Authorization**: Mock authentication controls, JWT algorithm pinning, expiration enforcement, production refusal of default secrets, and role claim boundaries.
2. **Model Deserialization & Registry Integrity**: SHA256 integrity validation vs authenticity of pickled LightGBM/scikit-learn models in `models/artifacts`, request-supplied path verification, and arbitrary code execution vectors.
3. **Input Handling & Injection Defense**: Cross-project tenant isolation (IDOR), prompt injection barriers (`<UNTRUSTED_DATA>` / `<USER_QUERY>`), oversized payload / ReDoS guardrails, and query filter injection.
4. **Secrets, Configuration & Information Disclosure**: Environment variables, CORS configuration, rate limiter identity tracking, and sensitive credential / exception leaks in logs and health endpoints.
5. **Dependencies**: Analysis of installed packages against known vulnerabilities and maintenance status.

### Summary of Actions Taken
- **10 Vulnerabilities Remediated (Fixed):** Project-scoping authorization on sprint and suggestion endpoints, production refusal of `ALLOW_MOCK_AUTH`, JWT algorithm pinning, `sort_by` allowlist validation, max-length limits on chat/suggestion inputs, prompt injection delimiters, MongoDB URI credential masking in logs, health check exception disclosure prevention, and JWT-aware rate limiting.
- **15 New Security Unit Tests Added:** Created [`tests/test_security_audit.py`](file:///d:/Gillo%20AI%20Service/tests/test_security_audit.py) verifying all fixes.
- **Suite Health:** 100 passed out of 100 tests (100% pass rate).
- **8 Architectural / Design Decision Items Documented (Open):** Documented with severity, concrete evidence, and specific remediation plans without modifying code prematurely.

---

## 2. Findings Table

| ID | Title | Severity | File:Line Reference | Evidence | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SEC-01** | Production refusal of `ALLOW_MOCK_AUTH` | **High** | [`app/config.py:77-82`](file:///d:/Gillo%20AI%20Service/app/config.py#L77-L82)<br>[`app/middleware/auth.py:48-54`](file:///d:/Gillo%20AI%20Service/app/middleware/auth.py#L48-L54) | In production mode, if `ALLOW_MOCK_AUTH=True` was passed via environment, callers could bypass JWT auth and impersonate any user/role via `x-mock-*` headers. | **FIXED** |
| **SEC-02** | JWT Algorithm Pinning (`none` / unpinned) | **Medium** | [`app/config.py:68-76`](file:///d:/Gillo%20AI%20Service/app/config.py#L68-L76) | `JWT_ALGORITHM` had no whitelist validation; setting algorithm to `"none"` or insecure schemes was not rejected at startup. | **FIXED** |
| **SEC-03** | Weak Production Secret Entropy Check | **Medium** | [`app/config.py:84-93`](file:///d:/Gillo%20AI%20Service/app/config.py#L84-L93) | Production JWT validation checked only against a short list of default strings without enforcing a minimum key length (e.g., &ge; 32 characters). | **FIXED** |
| **SEC-04** | Explicit `exp` Claim Requirement | **Medium** | [`app/middleware/auth.py:35`](file:///d:/Gillo%20AI%20Service/app/middleware/auth.py#L35) | Added explicit `verify_exp: True` alongside `require_exp: True` in python-jose decode options to ensure deterministic rejection of unexpired and tokenless expiry payloads. | **FIXED** |
| **SEC-05** | Role Claims Trust via Symmetric Secret | **High** | [`app/middleware/auth.py:38-42`](file:///d:/Gillo%20AI%20Service/app/middleware/auth.py#L38-L42) | Symmetric `HS256` secret is shared between Core Agile and AI Service; any compromised microservice with `JWT_SECRET` can mint tokens with `PM`/`TL` roles. | **OPEN** (Design) |
| **SEC-06** | Pickle Deserialization Integrity vs Authenticity | **High** | [`app/ml/registry.py:304-405`](file:///d:/Gillo%20AI%20Service/app/ml/registry.py#L304-L405)<br>[`app/ml/registry.py:465-498`](file:///d:/Gillo%20AI%20Service/app/ml/registry.py#L465-L498) | Model pickles are verified using SHA-256 hashes loaded from `metadata.json` placed directly beside the pickle files. Write access allows updating both, leading to RCE via `pickle.load()`. | **OPEN** (Design) |
| **SEC-07** | IDOR / Missing Project Scoping in Sprints | **High** | [`app/routers/sprints.py:75, 246, 484`](file:///d:/Gillo%20AI%20Service/app/routers/sprints.py#L75) | `POST /sprints/{id}/estimate`, `GET /sprints/{id}/fit-check`, and `POST /sprints/{id}/scan-risks` failed to verify that the sprint's `project_id` matched `current_user.project_id`. | **FIXED** |
| **SEC-08** | IDOR / Missing Project Scoping in Suggestions | **High** | [`app/routers/suggestions.py:127-147`](file:///d:/Gillo%20AI%20Service/app/routers/suggestions.py#L127-L147)<br>[`app/routers/suggestions.py:196-380`](file:///d:/Gillo%20AI%20Service/app/routers/suggestions.py#L196-L380) | `POST /suggestions/{id}/accept`, `reject`, `dismiss`, `apply`, and `comments` did not verify `current_user.project_id`, allowing cross-project suggestion state tampering. | **FIXED** |
| **SEC-09** | DoS via Unrestricted `sort_by` in Suggestions | **Medium** | [`app/routers/suggestions.py:76-83`](file:///d:/Gillo%20AI%20Service/app/routers/suggestions.py#L76-L83) | `GET /suggestions` accepted arbitrary `sort_by` parameters, enabling callers to trigger unindexed in-memory MongoDB sorts. | **FIXED** |
| **SEC-10** | DoS via Unbounded Input Field Lengths | **Medium** | [`app/routers/chat.py:32-35`](file:///d:/Gillo%20AI%20Service/app/routers/chat.py#L32-L35)<br>[`app/models/suggestions.py:60-71`](file:///d:/Gillo%20AI%20Service/app/models/suggestions.py#L60-L71) | `ChatRequest.query`, `decision_reason`, and comment `body` had no maximum length constraints, allowing megabyte-sized payloads. | **FIXED** |
| **SEC-11** | Prompt Injection in Chat Query | **Medium** | [`prompts/chat/v1.md:12-19`](file:///d:/Gillo%20AI%20Service/prompts/chat/v1.md#L12-L19) | User query was appended directly as `User Question: {query}` without delimitation, vulnerable to prompt injection instructions. | **FIXED** |
| **SEC-12** | Information Leak in Public `/health` Endpoint | **Low** | [`app/main.py:65-68`](file:///d:/Gillo%20AI%20Service/app/main.py#L65-L68) | `/health` leaked internal exception strings (`unreachable: {str(e)}`) including internal IP addresses and connection errors to unauthenticated callers. | **FIXED** |
| **SEC-13** | Credential Leak in MongoDB Logging | **Low** | [`app/database.py:10-33`](file:///d:/Gillo%20AI%20Service/app/database.py#L10-L33) | `logger.info` logged `settings.MONGO_URI` directly, printing cleartext credentials into log collectors if embedded in the URI. | **FIXED** |
| **SEC-14** | Rate Limiting Shared Across All JWT Callers | **Low** | [`app/middleware/rate_limit.py:33-65`](file:///d:/Gillo%20AI%20Service/app/middleware/rate_limit.py#L33-L65) | `check_llm_rate_limit` fell back to client IP for JWT users because `x-mock-user-id` was absent, grouping all users behind a NAT into one bucket. | **FIXED** |
| **SEC-15** | Insecure Permissive CORS with Credentials | **Medium** | [`app/main.py:48-54`](file:///d:/Gillo%20AI%20Service/app/main.py#L48-L54) | `allow_origins=["*"]` combined with `allow_credentials=True` violates CORS specifications and allows arbitrary web origins to send credentialed requests. | **OPEN** (Design) |
| **SEC-16** | Hardcoded Third-Party API Key in `.env` | **High** | [`.env:19`](file:///d:/Gillo%20AI%20Service/.env#L19) | Key was found, redacted from reports, must be revoked by the owner. | **OPEN until owner confirms revocation** |
| **SEC-17** | Unauthenticated Mongo Express Administrative UI | **High** | [`docker-compose.yml:25-26`](file:///d:/Gillo%20AI%20Service/docker-compose.yml#L25-L26) | `ME_CONFIG_BASICAUTH: "false"` on exposed port 8081 permits complete unauthenticated database read/write/drop access. | **OPEN** (Design) |
| **SEC-18** | Dependency Maintenance (`passlib`, `python-jose`) | **Low** | [`requirements.txt:11-12`](file:///d:/Gillo%20AI%20Service/requirements.txt#L11-L12) | `passlib` is unmaintained and unused in source; `python-jose` has historical algorithm confusion advisories (CVE-2024-33663). | **OPEN** (Design) |

---

## 3. Detailed Audit by Category

### Category 1: Authentication & Authorization

#### Findings & Fixes
- **Mock Authentication Hardening (SEC-01):**
  - *Vulnerability:* `ALLOW_MOCK_AUTH` was intended solely for local test suites. If configured as `True` in a production environment, unauthenticated attackers could supply arbitrary `x-mock-user-id` and `x-mock-role: TL` headers to bypass all JWT verification.
  - *Remediation:* Enforced two defensive layers:
    1. [`app/config.py`](file:///d:/Gillo%20AI%20Service/app/config.py): Added `validate_security_settings` Pydantic validator raising a fatal `ValueError` if `ENVIRONMENT == "production"` and `ALLOW_MOCK_AUTH == True`.
    2. [`app/middleware/auth.py`](file:///d:/Gillo%20AI%20Service/app/middleware/auth.py): In `get_current_user`, explicitly returns HTTP 401 Unauthorized if mock headers are received while `settings.ENVIRONMENT.lower() == "production"`.
- **JWT Algorithm Pinning (SEC-02):**
  - *Vulnerability:* Insecure JWT implementations risk algorithm confusion attacks (e.g. setting `"alg": "none"` or using HMAC with RSA public keys).
  - *Remediation:* Pydantic settings now strictly pin `JWT_ALGORITHM` against an approved set (`HS256`, `HS384`, `HS512`, `RS256`, `RS384`, `RS512`, `ES256`, `ES384`, `ES512`) and unconditionally reject `"none"`.
- **JWT Expiration & Secret Strength (SEC-03, SEC-04):**
  - *Vulnerability:* Default development secret (`"gillo_super_secret_dev_jwt_key_2026"`) was rejected in production, but keys shorter than 32 characters or with low entropy were not blocked.
  - *Remediation:* Required key length &ge; 32 characters in production; reinforced `jwt.decode` with `options={"verify_exp": True, "require_exp": True}`.

#### Open Item / Design Decision: Symmetric Key Trust (SEC-05)
- **Severity:** High
- **Analysis:** The AI service uses symmetric HMAC (`HS256`). This means any service sharing the secret key (e.g. Core Agile backend, integration tests, batch jobs) possesses the cryptographic capability to sign tokens claiming any role (`PM`, `TL`, `Developer`).
- **Proposed Architectural Mitigation:** Migrate to asymmetric JWTs (`RS256` or `ES256`). Core Agile signs tokens using its private key; Gillo AI Service loads only the public verification key (or fetches JWKS from `http://core-agile/.well-known/jwks.json`). This ensures the AI service can never be tricked by compromised internal peers.

---

### Category 2: Model Loading & Deserialization

#### Verification of Request-Supplied Paths
- **Audit Result:** **CONFIRMED CLEAN.**
- **Evidence:** We traced every invocation of `pickle.load` across the repository. All 10 deserialization calls reside in [`app/ml/registry.py`](file:///d:/Gillo%20AI%20Service/app/ml/registry.py#L392-L498).
- The model names (`model_sp_taskove`, `model_est`, `model2_sprint_carryover_risk_taskove`, `model_risk`) and version strings (`v2_taskove`, `v2_taskove_features`) are strictly hardcoded or loaded from `Settings` (`MODEL1_ACTIVE_VERSION`, `MODEL2_ACTIVE_VERSION`).
- **No HTTP route, query parameter, header, or payload body provides a file path or version string to any model loading mechanism.**

#### Open Item / Design Decision: Integrity vs Authenticity in Model Deserialization (SEC-06)
- **Severity:** High
- **Risk Assessment:** The model registry calculates the SHA-256 hash of `.pkl` files and verifies it against the hash in `metadata.json` located in the exact same directory (e.g. `models/artifacts/model_sp_taskove/v2_taskove/metadata.json`). While this provides **integrity** against accidental file corruption or incomplete downloads, it provides **zero authenticity or tamper-resistance**. An attacker with local file write permissions (e.g. via container volume mount or separate vulnerability) can write a weaponized pickle payload and simultaneously overwrite `metadata.json` with the new payload's SHA-256. Python's `pickle.load()` will execute the `__reduce__` exploit before returning.
- **Proposed Mitigation Plan:**
  1. *Immediate:* Mount `models/artifacts` as a strictly **read-only volume** (`read_only: true` in Docker/Kubernetes).
  2. *Near-Term:* Store canonical expected SHA-256 hashes in immutable application configuration (`app/config.py`) or inject them via signed CI/CD release manifests, decoupled from the mutable artifact directory.
  3. *Long-Term:* Deprecate Python `pickle` for model serving. Convert scikit-learn / LightGBM models to **ONNX** (`onnxruntime`), **safetensors**, or LightGBM native text dump formats (`Booster.save_model` / `lgb.Booster(model_file=...)`), which parse model weights without executing arbitrary code.

---

### Category 3: Input Handling & Injection Defense

#### Findings & Fixes
- **Project-Boundary Scoping / Tenant Isolation (SEC-07, SEC-08):**
  - *Vulnerability:* While `GET /suggestions` properly checked `current_user.project_id`, the mutation endpoints (`/suggestions/{id}/accept`, `/reject`, `/dismiss`, `/apply`, `/comments`) and sprint endpoints (`/sprints/{id}/estimate`, `/fit-check`, `/scan-risks`) lacked validation. A team lead assigned to Project A could inspect or mutate suggestions and sprint estimations in Project B simply by knowing or guessing the sprint/suggestion ID.
  - *Remediation:* Added project ownership validation on all 8 endpoints:
    - If `current_user.project_id` is present and does not match the target document's `project_id`, the service immediately aborts with `HTTP 403 Forbidden` (`"Access denied to this project's suggestion"`).
- **NoSQL / Query Filter Injection Guardrails (SEC-09):**
  - *Vulnerability:* `GET /suggestions` accepted user-controlled `sort_by` query parameters directly into `.sort(sort_by, sort_direction)`. Unindexed fields could cause excessive CPU and RAM utilization in MongoDB.
  - *Remediation:* Enforced an explicit allowlist: `{"created_at", "severity", "status", "type", "confidence"}`. Any unrecognized sort field returns `HTTP 422 Unprocessable Content`.
- **Payload Size & ReDoS Defense (SEC-10):**
  - *Vulnerability:* Pydantic schemas allowed arbitrary length for `ChatRequest.query`, `decision_reason`, and comment `body`.
  - *Remediation:* Applied `max_length=2000` to `query`, `decision_reason`, and `body` across [`app/models/suggestions.py`](file:///d:/Gillo%20AI%20Service/app/models/suggestions.py) and [`app/routers/chat.py`](file:///d:/Gillo%20AI%20Service/app/routers/chat.py).
- **Prompt Injection Defense (SEC-11):**
  - *Vulnerability:* In [`prompts/chat/v1.md`](file:///d:/Gillo%20AI%20Service/prompts/chat/v1.md), user questions were formatted as raw text without boundary tags.
  - *Remediation:* Wrapped user input inside explicit `<USER_QUERY>` delimiters and added system prompt instructions declaring all text within `<TOOL_DATA>` and `<USER_QUERY>` as inert, untrusted data.
- **Admin Role Blocking Verification:**
  - *Audit Result:* **CONFIRMED ENFORCED.**
  - Every single AI router (`/suggestions`, `/sprints`, `/standup`, `/retro`, `/release-requests`, `/chat`) enforces `require_roles(...)` or `get_current_user`, which explicitly checks `if user.role.lower() == "admin": raise HTTPException(403)`. Automated test [`test_role_matrix.py::test_admin_blocked_on_all_ai_endpoints`](file:///d:/Gillo%20AI%20Service/tests/test_role_matrix.py#L7-L38) confirms HTTP 403 across all 9 AI endpoints.

---

### Category 4: Secrets, Config & Privacy

#### Findings & Fixes
- **Information Leak in `/health` (SEC-12):**
  - *Vulnerability:* When MongoDB was unreachable, `health_check()` returned `{"database": f"unreachable: {str(e)}"}`. Network exception strings can disclose internal hostnames, cluster IP addresses, and topology.
  - *Remediation:* Sanitized response to `{"database": "unreachable"}` and logged the full exception internally via `logger.error`.
- **Sensitive Credential Logging (SEC-13):**
  - *Vulnerability:* `connect_to_mongo()` in [`app/database.py`](file:///d:/Gillo%20AI%20Service/app/database.py) logged the complete `settings.MONGO_URI`. If credentials (`mongodb://user:pass@host/db`) were configured, secrets were emitted to log collectors.
  - *Remediation:* Added `sanitize_mongo_uri()` regex mask replacing password components with `****`.
- **Rate Limiting Key Identification (SEC-14):**
  - *Vulnerability:* Sliding-window limiter in [`app/middleware/rate_limit.py`](file:///d:/Gillo%20AI%20Service/app/middleware/rate_limit.py) checked only `x-mock-user-id` or IP, grouping all authenticated JWT users sharing an egress gateway into a single limit.
  - *Remediation:* Added unverified claim inspection of the `Authorization: Bearer` header to key rate limits by authenticated `sub`/`user_id`.

#### Open Items / Design Decisions:
- **CORS Configuration (SEC-15):**
  - *File:* [`app/main.py:48-54`](file:///d:/Gillo%20AI%20Service/app/main.py#L48-L54)
  - *Evidence:* `allow_origins=["*"]` with `allow_credentials=True`.
  - *Proposed Fix:* Replace wildcard with configurable `CORS_ORIGINS: List[str]` in `Settings` (e.g. `["http://localhost:3000"]` in development, explicit domain allowlist in staging/production).
- **Hardcoded Secret in `.env` (SEC-16):**
  - *File:* [`.env:19`](file:///d:/Gillo%20AI%20Service/.env#L19)
  - *Evidence:* Key was found, redacted from reports, must be revoked by the owner.
  - *Proposed Fix:* Key must be revoked by the owner in the Groq Cloud dashboard. Remove hardcoded key and inject via environment variables or secrets vault.
  - *Status:* OPEN until owner confirms revocation.
- **Docker Compose Mongo Express (SEC-17):**
  - *File:* [`docker-compose.yml:25-26`](file:///d:/Gillo%20AI%20Service/docker-compose.yml#L25-L26)
  - *Evidence:* `ME_CONFIG_BASICAUTH: "false"` on port 8081.
  - *Proposed Fix:* Set `ME_CONFIG_BASICAUTH: "true"` with non-default credentials, or remove Mongo Express from non-local development environments.

---

### Category 5: Dependencies & Supply Chain

#### Package Inventory & Vulnerability Check (SEC-18)
- Complete package listing generated via `pip list` and saved to [`reports/security/pip_list_raw.txt`](file:///d:/Gillo%20AI%20Service/reports/security/pip_list_raw.txt).
- **Key Package Findings:**
  1. **`passlib[bcrypt]>=1.7.4`:** Listed in [`requirements.txt:12`](file:///d:/Gillo%20AI%20Service/requirements.txt#L12), but source code search confirms `passlib` is not imported or used anywhere in `app/` or `scripts/`. Passlib has not been updated since 2020 and has known incompatibilities with modern `bcrypt`. Recommendation: Remove from `requirements.txt`.
  2. **`python-jose[cryptography]==3.5.0`:** Listed in [`requirements.txt:11`](file:///d:/Gillo%20AI%20Service/requirements.txt#L11). Historical CVE-2024-33663 and CVE-2024-33664 affect python-jose. Recommendation: Schedule migration to actively maintained `PyJWT[crypto]>=2.8.0`.
  3. **`fastapi==0.143.2`, `pydantic==2.14.1`, `scikit-learn==1.7.2`, `lightgbm==4.7.0`:** Up to date.

---

## 4. List of Code Changes

The following files were modified to fix the clear, low-risk vulnerabilities:

1. **[`app/config.py`](file:///d:/Gillo%20AI%20Service/app/config.py):**
   - Added `validate_security_settings` post-validator.
   - Enforced JWT algorithm whitelist and blocked `"none"`.
   - Strictly forbade `ALLOW_MOCK_AUTH=True` in production mode.
   - Enforced minimum 32-character length for production `JWT_SECRET`.
2. **[`app/middleware/auth.py`](file:///d:/Gillo%20AI%20Service/app/middleware/auth.py):**
   - Added runtime block against mock authentication headers when `ENVIRONMENT == "production"`.
   - Added explicit `verify_exp: True` to JWT decode options.
3. **[`app/middleware/rate_limit.py`](file:///d:/Gillo%20AI%20Service/app/middleware/rate_limit.py):**
   - Added extraction of user subject claim from Bearer tokens to properly partition rate limits by authenticated identity rather than shared IP.
4. **[`app/database.py`](file:///d:/Gillo%20AI%20Service/app/database.py):**
   - Added `sanitize_mongo_uri()` function masking username/password credentials before logging MongoDB connection strings.
5. **[`app/main.py`](file:///d:/Gillo%20AI%20Service/app/main.py):**
   - Replaced exception string disclosure in `/health` with generic `"unreachable"`, routing error details to server logs.
6. **[`app/routers/sprints.py`](file:///d:/Gillo%20AI%20Service/app/routers/sprints.py):**
   - Enforced project-scoping authorization on `estimate_sprint_tasks`, `check_sprint_fit`, and `scan_sprint_risks`. Returns HTTP 403 on cross-project access.
7. **[`app/routers/suggestions.py`](file:///d:/Gillo%20AI%20Service/app/routers/suggestions.py):**
   - Added allowlist validation for `sort_by` parameter in `list_suggestions` (HTTP 422 on invalid field).
   - Enforced project-scoping checks on `accept_suggestion`, `reject_suggestion`, `dismiss_suggestion`, `apply_confirmation`, and `add_suggestion_comment`.
8. **[`app/models/suggestions.py`](file:///d:/Gillo%20AI%20Service/app/models/suggestions.py):**
   - Added `max_length=2000` to `SuggestionDecisionRequest.decision_reason` and `SuggestionCommentRequest.body`.
9. **[`app/routers/chat.py`](file:///d:/Gillo%20AI%20Service/app/routers/chat.py):**
   - Added `max_length=2000` to `ChatRequest.query` and `max_length=100` to `session_id`.
   - Passed `project_id` to `chat_tools.get_release_status`.
10. **[`app/services/chat_tools.py`](file:///d:/Gillo%20AI%20Service/app/services/chat_tools.py):**
    - Scoped `get_my_tasks`, `get_sprint_status`, and `get_release_status` queries to caller's `project_id`.
11. **[`prompts/chat/v1.md`](file:///d:/Gillo%20AI%20Service/prompts/chat/v1.md):**
    - Enclosed user query inside `<USER_QUERY>` tags with anti-injection instructions.
12. **[`tests/test_security_audit.py`](file:///d:/Gillo%20AI%20Service/tests/test_security_audit.py):**
    - New automated test suite with 15 security regression tests.

---

## 5. Verification Results

All unit and integration tests passed cleanly:

```
tests/test_security_audit.py: 15 passed in 0.57s
Full Test Suite: 100 passed in 22.46s (0 failed, 0 errors)
```

Raw outputs are archived under:
- Regression Test Output: [`reports/security/pytest_security_audit_raw.txt`](file:///d:/Gillo%20AI%20Service/reports/security/pytest_security_audit_raw.txt)
- Complete Suite Output: [`reports/security/pytest_full_suite_raw.txt`](file:///d:/Gillo%20AI%20Service/reports/security/pytest_full_suite_raw.txt)
- Dependency Inventory: [`reports/security/pip_list_raw.txt`](file:///d:/Gillo%20AI%20Service/reports/security/pip_list_raw.txt)
