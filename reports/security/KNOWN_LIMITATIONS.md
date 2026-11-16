# Security Architecture: Known Limitations & Long-Term Roadmap

This document details identified architectural limitations and long-term security recommendations for the Gillo AI Service. These items require architectural coordination or operational controls rather than simple single-line code patches.

---

## 1. SEC-05: Symmetric HS256 Shared Secret (JWT)

### Context & Current State
The AI Service authenticates requests originating from users and the Core Agile backend using JSON Web Tokens (JWT). Currently, tokens are signed using HMAC-SHA256 (`HS256`) with a shared symmetric secret (`JWT_SECRET`).

### Risk & Threat Model
Because `HS256` is symmetric, any service that verifies tokens must also possess the exact secret required to mint valid tokens. If either the AI Service or the upstream Core Agile backend suffers a credential compromise or unauthorized secret disclosure, an attacker can forge arbitrary JWTs with any role (`PM`, `TL`, `Developer`) and bypass authentication.

### Recommended Mitigation
- **Asymmetric Signature Migration (RS256 / ES256):** Migrate token signing to asymmetric cryptography (e.g., RSA with SHA-256 or ECDSA with P-256).
- **Core Agile Signs, AI Service Verifies:** The Core Agile authentication authority holds the private signing key. The AI Service only requires the public verification key.
- **JWKS Endpoint:** Implement JSON Web Key Set (JWKS) discovery (`/.well-known/jwks.json`) on the auth provider, allowing the AI service to periodically fetch and cache public keys with automated key rotation.

---

## 2. SEC-06: Pickle Deserialization & Artifact Integrity vs. Authenticity

### Context & Current State
Machine learning models (such as Sprint Risk and Estimation models) are loaded via Python's standard `pickle.load()` from `models/artifacts/`. An integrity validation routine calculates the SHA-256 checksum of the artifact and compares it against the digest recorded in `metadata.json` beside the file.

### Risk & Threat Model
Python `pickle` files can contain arbitrary Python bytecode instructions executed during unpickling. While the SHA-256 checksum check validates *integrity* (protecting against disk corruption or accidental truncation), it does not validate *authenticity* or *provenance*:
- An adversary or compromised process with local write access to the filesystem could overwrite both `model.pkl` and `metadata.json` with a malicious payload and its matching SHA-256 hash.
- Deserializing the tampered pickle would yield arbitrary remote code execution (RCE) in the context of the running service.

*Note:* Audit confirmed that no endpoint or code path accepts a user-controlled model path or deserializes pickles from HTTP requests; paths are strictly resolved against configured model directories.

### Recommended Mitigation
- **Operational Defense (Immediate):** Mount `models/artifacts` as a strictly **read-only volume** (`:ro`) in Docker/container orchestrator deployments so runtime processes cannot modify artifacts or metadata.
- **Cryptographic Signatures (Short-Term):** Instead of storing plain SHA-256 hashes in `metadata.json`, sign model digests with a trusted signing key (e.g. Ed25519 / RSA) during the CI/CD training pipeline, and verify the signature in `ModelRegistry` before loading.
- **Safe Format Migration (Long-Term):** Replace `pickle` with safe serialization formats that cannot execute code:
  - `safetensors` for neural network tensors and weights.
  - ONNX runtime (`.onnx`) for portable, sandboxed inference of Scikit-Learn/XGBoost models.
  - JSON / Parquet / Treelite for tabular model parameters.

---

## 3. Dependency Maintenance: `python-jose`

### Context & Current State
The service currently uses `python-jose[cryptography]` for JWT signature verification, claims validation, and token decoding.

### Risk & Threat Model
The upstream `python-jose` library has entered a dormant/unmaintained state, with infrequent maintenance releases and known security advisories concerning algorithm confusion and edge cases in older releases. While algorithm pinning (`algorithms=["HS256"]`) and explicit `verify_exp: True` have been enforced to prevent common exploits, maintaining dormant dependencies introduces supply chain risk.

### Recommended Mitigation
- **Migrate to `PyJWT[crypto]`:** Transition all JWT decoding and encoding to `PyJWT`, which is actively maintained, extensively audited, and provides native support for JWKS and modern cryptography standards.
