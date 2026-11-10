# KB Manager content packs

Versioned YAML packs served by `GET /registry/api/*` and loaded at ingest/search time.

| Kind | Path | Purpose |
|------|------|---------|
| `schemas` | `schemas/vX.Y.Z.yaml` | XLSX sheet schema detection (no hardcoded columns in parsers) |
| `taxonomy` | `taxonomy/vX.Y.Z.yaml` | Subspace graph, filter paths, boosts, evidence thresholds, search UI groups |
| `canned` | `canned/vX.Y.Z.yaml` | Greetings / refusals / trigger phrases |
| `prompts` | `prompts/orchestrator/vX.Y.Z.yaml` | Orchestrator classify / hybrid / clarify / rewrite prompts |
| `retrieval` | `retrieval/vX.Y.Z.yaml` | Retrieval knobs + `active_packs` pins |
| `manifests` | `manifests/kb-*.json` | IndexManifest promotion records + `ACTIVE.json` |

Active pins are read from `retrieval.active_packs` (falling back to latest semver per kind).

Bump a pack by adding `vX.Y.Z.yaml`, then update `retrieval/v*.yaml` `active_packs` and build a new IndexManifest.
