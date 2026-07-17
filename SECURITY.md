# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 0.1.x | Yes |

## Reporting a Vulnerability

This is a research forecasting project with no production deployment. If you
discover a security issue, please open a GitHub issue with the "security"
label. Do not disclose vulnerabilities publicly until they have been addressed.

## Known Security Considerations

- No network connections at runtime (data is downloaded once via scripts).
- No API keys, credentials, or secrets required.
- The dataset is public (UCI ML Repository, CC BY 4.0).
- The project uses minimal dependencies (numpy, pandas, scipy, scikit-learn,
  matplotlib, pyyaml), all well-maintained.
- Run `pip-audit` periodically to check for dependency vulnerabilities.
