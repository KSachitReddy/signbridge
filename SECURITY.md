# Security Policy

## Supported Versions

The following versions of SignBridge currently receive security updates.

| Version | Supported |
| ------- | --------- |
| 1.x     | ✅        |
| < 1.0   | ❌        |

---

# Reporting a Vulnerability

The SignBridge team takes security seriously.

If you discover a security vulnerability, please report it responsibly and privately.

## Contact

Security reports should be sent to:

**Email:** [sachitreddy.7@gmail.com](mailto:sachitreddy.7@gmail.com)

Please do not create public issues for security vulnerabilities.

Instead, send a private report containing:

- Vulnerability description
- Affected component
- Steps to reproduce
- Potential impact
- Proof of concept (if available)
- Suggested mitigation (optional)

---

# Response Process

After receiving a report:

1. Acknowledgement within 72 hours.
2. Initial assessment and validation.
3. Risk classification.
4. Development of remediation.
5. Security testing of the fix.
6. Release of security patch.
7. Public disclosure after remediation.

---

# Vulnerability Severity Levels

## Critical

Examples:

- Remote code execution
- Authentication bypass
- Arbitrary file access
- Privilege escalation
- Database compromise

Target remediation:

- Immediate investigation
- Patch released as soon as possible

---

## High

Examples:

- Exposure of sensitive user data
- Unauthorized access to protected resources
- Session hijacking
- JWT token compromise

Target remediation:

- High priority patch release

---

## Medium

Examples:

- Limited information disclosure
- Input validation weaknesses
- Configuration weaknesses

Target remediation:

- Included in the next scheduled release

---

## Low

Examples:

- Security hardening recommendations
- Minor security misconfigurations

Target remediation:

- Future maintenance release

---

# Security Practices

SignBridge follows the following security practices:

## Authentication

- Password hashing using modern cryptographic algorithms
- Token-based authentication
- Session validation
- Access control enforcement

## Secrets Management

- Secrets must never be committed to Git
- Environment variables must be used
- Production secrets must be stored securely

## Dependency Security

The project regularly audits dependencies using:

- npm audit
- pnpm audit
- GitLab Dependency Scanning

Vulnerable dependencies should be updated promptly.

## Secret Scanning

The repository includes:

- Gitleaks
- Pre-commit scanning
- CI security checks

Any detected secrets must be removed immediately.

## Input Validation

All external input should be:

- Validated
- Sanitized
- Type checked

Input should never be trusted by default.

## Principle of Least Privilege

Services, users, and applications should operate with the minimum permissions necessary.

---

# Secure Development Guidelines

Developers contributing to SignBridge must:

- Follow secure coding practices
- Validate all user input
- Avoid hardcoded credentials
- Avoid storing secrets in source code
- Keep dependencies updated
- Write security-focused tests
- Review authentication and authorization logic carefully

---

# Responsible Disclosure

We ask security researchers to:

- Avoid disrupting production services
- Avoid accessing user data unnecessarily
- Avoid public disclosure before remediation
- Provide sufficient technical detail for reproduction

Researchers acting in good faith will be treated respectfully and professionally.

---

# Compliance

SignBridge security processes aim to align with:

- OWASP Top 10
- Secure Software Development Lifecycle (SSDLC)
- Principle of Least Privilege
- Responsible Disclosure Practices

---

# Security Updates

Security advisories and remediation information will be published through project releases and repository announcements.

Thank you for helping keep SignBridge secure.
