# Contributing to SignBridge AI

> Thank you for your interest in contributing to **SignBridge AI**! Every contribution — whether code, sign language data, translations, design, or documentation — helps us build a more inclusive world for deaf and mute individuals.

---

## Table of Contents

1. [Code of Conduct](#1-code-of-conduct)
2. [Ways to Contribute](#2-ways-to-contribute)
3. [Getting Started](#3-getting-started)
4. [Development Workflow](#4-development-workflow)
5. [Branch & Commit Conventions](#5-branch--commit-conventions)
6. [Pull Request Process](#6-pull-request-process)
7. [Code Style & Quality](#7-code-style--quality)
8. [Testing](#8-testing)
9. [Contributing Sign Language Data](#9-contributing-sign-language-data)
10. [Contributing Translations / Localization](#10-contributing-translations--localization)
11. [Reporting Bugs](#11-reporting-bugs)
12. [Suggesting Features](#12-suggesting-features)
13. [Community & Communication](#13-community--communication)

---

## 1. Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing. We are committed to keeping this a welcoming, harassment-free space for everyone.

---

## 2. Ways to Contribute

You do not need to be a developer to contribute. Here are all the ways you can help:

| Contribution Type   | Description                                                      |
| ------------------- | ---------------------------------------------------------------- |
| 🐛 Bug reports      | Report issues you encounter while using the app                  |
| 💡 Feature requests | Suggest improvements or new ideas                                |
| 💻 Code             | Fix bugs, implement features, improve performance                |
| 🤟 Sign data        | Contribute sign gesture datasets for new vocabulary or languages |
| 🌐 Translations     | Translate UI strings into supported or new languages             |
| 📖 Documentation    | Improve the README, User Manual, or inline code comments         |
| ♿ Accessibility    | Identify and fix accessibility gaps                              |
| 🧪 Testing          | Write or improve unit, integration, or E2E tests                 |
| 🎨 Design           | Improve UI/UX, icons, or visual design                           |

---

## 3. Getting Started

### Prerequisites

- **Node.js** v18 or higher
- **npm** v9+ or **yarn** v1.22+
- A modern browser with webcam support (for testing gesture features)
- **Git** 2.30+

### Fork & Clone

```bash
# 1. Fork the repository on GitHub (click the Fork button)

# 2. Clone your fork
git clone https://github.com/<your-username>/signbridge-ai.git

# 3. Navigate into the project
cd signbridge-ai

# 4. Add the upstream remote
git remote add upstream https://github.com/signbridge/signbridge-ai.git
```

### Install Dependencies

```bash
npm install
```

### Environment Variables

Copy the example environment file and fill in any required values:

```bash
cp .env.example .env.local
```

Refer to `.env.example` for descriptions of each variable. Never commit `.env.local` — it is already in `.gitignore`.

### Run the Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser. The app hot-reloads on file changes.

---

## 4. Development Workflow

```
1. Sync your fork with upstream
2. Create a feature branch from main
3. Make your changes
4. Write or update tests
5. Run linting and tests locally
6. Commit using Conventional Commits
7. Push and open a Pull Request
```

### Keeping Your Fork Up to Date

Before starting any new work, sync with upstream:

```bash
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

---

## 5. Branch & Commit Conventions

### Branch Naming

Use the following prefixes:

| Prefix      | Use for                                  |
| ----------- | ---------------------------------------- |
| `feature/`  | New features                             |
| `fix/`      | Bug fixes                                |
| `docs/`     | Documentation changes only               |
| `refactor/` | Code refactoring without behavior change |
| `test/`     | Adding or improving tests                |
| `chore/`    | Build, tooling, or dependency updates    |
| `a11y/`     | Accessibility improvements               |
| `i18n/`     | Localization / translation changes       |

**Examples:**

```
feature/tulu-language-support
fix/safari-camera-black-screen
docs/update-user-manual
a11y/add-aria-labels-translation-panel
```

### Conventional Commits

All commit messages must follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <short description>

[optional body]

[optional footer(s)]
```

**Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `a11y`, `i18n`

**Examples:**

```
feat(language): add Tulu (TCY) language support

fix(camera): resolve black screen on Safari iOS 17+

docs(manual): update troubleshooting section for Firefox

a11y(translate): add ARIA labels to sign output panel

test(gesture): add unit tests for dual-hand detection
```

Commits that do not follow this format will be flagged during the PR review.

---

## 6. Pull Request Process

### Before Opening a PR

- [ ] Your branch is up to date with `upstream/main`
- [ ] All tests pass (`npm test`)
- [ ] Linting passes (`npm run lint`)
- [ ] You have added or updated tests for your changes
- [ ] You have updated relevant documentation (README, USER_MANUAL, etc.) if needed
- [ ] Your commits follow the Conventional Commits format

### Opening the PR

1. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
2. Go to the [SignBridge AI repository](https://github.com/signbridge/signbridge-ai) on GitHub.
3. Click **Compare & pull request**.
4. Fill in the PR template completely:
   - **What does this PR do?**
   - **How has it been tested?**
   - **Screenshots / recordings** (for UI changes)
   - **Related issues** (use `Closes #123` to auto-close)

### PR Review Process

- A maintainer will review your PR within **3–5 business days**.
- You may be asked to make changes. Push additional commits to the same branch — do not close and reopen the PR.
- Once approved, a maintainer will squash-merge your PR into `main`.
- Your contribution will appear in the next `CHANGELOG.md` release entry.

### PR Size Guidelines

Keep PRs small and focused. A PR that does one thing is reviewed faster and merged sooner.

| PR Size   | Lines Changed | Guidance                                  |
| --------- | ------------- | ----------------------------------------- |
| Small ✅  | < 200         | Ideal — quick to review                   |
| Medium ⚠️ | 200–500       | Acceptable with clear description         |
| Large 🔴  | 500+          | Please split into smaller PRs if possible |

---

## 7. Code Style & Quality

### Linting & Formatting

We use **ESLint** and **Prettier** to enforce consistent code style.

```bash
# Check for lint errors
npm run lint

# Auto-fix lint errors
npm run lint:fix

# Format code with Prettier
npm run format
```

Lint and format checks run automatically on every PR via GitHub Actions. PRs with lint failures will not be merged.

### TypeScript

- All new code must be written in **TypeScript**.
- Avoid using `any` — use proper types or generics.
- Export types/interfaces from a dedicated `types.ts` file where appropriate.

### General Guidelines

- Write **self-documenting code** — prefer clear variable and function names over inline comments.
- Add **JSDoc comments** to all exported functions and components.
- Keep components **small and single-purpose**. If a component exceeds ~150 lines, consider splitting it.
- Avoid hardcoding strings — use the i18n system for all user-facing text.
- Do not introduce new dependencies without discussing in an issue first.

---

## 8. Testing

### Running Tests

```bash
# Run all unit and integration tests
npm test

# Run tests in watch mode (during development)
npm run test:watch

# Run with coverage report
npm run test:coverage
```

### Test Requirements

- All new features must include unit tests.
- Bug fixes should include a regression test that fails before the fix and passes after.
- Aim to maintain **80%+ code coverage**. PRs that significantly reduce coverage will be asked to add tests.

### Test Structure

```
src/
└── lib/
    └── agents/
        ├── GestureAgent.ts
        └── __tests__/
            └── GestureAgent.test.ts   ← test file lives beside the source
```

### Writing Tests

We use **Jest** and **React Testing Library**.

```typescript
// Example: src/lib/agents/__tests__/LanguageAgent.test.ts
import { LanguageAgent } from '../LanguageAgent';

describe('LanguageAgent', () => {
  it('returns translated text for supported language', async () => {
    const agent = new LanguageAgent();
    const result = await agent.process({ text: 'Hello', targetLang: 'TE' });
    expect(result.text).toBeDefined();
    expect(result.language).toBe('TE');
  });

  it('falls back to English for unmapped signs', async () => {
    const agent = new LanguageAgent();
    const result = await agent.process({ text: 'UNMAPPED_SIGN', targetLang: 'TCY' });
    expect(result.fallback).toBe(true);
  });
});
```

---

## 9. Contributing Sign Language Data

Sign language vocabulary is central to SignBridge. We welcome contributions of gesture data for new signs or new languages.

### Format

Sign data is stored in `src/lib/signs/` as JSON files, one per language:

```
src/lib/signs/
├── en.json
├── te.json
├── hi.json
└── ...
```

Each entry follows this schema:

```json
{
  "sign_id": "GREETING_HELLO",
  "label": {
    "en": "Hello",
    "te": "నమస్కారం",
    "hi": "नमस्ते"
  },
  "category": "greetings",
  "handshape": "open_b",
  "movement": "wave_outward",
  "location": "head_level",
  "dominant_hand": "right",
  "non_dominant_hand": null,
  "video_ref": "signs/hello.mp4",
  "added_by": "community",
  "version": "1.0"
}
```

### Steps to Contribute Sign Data

1. Open an issue first using the **Sign Data Contribution** template to discuss the vocabulary you want to add.
2. Add your entries to the relevant language JSON file(s).
3. If contributing a video demonstration, place it under `public/signs/` and reference it in `video_ref`.
4. Open a PR with the label `sign-data`.

### Video Demonstration Guidelines

- Format: MP4 (H.264), max 5MB per file
- Duration: 2–5 seconds, loopable
- Background: Plain white or light grey
- Lighting: Well lit, hands clearly visible
- Hands: No jewelry, plain skin-toned background preferred

---

## 10. Contributing Translations / Localization

UI strings are managed in `src/lib/i18n/` as JSON files:

```
src/lib/i18n/
├── en.json
├── te.json
├── hi.json
├── ta.json
├── kn.json
├── ml.json
└── tcy.json
```

### Adding a New UI Translation

1. Copy `en.json` as your starting point.
2. Translate all string values (keys must remain unchanged).
3. Place the file in `src/lib/i18n/<language_code>.json`.
4. Add the language to the selector in `src/components/LanguageSelector.tsx`.
5. Open a PR with the label `i18n`.

### Adding a Brand New Language

Open an issue using the **New Language Request** template before starting. New languages require:

- A UI translation JSON file
- At least partial sign vocabulary coverage in `src/lib/signs/`
- Confirmation that the language's script renders correctly

---

## 11. Reporting Bugs

Before filing a bug, please:

- Search [existing issues](https://github.com/signbridge/signbridge-ai/issues) to avoid duplicates.
- Confirm you're using a supported browser and version.

### Bug Report Template

When creating an issue, use the **Bug Report** template and include:

```
**Describe the bug**
A clear description of what happened.

**Steps to reproduce**
1. Go to '...'
2. Click on '...'
3. See error

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened.

**Screenshots / recordings**
If applicable.

**Environment**
- OS: [e.g. Windows 11, macOS 14, Android 13]
- Browser: [e.g. Chrome 124, Safari 17]
- Device: [e.g. Desktop, iPhone 14]
- SignBridge version: [e.g. 1.3.0]
```

---

## 12. Suggesting Features

We love feature ideas! Before submitting:

- Check the [Roadmap in README](README.md#roadmap) — it may already be planned.
- Search [existing issues](https://github.com/signbridge/signbridge-ai/issues) for duplicates.

Use the **Feature Request** template and describe:

- The problem your feature solves
- Who it helps (deaf users, educators, hearing users, etc.)
- Your proposed solution
- Any alternatives you considered

Features that directly improve accessibility and communication for deaf and mute users are prioritized.

---

## 13. Community & Communication

| Channel                                                                       | Purpose                                                |
| ----------------------------------------------------------------------------- | ------------------------------------------------------ |
| [GitHub Issues](https://github.com/signbridge/signbridge-ai/issues)           | Bug reports, feature requests, sign data contributions |
| [GitHub Discussions](https://github.com/signbridge/signbridge-ai/discussions) | General questions, ideas, community chat               |
| 📧 contribute@signbridge.ai                                                   | Reach the maintainers directly                         |

### Response Times

| Type             | Expected Response    |
| ---------------- | -------------------- |
| Bug reports      | 2–3 business days    |
| Feature requests | 3–5 business days    |
| Pull requests    | 3–5 business days    |
| Security issues  | 24 hours (see below) |

### Security Vulnerabilities

**Please do not report security vulnerabilities in public GitHub issues.**
Email **security@signbridge.ai** directly. We will acknowledge within 24 hours and aim to release a fix within 7 days.

---

## Recognition

All contributors are acknowledged in our release notes and CHANGELOG. Significant contributors may be invited to join the SignBridge maintainer team.

Thank you for helping us build technology that makes the world more inclusive. 🤟

---

_© 2025 SignBridge. All rights reserved._
