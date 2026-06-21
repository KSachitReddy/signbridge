# Changelog

All notable changes to **SignBridge AI** will be documented in this file.

This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and the format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

### Planned

- Multi-language sign language support (ISL, BSL, ASL extended vocabulary)
- Mobile application via React Native
- Offline mode with on-device model inference
- Integration with video calling platforms (Google Meet, Zoom)
- Community forum for learners and educators

---

### Added

- **Tulu (TCY)** language support — SignBridge now supports 7 regional Indian languages
- UI language selector with instant switching between all supported languages
- Privacy-focused on-device video processing — no webcam data is sent to servers
- Gamification system for Learning Mode: streaks, badges, and progress milestones

### Changed

- Improved gesture recognition accuracy for Malayalam and Kannada sign mappings
- Upgraded to Next.js latest stable release for improved rendering performance
- Redesigned the onboarding flow for first-time users

### Fixed

- Webcam permission dialog not appearing on Safari (iOS 17+)
- Language selector state not persisting after page refresh
- Minor layout overflow in Translation Mode on mobile screens below 375px

---

## [1.2.0] — 2025-04-15

### Added

- **Tamil (TA)**, **Kannada (KN)**, and **Malayalam (ML)** language support
- Text-to-sign demonstration panel for hearing users
- Instant feedback system in AI-powered practice sessions
- Accessibility audit pass — WCAG 2.1 AA compliance verified

### Changed

- Refactored gesture recognition pipeline for lower latency (<120ms on average)
- Migrated styling to Tailwind CSS utility-first approach
- Updated component library to align with accessible design tokens

### Fixed

- Sign recognition failure when multiple hands appear in frame
- Erratic frame drops in Translation Mode under low-light conditions
- Screen reader labels missing on interactive UI elements

---

## [1.1.0] — 2025-03-01

### Added

- **Hindi (HI)** and **Telugu (TE)** language support alongside English
- Interactive video lesson library in Learning Mode (initial set of 30 lessons)
- Real-time sign-to-text translation using webcam-based gesture recognition
- Dual-direction communication: sign-to-text and text-to-sign
- Fully responsive layout for desktop and mobile browsers

### Changed

- Rewrote AI inference layer using optimized computer vision pipeline
- Improved onboarding UX with guided tutorial for first-time users

### Deprecated

- Legacy REST polling endpoint for gesture results — replaced with WebSocket stream

### Fixed

- Black screen on initial webcam load in Chromium-based browsers
- Translation panel text overflow with long Hindi strings

---

## [1.0.0] — 2025-01-20

### Added

- Initial public release of SignBridge AI
- Learning Mode with basic English sign language video lessons
- Translation Mode with English sign-to-text recognition (alpha)
- Vercel deployment pipeline with CI/CD via GitHub Actions
- MIT License and project documentation (README, contributing guide)
- Static analysis, unit testing, and test coverage hooks
- Secret scanning and conventional changelog enforcement

---

## Version Legend

| Symbol       | Meaning                                           |
| ------------ | ------------------------------------------------- |
| `Added`      | New features introduced                           |
| `Changed`    | Changes in existing functionality                 |
| `Deprecated` | Features that will be removed in a future release |
| `Removed`    | Features removed in this release                  |
| `Fixed`      | Bug fixes                                         |
| `Security`   | Addressed vulnerabilities                         |

---

[Unreleased]: https://github.com/signbridge/signbridge-ai/compare/v1.3.0...HEAD
[1.3.0]: https://github.com/signbridge/signbridge-ai/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/signbridge/signbridge-ai/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/signbridge/signbridge-ai/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/signbridge/signbridge-ai/releases/tag/v1.0.0
