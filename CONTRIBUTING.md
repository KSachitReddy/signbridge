# Contributing to SignBridge AI

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Adding new languages

## Our Development Process

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Internationalization (i18n) Contributions

To add a new language:
- Add a new JSON file in `locales/`.
- Register the language in `modules/locales/i18n.py` (for the Streamlit backend).
- Register the language in `src/components/LanguageSwitcher.tsx` (for the React frontend).

## Any questions?

Feel free to open an issue or contact the maintainers.
