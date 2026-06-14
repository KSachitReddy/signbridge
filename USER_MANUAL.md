# User Manual — SignBridge AI

> **Sign. Connect. Belong.**
> Version 1.3.0 | © 2025 SignBridge

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Getting Started](#2-getting-started)
3. [Interface Overview](#3-interface-overview)
4. [Learning Mode](#4-learning-mode)
5. [Translation Mode](#5-translation-mode)
6. [Language Settings](#6-language-settings)
7. [Privacy & Data](#7-privacy--data)
8. [Accessibility Features](#8-accessibility-features)
9. [Troubleshooting](#9-troubleshooting)
10. [FAQs](#10-faqs)
11. [Contact & Support](#11-contact--support)

---

## 1. Introduction

**SignBridge AI** is a free, browser-based application that helps bridge communication gaps for deaf and mute individuals. It requires no downloads, no special hardware, and no account — just a modern browser and a webcam.

### What can SignBridge do?

- **Teach** sign language through interactive, gamified video lessons
- **Translate** your signs into text in real time using your webcam
- **Demonstrate** text as signs for hearing users who want to communicate back
- **Support** 7 languages: English, Telugu, Hindi, Tamil, Kannada, Malayalam, and Tulu

### Who is SignBridge for?

| You are... | SignBridge helps you... |
|------------|------------------------|
| A deaf or mute individual | Communicate through sign translation and express yourself digitally |
| A hearing person | Learn sign language and communicate back with sign demonstrations |
| An educator | Teach sign language interactively with structured lessons |
| A business or organization | Support inclusive workplace communication |

---

## 2. Getting Started

### System Requirements

| Requirement | Minimum |
|-------------|---------|
| Browser | Chrome 90+, Firefox 88+, Edge 90+, Safari 15+ |
| Camera | Any built-in or USB webcam |
| Internet | Required for initial load; gesture recognition runs on-device |
| OS | Windows, macOS, Linux, Android, iOS |

No account or login is required.

### Opening SignBridge

1. Open your browser and navigate to: **https://signbridge-ai-final.vercel.app**
2. The app loads in a few seconds. No installation is needed.
3. Select your preferred language from the top-right language selector.

### Granting Camera Permission

When you open **Translation Mode** for the first time, your browser will ask for camera access.

- Click **Allow** when prompted.
- If you accidentally clicked **Block**, see [Troubleshooting → Camera Blocked](#camera-permission-blocked).
- Your camera feed never leaves your device. See [Privacy & Data](#7-privacy--data).

---

## 3. Interface Overview

```
┌─────────────────────────────────────────────────┐
│  🤟 SignBridge AI           [Language ▼]  [Menu]│
├─────────────┬───────────────────────────────────┤
│             │                                   │
│  Navigation │         Main Content Area         │
│             │                                   │
│  📚 Learn   │  (Learning Mode / Translation     │
│  🔄 Translate│   Mode content appears here)     │
│             │                                   │
└─────────────┴───────────────────────────────────┘
```

### Navigation Sidebar

- **📚 Learn** — Opens Learning Mode with lessons and practice sessions
- **🔄 Translate** — Opens Translation Mode for real-time sign-to-text and text-to-sign

### Language Selector (top right)

Switch your display and output language at any time. Available: EN, TE, HI, TA, KN, ML, TCY.

---

## 4. Learning Mode

Learning Mode is your interactive classroom for sign language. It is structured into **Lessons** and **Practice Sessions**.

### 4.1 Browsing Lessons

1. Click **📚 Learn** in the sidebar.
2. The lesson library appears, grouped by topic (Greetings, Numbers, Alphabet, Common Phrases, etc.).
3. Click any lesson card to open it.

### 4.2 Watching a Lesson

- Each lesson includes a **video demonstration** of the sign.
- Use the playback controls to pause, rewind, or slow down the video.
- Written descriptions and tips appear beside the video.

### 4.3 Practicing with AI Feedback

1. After watching a lesson, click **Practice Now**.
2. Allow camera access if prompted.
3. Perform the sign in front of your webcam.
4. SignBridge AI evaluates your sign in real time and shows:
   - A **score out of 100**
   - Specific **feedback** on handshape, movement, timing, and location
   - An **encouragement message**
5. Keep practicing until you reach a passing score (80+) to unlock the next lesson.

### 4.4 Gamification & Progress

- Complete lessons to earn **badges** and maintain your **learning streak**.
- Your progress is saved locally in your browser.
- The **Progress Dashboard** shows lessons completed, your streak, and badges earned.

> **Tip:** Practice in a well-lit area with your hands clearly visible against a plain background for the best recognition accuracy.

---

## 5. Translation Mode

Translation Mode enables real-time communication in both directions.

### 5.1 Sign-to-Text (for Deaf & Mute Users)

Use this when you want to sign and have your signs converted into written text.

1. Click **🔄 Translate** in the sidebar.
2. Ensure **Sign → Text** is selected (default).
3. Click **Start Camera**.
4. Begin signing in front of your webcam.
5. Translated text appears in the **Output Panel** on the right in real time.
6. Click **Copy** to copy the text, or **Clear** to reset the output.

**Tips for best accuracy:**
- Keep your hands within the camera frame at all times.
- Sign at a natural, moderate pace — very fast signing may reduce accuracy.
- Ensure good lighting on your hands.
- Use a plain, non-cluttered background if possible.

### 5.2 Text-to-Sign (for Hearing Users)

Use this when you want to type a message and see how it is signed, to communicate back.

1. In Translation Mode, select **Text → Sign**.
2. Type your message in the **Input Box**.
3. Click **Show Sign** (or press Enter).
4. The animated sign demonstration plays in the display panel.
5. You can adjust playback speed using the speed control (0.5×, 1×, 1.5×).

### 5.3 Dual Communication

Both panels can be open side by side for a two-way conversation between a signing user and a hearing user sharing a screen or device.

---

## 6. Language Settings

### Changing the Output Language

1. Click the **Language Selector** dropdown in the top-right corner.
2. Choose from: English (EN), Telugu (TE), Hindi (HI), Tamil (TA), Kannada (KN), Malayalam (ML), Tulu (TCY).
3. The interface and translation output update immediately.

### Language Fallback

If a particular sign does not have a direct mapping in the selected language, SignBridge will display the English equivalent with a note: *"No direct [Language] equivalent — showing English."*

---

## 7. Privacy & Data

SignBridge AI is designed with privacy as a core principle.

| Data Type | How It Is Handled |
|-----------|------------------|
| Webcam video | Processed entirely on your device. Never transmitted to any server. |
| Learning progress | Stored locally in your browser (localStorage). Never sent to SignBridge servers. |
| Usage analytics | Anonymized, aggregated page visit data only. No personal data. |
| Account data | No account required. No personal data is collected. |

**To clear your local data:** Open your browser settings → Site data → Clear data for signbridge-ai-final.vercel.app.

---

## 8. Accessibility Features

SignBridge AI is built to be accessible to all users.

- **Screen reader support** — All interactive elements have ARIA labels. Tested with NVDA and VoiceOver.
- **Keyboard navigation** — Full keyboard navigation is supported throughout the application.
- **High contrast support** — Respects your OS high-contrast mode settings.
- **Text scaling** — The interface scales correctly with browser font size settings up to 200%.
- **No auto-play** — Video lessons do not auto-play; they require a deliberate user action.
- **Captions** — All lesson videos include captions.

If you encounter any accessibility issue, please report it at **accessibility@signbridge.ai**.

---

## 9. Troubleshooting

### Camera Permission Blocked

**Problem:** You accidentally denied camera access and the camera does not start.

**Solution (Chrome):**
1. Click the 🔒 lock icon in the address bar.
2. Find **Camera** and change it to **Allow**.
3. Refresh the page.

**Solution (Firefox):**
1. Click the camera icon in the address bar.
2. Remove the blocked permission.
3. Refresh and click **Allow** when prompted again.

**Solution (Safari):**
1. Go to **Safari → Settings → Websites → Camera**.
2. Find SignBridge and set it to **Allow**.
3. Refresh the page.

---

### Camera Shows Black Screen

**Possible causes and fixes:**

| Cause | Fix |
|-------|-----|
| Another app is using the camera | Close other apps using the webcam (Zoom, Teams, etc.) and refresh |
| Browser camera permission not granted | See Camera Permission Blocked above |
| Outdated browser | Update your browser to the latest version |
| Driver issue (Windows) | Update your webcam drivers via Device Manager |

---

### Signs Not Being Recognized

- **Lighting:** Ensure your hands are well lit. Avoid backlighting (e.g., sitting in front of a bright window).
- **Background:** A plain, non-cluttered background improves detection accuracy.
- **Frame:** Keep your hands fully within the camera frame. Partially off-screen hands are not recognized.
- **Speed:** Sign at a natural, moderate pace.
- **Clothing:** High contrast between your hands and clothing helps; avoid gloves.

---

### App Not Loading

1. Check your internet connection.
2. Clear your browser cache and reload.
3. Try a different browser.
4. Check the SignBridge status page: **https://status.signbridge.ai** *(if available)*.

---

### Progress Not Saved

Learning progress is saved in your browser's local storage. Progress is lost if you:
- Clear browser data / site data
- Use a different browser or device
- Use private/incognito mode

To back up progress, use the **Export Progress** button in the Progress Dashboard.

---

## 10. FAQs

**Q: Is SignBridge AI free to use?**
A: Yes, SignBridge AI is completely free. No subscription, no account, no fees.

**Q: Does SignBridge work offline?**
A: Currently, an internet connection is required to load the app. Offline support is on the roadmap.

**Q: Which sign language does SignBridge use?**
A: SignBridge currently focuses on common sign vocabulary used in the Indian context, with support for 7 regional languages. Support for formal ISL (Indian Sign Language) and BSL is on the roadmap.

**Q: Can I use SignBridge on my phone?**
A: Yes. SignBridge is fully responsive and works on mobile browsers with a front-facing camera. A mobile app (React Native) is on the roadmap.

**Q: Is my video stored anywhere?**
A: No. All video processing happens on your device. Nothing is recorded or transmitted.

**Q: Can I contribute new sign language vocabulary?**
A: Yes! See our [Contributing Guide](CONTRIBUTING.md) on GitHub to learn how to contribute sign data.

**Q: The app is slow on my computer. What can I do?**
A: Close unused browser tabs, ensure no other apps are using the webcam, and try Chrome for best WebAssembly performance.

---

## 11. Contact & Support

| Channel | Details |
|---------|---------|
| 🌐 Live App | https://signbridge-ai-final.vercel.app |
| 📧 General Support | support@signbridge.ai |
| ♿ Accessibility | accessibility@signbridge.ai |
| 💬 GitHub Issues | https://github.com/signbridge/signbridge-ai/issues |
| 📋 Code of Conduct | See `CODE_OF_CONDUCT.md` |

For bug reports, please include:
- Your browser and version
- Your operating system
- A description of what you expected vs. what happened
- A screenshot if possible

---

*Made with ❤️ by the SignBridge Team | © 2025 SignBridge. All rights reserved.*