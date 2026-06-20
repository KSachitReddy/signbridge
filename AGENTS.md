# AGENTS.md — AI Agent Architecture

> This document describes the AI agents, models, and intelligent components that power SignBridge AI.

---

## Overview

SignBridge AI is built around a multi-agent AI architecture where each agent has a clearly scoped responsibility. Agents communicate via well-defined interfaces, keeping the system modular, testable, and replaceable as better models become available.

```
User Input (webcam / text)
        │
        ▼
┌─────────────────────┐
│   Input Router      │  ← Determines input type and directs to correct agent
└────────┬────────────┘
         │
   ┌─────┴──────┐
   ▼            ▼
[GestureAgent] [TextAgent]
   │            │
   └─────┬──────┘
         ▼
 ┌───────────────────┐
 │  Translation Agent │  ← Converts between sign and text representations
 └────────┬──────────┘
          ▼
 ┌───────────────────┐
 │  Language Agent   │  ← Applies target language (EN/TE/HI/TA/KN/ML/TCY)
 └────────┬──────────┘
          ▼
 ┌───────────────────┐
 │  Feedback Agent   │  ← Generates learning feedback (Learning Mode only)
 └────────┬──────────┘
          ▼
       UI Output
```

---

## Agent Definitions

### 1. Input Router

**Role:** Classifies incoming user input and dispatches it to the appropriate downstream agent.

| Property       | Value                        |
| -------------- | ---------------------------- |
| Type           | Rule-based classifier        |
| Input          | Webcam stream OR text string |
| Output         | Routing decision + payload   |
| Latency Target | < 10ms                       |

**Logic:**

- If webcam stream is active → route to `GestureAgent`
- If text is provided → route to `TextAgent`
- If both → run in parallel for dual-mode (advanced)

---

### 2. GestureAgent

**Role:** Processes webcam frames in real-time to detect and classify hand gestures into sign language tokens.

| Property       | Value                                      |
| -------------- | ------------------------------------------ |
| Type           | Computer Vision model (on-device)          |
| Framework      | MediaPipe / TensorFlow.js                  |
| Input          | Video frames (30fps)                       |
| Output         | Sign token sequence with confidence scores |
| Processing     | Client-side (no data leaves the browser)   |
| Latency Target | < 120ms per frame                          |

**Key behaviors:**

- Detects hand landmarks using a 21-point skeleton model
- Classifies static signs (letters) and dynamic signs (words/phrases)
- Handles single-hand and dual-hand gestures
- Degrades gracefully under low-light or partial occlusion conditions

**Privacy guarantee:** All video processing runs entirely in the browser. No frames are transmitted to any server.

---

### 3. TextAgent

**Role:** Accepts text input from hearing users and prepares it for sign language demonstration output.

| Property       | Value                              |
| -------------- | ---------------------------------- |
| Type           | NLP pre-processor                  |
| Input          | Raw text string                    |
| Output         | Tokenized, normalized text payload |
| Latency Target | < 30ms                             |

**Key behaviors:**

- Normalizes text (lowercasing, punctuation removal, abbreviation expansion)
- Tokenizes into sign-mappable units
- Detects language of input text for correct downstream routing

---

### 4. Translation Agent

**Role:** The core translation engine that converts between sign token sequences and natural language text, in both directions.

| Property    | Value                                               |
| ----------- | --------------------------------------------------- |
| Type        | Sequence-to-sequence model                          |
| Direction A | Sign tokens → Natural language text                 |
| Direction B | Natural language text → Sign animation cues         |
| Input       | Token sequence from GestureAgent or TextAgent       |
| Output      | Translated text string OR sign animation descriptor |

**Key behaviors:**

- Supports contextual translation (maintains short-term conversation context)
- Handles grammatical differences between sign language syntax and spoken/written language syntax
- Emits confidence scores; low-confidence outputs are flagged to the user

---

### 5. Language Agent

**Role:** Applies the user's selected output language, translating the normalized output from the Translation Agent into the target language.

| Property            | Value                             |
| ------------------- | --------------------------------- |
| Type                | Localization + translation layer  |
| Supported Languages | EN, TE, HI, TA, KN, ML, TCY       |
| Input               | Normalized translated text        |
| Output              | Localized text in target language |

**Key behaviors:**

- Loads language packs dynamically; only the active language is loaded into memory
- Falls back to English if a sign has no direct mapping in the target language
- Respects regional script rendering (Devanagari, Telugu, Tamil, Kannada, Malayalam scripts)

---

### 6. Feedback Agent

**Role:** Active only in **Learning Mode**. Evaluates the learner's gesture attempts and generates instructional feedback.

| Property | Value                                                  |
| -------- | ------------------------------------------------------ |
| Type     | Scoring + generative feedback model                    |
| Input    | Learner's gesture output + expected gesture descriptor |
| Output   | Score (0–100), textual feedback, correction hints      |
| Trigger  | Practice session in Learning Mode                      |

**Scoring dimensions:**

- **Handshape accuracy** — correct finger positions
- **Movement accuracy** — correct motion path
- **Timing** — correct speed and rhythm
- **Location** — correct position relative to body

**Feedback format:**

```json
{
  "score": 82,
  "pass": true,
  "feedback": "Great handshape! Try to move your wrist slightly outward on the return stroke.",
  "corrections": ["wrist_orientation", "movement_return"],
  "encouragement": "You're 82% of the way there — keep going!"
}
```

---

## Agent Communication Protocol

Agents communicate via an internal event bus (browser-side for on-device agents, WebSocket for any server-side extensions).

```
Event format:
{
  "agent": "GestureAgent",
  "event": "sign_detected",
  "payload": {
    "tokens": ["HELLO", "MY", "NAME"],
    "confidence": [0.95, 0.88, 0.91],
    "timestamp": 1718000000000
  }
}
```

---

## Adding a New Agent

1. Create your agent class under `src/lib/agents/`
2. Implement the standard `Agent` interface:
   ```typescript
   interface Agent {
     name: string;
     process(input: AgentInput): Promise<AgentOutput>;
     onError(error: Error): void;
   }
   ```
3. Register your agent in `src/lib/agents/index.ts`
4. Add unit tests under `src/lib/agents/__tests__/`
5. Document it in this file under a new numbered section.

---

## Future Agents (Roadmap)

| Agent            | Description                                                      | Status  |
| ---------------- | ---------------------------------------------------------------- | ------- |
| `ContextAgent`   | Maintains multi-turn conversation context across sessions        | Planned |
| `VideoCallAgent` | Integrates with video calling platforms for real-time captioning | Planned |
| `OfflineAgent`   | Manages on-device model caching for offline support              | Planned |
| `CommunityAgent` | Powers the learner community forum and peer feedback features    | Planned |

---

_© 2025 SignBridge. All rights reserved._
