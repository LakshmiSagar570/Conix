# Real-Time Webcam-Based Arousal Visualization Using Facial and Ocular Micro-Signals

## 📌 Project Definition
A real-time, on-device system that estimates and visualizes physiological arousal (tension) from facial and ocular micro-signals using a standard webcam. This tool functions as a **visual biofeedback and research instrument**, intentionally avoiding high-level claims regarding intent, truthfulness, or specific emotional states.

---

## ⚙️ Core Objective
The system is designed to:
* **Quantify Arousal:** Measure physiological correlates of tension over time.
* **Visual Feedback:** Provide a live tension graph for self-regulation or research observation.
* **Data Archiving:** Support post-session analysis of stress dynamics via local logs.

**What this is NOT:**
* ❌ A Lie Detector (Polygraph substitute).
* ❌ An Intent Reader or "Mind Reading" tool.
* ❌ A Psychological Diagnostic System.

---

## 🛠️ Technical Stack
The system is built on a **fully local, privacy-first architecture** with no external AI API dependencies.

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | `Python 3.10+` | Core logic and signal processing. |
| **Computer Vision** | `MediaPipe` | 468-point Face Mesh & Iris tracking. |
| **Frame Handling** | `OpenCV` | Camera I/O and visual overlays. |
| **Processing** | `NumPy` & `SciPy` | Vectorized math, EMA filtering, and Z-score normalization. |
| **Frontend/UI** | `Streamlit` | Real-time dashboard and live graphing. |
| **Visualization** | `Plotly` | Dynamic, interactive time-series charts. |
| **Storage** | `SQLite` | Local logging of feature vectors and tension scores. |

---

## 🧠 Signal Processing & Logic

### 1. Input Signals (Observable Only)
The system extracts raw geometric data from the video feed:
* **Ocular:** Blink rate/irregularity, gaze aversion duration, and relative pupil variability.
* **Kinematic:** Head pose jitter (Yaw/Pitch variance).
* **Morphological:** Micro-facial motion energy and landmark displacement (tension asymmetry).

### 2. The Tension Index ($T_i$)
The Tension Index is a bounded scalar (0–100) representing relative arousal compared to the user's specific baseline. It uses a weighted formula:

$$T_i(t) = \sum (w_1 \cdot \text{blink}_z + w_2 \cdot \text{gaze}_z + w_3 \cdot \text{jitter}_z + w_4 \cdot \text{motion}_z)$$

* **Calibration:** Every session begins with a 30–60 second "neutral" phase to establish $\mu$ and $\sigma$.
* **Normalization:** All features are Z-score normalized to ensure the system adapts to the individual.

---

## 🛡️ Ethical Boundaries & Privacy
* **No Cloud Processing:** All video frames are processed in volatile memory (RAM) and are **never** uploaded to an external API (No OpenAI, Gemini, or Azure).
* **Descriptive, Not Interpretive:** Annotations describe patterns (e.g., "Sustained arousal increase") rather than making judgments (e.g., "Subject is nervous").
* **Transparency:** The UI explicitly states: *"This system visualizes arousal patterns. It does not detect lies or mental states."*

---

## 🚀 Execution Roadmap
1.  **Phase 1:** MediaPipe Integration (Face + Iris landmark extraction).
2.  **Phase 2:** Signal Normalization (Baseline calibration and Z-score logic).
3.  **Phase 3:** Visualization Dashboard (Streamlit + Plotly live graphing).
4.  **Phase 4:** Persistence (SQLite logging for research analysis).
