# 📝 LinkedIn Post Manager (Rich TUI & CLI Edition)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Llama 3.3](https://img.shields.io/badge/AI%20Power-NVIDIA%20NIM-magenta.svg)](https://build.nvidia.com)

A feature-complete terminal application for creating, scheduling, and tracking LinkedIn posts. Featuring a beautiful **Rich-powered TUI** (Terminal User Interface) and direct **CLI power-user commands**, it integrates with **NVIDIA NIM API (Llama 3)** for natural, human-sounding post generations while degrading gracefully to a local offline manual manager.

---

## 🌟 Key Features

* **Beautiful Terminal UI:** Styled with panels, responsive tables, interactive prompt forms, progress spinner overlays, and raw markdown rendering.
* **Onboarding & Memory Store:** First-run onboarding collects your role, interests, goals, and desired writing tone, saving them locally in an offline JSON-based store.
* **Anti-AI Writing Engine:** Generates posts that read naturally and avoid structural and vocabulary "AI-isms" (filler words like *Moreover*, *Pivotal*, *Revolutionize*, *Leverage*, or *Delve* are strictly filtered).
* **Fidelity-Rich Image Prompts:** Automatically appends camera parameters, lighting details, styles, and palettes for Stable Diffusion or Midjourney.
* **Interactive Template Wizard:** Choose from pre-defined structural templates, and fill in the placeholders dynamically with a step-by-step wizard.
* **Weekly Planner & Pacing Goals:** Mon-Sun weekly planner that computes how many posts are generated and published, nudging you if you fall behind your weekly goals.
* **Daily Streak Mechanics:** Gamified statistics dashboard calculating consecutive active usage days.
* **Graceful Degradation:** Toggles between AI-assisted mode and manual drafting depending on API Key presence.

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/linkedin-post-manager.git
   cd linkedin-post-manager
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Get an NVIDIA NIM API Key:**
   Get a free development key from [NVIDIA Build API](https://build.nvidia.com).

---

## ⚙️ Configuration & API Key Safety

To ensure your API credentials **never get leaked** to public repositories, the project includes a strict `.gitignore` file configuration. 

You can set your API key in three safe ways:

### Option A: The Settings Menu (Recommended)
1. Run the manager: `python linkedin_manager.py`
2. Select Option **7** (Settings).
3. Select Option **K** (Set NIM API Key) and paste your key.
4. The application will automatically create and populate `nim_config.json` locally.

### Option B: Manual Configuration Template
1. Copy the example configuration template:
   ```bash
   cp nim_config.json.example nim_config.json
   ```
2. Open `nim_config.json` and replace `nvapi-YOUR_NVIDIA_NIM_API_KEY_HERE` with your actual NIM developer key.

### Option C: Environment Variable
You can export the environment variable `NIM_API_KEY` before running:

* **Windows PowerShell:**
  ```powershell
  $env:NIM_API_KEY="nvapi-YourKeyHere"
  ```
* **Linux/macOS Bash:**
  ```bash
  export NIM_API_KEY="nvapi-YourKeyHere"
  ```

---

## 🚀 Usage

The project supports two main operating modes:

### 1. Interactive Menu (TUI)
Launch the interactive dashboard to manage drafts, templates, planners, and settings:
```bash
python linkedin_manager.py
```

### 2. Direct CLI Sub-commands (Power-User Mode)
Skip the interactive menu and run commands directly from your terminal:

* **AI Generation:**
  ```bash
  python linkedin_manager.py generate --topic "Clean Code principles" --style tip
  ```
* **Manual Drafting:**
  ```bash
  python linkedin_manager.py write
  ```
* **List Drafts/Published:**
  ```bash
  python linkedin_manager.py list
  ```
* **View a specific post:**
  ```bash
  python linkedin_manager.py view --id 3
  ```
* **Mark a post as published:**
  ```bash
  python linkedin_manager.py publish --id 3
  ```
* **Check planner, stats, templates, or help:**
  ```bash
  python linkedin_manager.py week
  python linkedin_manager.py stats
  python linkedin_manager.py templates
  python linkedin_manager.py help
  ```

---

## 📂 Project Structure

```
├── linkedin_manager.py     # Main TUI loop, CLI argument dispatcher, and UI menus
├── nim_client.py           # OpenAI-compatible API wrapper with custom prompt rules
├── nim_config.json.example # Template file showing required API configuration layout
├── requirements.txt        # Third-party dependency definitions
├── templates.json          # Structural writing templates database
├── memory/
│   ├── __init__.py         # Local storage module initialization
│   └── store.py            # Facade managing local profile, prefs, and history JSONs
└── .gitignore              # Safety exclusion file protecting configs and data
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
