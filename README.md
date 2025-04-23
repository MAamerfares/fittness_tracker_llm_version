# Fitness Tracker Pro with AI Chat (Gemini Powered)

## Overview

This Python application utilizes computer vision and AI to provide a comprehensive fitness tracking experience. It uses your webcam or a video file to detect body pose (via MediaPipe), count repetitions for various exercises, provide basic form feedback, track workout statistics, and features an integrated AI chat assistant powered by Google Gemini to answer questions about your progress.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-blue.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-orange.svg)](https://developers.google.com/mediapipe)
[![Gemini API](https://img.shields.io/badge/Google%20AI-Gemini%20API-green.svg)](https://ai.google.dev/)

**(Optional: Add a Screenshot/GIF of the app in action here)**
<!-- ![App Screenshot](path/to/screenshot.png) -->

## Key Features

*   **Real-time Pose Estimation:** Uses Google's MediaPipe library for accurate body landmark detection.
*   **Multi-Exercise Tracking:** Supports:
    *   Bicep Curls (Left/Right independent counting)
    *   Squats
    *   Push-ups
    *   Pull-ups
    *   Deadlifts
*   **Repetition Counting:** Implements hysteresis logic based on joint angles to accurately count reps per exercise.
*   **Basic Form Correction:** Provides real-time feedback on common form issues like:
    *   Excessive back arching/rounding (exercise-specific thresholds)
    *   Upper arm movement during bicep curls
    *   Push-up body straightness
    *   Squat depth and knee valgus (knee caving in)
    *   Deadlift back angle during lift and lockout
    *   Highlights problematic joints/connections in red.
*   **Structured Workouts:** Option to configure workouts with specific sets, reps per set, and rest periods.
*   **Free Play Mode:** Track reps and form without a predefined set/rep structure.
*   **User Profiles:** Create and select user profiles to store basic information (Age, Height, Weight).
*   **Persistent Statistics:** Saves workout data (total reps, estimated calories, last workout config) per user per exercise to JSON files (`profiles.json`, `stats.json`).
*   **Statistics Visualization:** Displays overall calorie distribution via a pie chart and a textual summary of exercise stats.
*   **Visual Exercise Guides:** Shows animated GIFs for selected exercises before starting.
*   **AI Chat Assistant (Google Gemini):**
    *   Ask questions about your profile or workout statistics (e.g., "How many squats did I do?", "What is my height?").
    *   The AI uses your saved profile and stats data as context to provide informed answers.
*   **Flexible Input:** Works with both live webcam feed and pre-recorded video files (video file analysis does not save stats or support set structures).
*   **Custom UI:** Interface built using OpenCV drawing functions. Standard dialogs leverage Tkinter.

## Technology Stack

*   **Python:** 3.9+
*   **OpenCV:** (`opencv-python`) for video capture, image processing, and UI drawing.
*   **MediaPipe:** For real-time pose estimation.
*   **NumPy:** For numerical calculations (angles, vectors).
*   **Tkinter:** (Python standard library) For native file dialogs, message boxes, and input prompts.
*   **Matplotlib:** For generating the statistics pie chart.
*   **imageio:** For loading animated GIF guides.
*   **Google Generative AI:** (`google-generativeai`) for interacting with the Gemini language model.
*   **JSON:** For saving user profiles and statistics.

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install opencv-python mediapipe numpy matplotlib imageio google-generativeai
    ```
    *(Note: `openai` library is no longer required if project.

```markdown
# OpenCV Fitness Tracker with AI Coach (Gemini)

## Overview

This project is a real-time fitness tracker application built using Python, OpenCV, and MediaPipe. It utilizes your webcam (or a video file) to estimate your pose and track various exercises, count repetitions, provide basic form correction feedback, and log your workout statistics.

A key feature is the integration of an AI Fitness Coach powered by Google's Gemini model. Users can ask questions related to their profile, tracked statistics, or general fitness advice, and the AI will provide context-aware responses.

![Demo Image/GIF Placeholder](placeholder.gif)
*(Replace `placeholder.gif` with an actual screenshot or GIF demonstrating the app in action, perhaps showing exercise tracking and the chat interface)*

## Features

*   **Real-time Pose Estimation:** Uses MediaPipe Pose to detect body landmarks.
*   **Multi-Exercise Tracking:** Supports:
    *   Bicep Curls (Counts reps for each arm)
    *   Squats
    *   Push-ups
    *   Pull-ups
    *   Deadlifts
*   **Repetition Counting:** Automatically counts reps based on exercise-specific joint angles using hysteresis to prevent miscounts.
*   **Form Correction Feedback:** Provides real-time textual feedback for common form issues (e.g., back angle, knee valgus, arm movement, body straightness). Highlights problematic joints on the skeleton overlay.
*   **Structured Workouts:** Option to configure workouts with specific Sets, Reps per Set, and Rest times.
*   **Free Play Mode:** Track exercises without a predefined set/rep structure.
*   **User Profiles:** Create and select user profiles storing basic information (Age, Height, Weight).
*   **Workout Statistics:**
    *   Logs total reps per exercise per user.
    *   Estimates calories burned per session (based on MET values and profile weight).
    *   Remembers the last used Set/Rep/Rest configuration per user/exercise.
    *   Basic statistics visualization (calorie distribution pie chart).
*   **Visual Exercise Guides:** Displays animated GIFs for selected exercises before starting.
*   **AI Fitness Coach (Google Gemini):**
    *   Interact with an AI assistant via a chat interface.
    *   Ask questions about your profile, statistics, or general fitness.
    *   The AI uses your profile and tracked stats as context for its responses.
*   **Cross-Platform:** Built with libraries generally compatible with Windows, macOS, and Linux.

## Technologies Used

*   **Python 3.x**
*   **OpenCV (`opencv-python`)**: For video capture, image processing, and drawing the UI.
*   **MediaPipe (`mediapipe`)**: For real-time pose estimation.
*   **NumPy**: For numerical calculations (angles, vectors).
*   **Tkinter**: (Built-in Python library) For native OS dialog boxes (file selection, popups, chat input).
*   **Google Generative AI (`google-generativeai`)**: For interacting with the Gemini LLM.
*   **Matplotlib**: For generating the statistics pie chart.
*   **Imageio**: For loading animated GIF exercise guides.
*   **JSON**: For storing user profiles and statistics data.

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    It's recommended to create a `requirements.txt` file. You can generate one if you have the libraries installed using:
    ```bash
    pip freeze > requirements.txt
    ```
    Then install using:
    ```bash
    pip install -r requirements.txt
    ```
    Alternatively, install the core libraries manually:
    ```bash
    pip install opencv-python mediapipe numpy matplotlib imageio google-generativeai
    ```

4.  **API Key Configuration (Crucial for AI Chat):**
    *   Obtain an API key for the Gemini API from Google AI Studio: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
    *   **Set the API key as an environment variable BEFORE running the script.** Do **NOT** hardcode the key in the Python file. Replace `YOUR_GOOGLE_API_KEY` with your actual key.
        *   **Windows (Command Prompt):**
            ```bash
            set GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
            ```
        *   **Windows (PowerShell):**
            ```powershell
            $env:GOOGLE_API_KEY='YOUR_GOOGLE_API_KEY'
            ```
        *   **macOS/Linux (Bash/Zsh):**
            ```bash
            export GOOGLE_API_KEY='YOUR_GOOGLE_API_KEY'
            ```
    *   You need to set this variable in the *same terminal session* where you will run the Python script.

5.  **Exercise Guide GIFs:**
    *   Create a directory named `GIFs` in the same folder as the Python script.
    *   Place the corresponding GIF files (e.g., `bicep.gif`, `squats.gif`, etc.) inside the `GIFs` directory. The filenames should match those defined in the `EXERCISE_GIF_MAP` dictionary within the script.

## Usage

1.  **Ensure the `GOOGLE_API_KEY` environment variable is set** in your current terminal session.
2.  **Navigate to the project directory** in your terminal.
3.  **Run the script:**
    ```bash
    python your_script_name.py
    ```
    (Replace `your_script_name.py` with the actual name of your Python file, e.g., `llm.py`).
4.  **Home Screen:**
    *   Use "Create Profile" or "Select Profile" first.
    *   Click "View Stats" to see progress.
    *   Click "AI Chat" (requires a profile) to talk to the AI coach.
    *   Choose "Start Webcam Workout" or "Load Video".
5.  **Exercise Selection:**
    *   Click on the desired exercise.
    *   If using Webcam: Choose "Configure Sets" or "Start Free Play".
    *   If using Video: Click "Start Video".
6.  **Set Configuration (if applicable):** Adjust Sets, Reps, Rest using +/- buttons and click "Confirm & Start".
7.  **Guide Screen (if applicable):** View the exercise GIF. Click "Start Exercise" or wait for the timer.
8.  **Tracking Screen:** Perform the exercise. View rep counts and form feedback. Switch exercises using the top buttons if needed (resets workout structure for webcam source). Click the bottom-right ' directory of your project (the same folder where your Python script is).
2.  Copy the entire content below and paste it into that `README.md` file.
3.  **Crucially:** Replace the placeholder sections (like adding screenshots) with your actual content.
4.  Commit and push this file to your GitHub repository.

---

```markdown
# Fitness Tracker Pro with AI Chat (OpenCV & MediaPipe)

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg) <!-- Choose a license and update badge -->

Real-time fitness tracker using OpenCV for video processing and MediaPipe for pose estimation. Tracks reps and provides basic form feedback for various exercises. Features include user profiles, statistics tracking, structured workouts (sets/reps/rest), free play mode, visual guides (GIFs), and an integrated AI chat assistant powered by Google Gemini for personalized insights.

<!--
*** ADD SCREENSHOTS/GIFS HERE! ***
A few visuals demonstrating the main screen, tracking in action (with landmarks),
the stats screen, and the chat interface would be very helpful.
![Screenshot 1](path/to/screenshot1.png)
![Tracking GIF](path/to/tracking.gif)
-->

## Features

*   **Multi-Exercise Tracking:** Supports Bicep Curls, Squats, Push-ups, Pull-ups, and Deadlifts.
*   **Real-time Rep Counting:** Uses angle calculations and hysteresis logic to count repetitions accurately.
*   **Basic Form Feedback:** Provides real-time visual and text feedback on common form issues (e.g., back angle, arm/leg positioning, rep range).
*   **Structured Workouts:** Configure target sets, reps per set, and rest duration for guided sessions.
*   **Free Play Mode:** Track exercises without predefined sets or reps.
*   **User Profiles:** Create and select user profiles storing basic information (age, height, weight) used for calorie estimation.
*   **Statistics Tracking:**
    *   Saves total reps per exercise per user.
    *   Estimates total calories burned per exercise per user.
    *   Remembers the last used Set/Rep/Rest configuration per exercise for convenience.
    *   Displays overall stats with a pie chart for calorie distribution.
*   **Visual Exercise Guides:** Displays animated GIFs for selected exercises before starting.
*   **AI Chat Assistant:** Integrated chat interface using Google Gemini. Ask questions about your progress, form, or general fitness based on your profile and tracked stats.
*   **Cross-Platform:** Uses standard libraries compatible with Windows, macOS, and Linux (performance may vary).
*   **Custom UI:** Interface built using OpenCV drawing functions.

## Technology Stack

*   **Python:** 3.8 or higher recommended.
*   **OpenCV (`opencv-python`):** For camera/video input, image processing, and UI drawing.
*   **MediaPipe (`mediapipe`):** For real-time pose estimation.
*   **NumPy (`numpy`):** For numerical calculations (angles, vectors).
*   **Matplotlib (`matplotlib`):** For generating the statistics pie chart (uses Agg backend).
*   **Imageio (`imageio`):** For loading animated GIFs. (May require backend like `imageio[ffmpeg]` depending on GIF format and OS).
*   **Google Generative AI (`google-generativeai`):** For powering the AI Chat Assistant via the Gemini API.
*   **Tkinter:** Python's standard GUI toolkit (used for popups like file dialogs, message boxes, and chat input).

## Setup and Installation

1.  **Prerequisites:**
    *   Python 3.8+ installed.
    *   `pip` (Python package installer).
    *   Git (optional, for cloning).

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your_username/your_repository_name.git
    cd your_repository_name
    ```
    (Replace with your actual username and repository name)

3.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate it:
    # Windows:
    .\venv\Scripts\activate
    # macOS/Linux:
    source venv/bin/activate
    ```

4.  **Install Dependencies:**
    Create a `requirements.txt` file in the project root with the following content:
    ```txt
    opencv-python
    mediapipe
    numpy
    matplotlib
    imageio
    # Optional: Add imageio backend if needed, e.g., imageio[ffmpeg]
    google-generativeai
    ```
    Then install the requirements:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Tkinter is usually included with standard Python installations)*

5.  **Set Up Google Gemini API Key:**
    *   Obtain an API key from Google AI Studio: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
    *   **Crucially:** Set the API key as an environment variable named `GOOGLE_API_KEY` before running the script. **Do not hardcode the key in the Python file.**
        *   **Windows (Command Prompt):**
            ```bash
            set GOOGLE_API_KEY=YOUR_ACTUAL_GOOGLE_API_KEY
            ```
        *   **Windows (PowerShell):**
            ```powershell
            $env:GOOGLE_API_KEY='YOUR_ACTUAL_GOOGLE_API_KEY'
            ```
        *   **macOS / Linux (Bash/Zsh):**
            ```bash
            export GOOGLE_API_KEY='YOUR_ACTUAL_GOOGLE_API_KEY'
            ```
        *(You need to do this in the specific terminal session you use to run the script, or configure it more permanently in your OS)*

6.  **Prepare Exercise GIFs:**
    *   Create a directory named `GIFs` in the project root.
    *   Add the required exercise GIFs to this folder. The script expects filenames matching the `EXERCISE_GIF_MAP` dictionary (e.g., `bicep.gif`, `squats.gif`, etc.).

## Usage

1.  **Ensure Environment Variable is Set:** Make sure `GOOGLE_API_KEY` is set in your current terminal session.
2.  **Run the Script:**
    ```bash
    python your_script_name.py
    ```
    (Replace `your_script_name.py` with the actual name of your Python file, e.g., `llm.py`).
3.  **Initial Setup:**
    *   The application should launch (potentially fullscreen).
    *   Select or CreateX' button to end the session and save stats (webcam only).
9.  **Rest Screen (if applicable):** Wait for the timer or click "Skip Rest". Click 'X' to end the session early.
10. **Chat Screen:** Click "Ask Question", type your query in the popup, and wait for the AI response. Click "Back" to return home.
11. **Quit:** Press the 'Q' key at any time on the main screens (Home, Tracking, etc.) to quit the application.

## Future Enhancements

*   Add tracking for more exercises.
*   Implement more detailed and nuanced form correction rules.
*   Refactor UI using a dedicated GUI toolkit (PyQt, Kivy) for better widgets and layout.
*   Make AI API calls asynchronous to prevent freezing the UI.
*   Save and load entire workout sessions/history.
*   Add audio feedback for reps and form correction.
*   Improve statistics visualization (graphs over time).
*   Add a user settings page (e.g., customize thresholds, colors).

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs or feature suggestions.

## License

*Consider adding a license file (e.g., MIT License) to define how others can use your code.*
