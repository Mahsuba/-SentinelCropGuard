━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SentinelCropGuard — How to Run (Read this carefully!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This app works COMPLETELY by itself.
You do NOT need to download anything from Colab.
You do NOT need internet (except to see the map tiles).
You only need to do the steps below ONCE.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — Install Python (only do this once)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. Go to: https://www.python.org/downloads/
  2. Click the big yellow "Download Python 3.12.x" button
  3. Run the installer
  4. IMPORTANT: on the first screen, tick the box that says
     "Add Python to PATH"  ← this is very important!
  5. Click "Install Now"
  6. Wait for it to finish

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — Put this folder somewhere easy to find
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Make sure the folder "SentinelCropGuard" is on your Desktop
  or in your Documents folder. It should contain:

    SentinelCropGuard/
        app.py
        requirements.txt
        README.txt   ← this file

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — Open VS Code
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. Open VS Code
  2. Click "File" → "Open Folder"
  3. Find and select the "SentinelCropGuard" folder
  4. Click "Select Folder"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 4 — Open the Terminal in VS Code
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  At the top of VS Code, click:
  Terminal → New Terminal

  A black/dark box will appear at the bottom of VS Code.
  That is your terminal. You will type commands there.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 5 — Install packages (only do this once!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  In the terminal, type this EXACTLY and press Enter:

    pip install -r requirements.txt

  Wait for it to finish. It will take 1-3 minutes.
  You will see a lot of text — that is normal.
  When it says "Successfully installed" you are done.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 6 — Run the app!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  In the terminal, type this and press Enter:

    streamlit run app.py

  Wait about 5 seconds.
  Your web browser will open automatically with the app!

  If the browser does NOT open automatically:
  Look in the terminal for a line that says:
      Local URL: http://localhost:8501
  Copy that address and paste it into your browser.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVERY TIME AFTER THAT (next day, next week...)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  You only need to do Steps 3, 4, and 6.
  You do NOT need to install packages again.

  Quick reminder:
    1. Open VS Code
    2. Open the SentinelCropGuard folder
    3. Terminal → New Terminal
    4. Type:  streamlit run app.py
    5. Done!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TO STOP THE APP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Click inside the terminal and press:  Ctrl + C
  The app will stop.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMMON PROBLEMS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Problem: "pip is not recognized"
  Fix: You forgot to tick "Add Python to PATH" during install.
       Uninstall Python and reinstall, this time tick that box.

  Problem: "streamlit is not recognized"
  Fix: Run this first:  pip install streamlit
       Then try again.

  Problem: "ModuleNotFoundError"
  Fix: Run:  pip install -r requirements.txt
       again and wait for it to finish.

  Problem: The map is not showing
  Fix: You need internet for the map background tiles.
       The rest of the app works offline.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
