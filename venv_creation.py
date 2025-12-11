import venv
import subprocess
import os
import sys

# --- Configuration ---
ENV_DIR = ".venv"
REQUIREMENTS_FILE = "requirements.txt"
# ---------------------

def setup_virtual_environment():
    """
    Creates the virtual environment and installs dependencies from 
    requirements.txt using the new environment's pip.
    """
    print(f"--- 🚀 Project Setup Automation ---")
    
    # 1. Create the Virtual Environment
    print(f"1. Creating virtual environment in '{ENV_DIR}'...")
    try:
        # with_pip=True is the default and is required for the next step
        venv.create(ENV_DIR, with_pip=True, clear=False) 
        print(f"   ✅ Environment structure created.")
    except Exception as e:
        print(f"   ❌ ERROR during venv creation: {e}")
        return

    # 2. Determine the path to the new environment's PIP executable
    # OS-independent way to find the executable directory
    bin_dir = "Scripts" if sys.platform.startswith('win') else "bin"
    pip_path = os.path.join(ENV_DIR, bin_dir, "pip")
    
    if not os.path.exists(pip_path):
        print(f"   ❌ ERROR: Could not find pip executable at '{pip_path}'. Aborting installation.")
        return

    # 3. Install dependencies using the new environment's pip
    if not os.path.exists(REQUIREMENTS_FILE):
        print(f"2. Skipping installation: '{REQUIREMENTS_FILE}' not found.")
        return

    print(f"2. Installing dependencies from '{REQUIREMENTS_FILE}'...")
    install_command = [pip_path, "install", "-r", REQUIREMENTS_FILE]
    
    try:
        # Execute the installation command using subprocess
        subprocess.run(
            install_command, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        print(f"   ✅ All packages installed successfully.")
        
    except subprocess.CalledProcessError as e:
        print(f"   ❌ ERROR during package installation:")
        print(f"   {e.stderr.strip()}")
        print("   Please check your requirements file for errors.")
        return
        
    print(f"\n--- 🎉 Setup Complete ---")
    print("ACTION REQUIRED: You must select the Python interpreter in VS Code (Ctrl+Shift+P and search for 'Python: Select Interpreter').")


if __name__ == "__main__":
    setup_virtual_environment()