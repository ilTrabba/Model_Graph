import subprocess
import sys
import os
import signal
import time
from typing import Optional

# Define the global virtual environment name
GLOBAL_VENV_NAME = "ModelHeritageEnv"

# Colors for better readability
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'

# Function to print colored messages
def print_status(message: str):
    print(f"{Colors.BLUE}[{time.strftime('%H:%M:%S')}]{Colors.NC} {message}")

def print_success(message: str):
    print(f"{Colors.GREEN}✅ [{time.strftime('%H:%M:%S')}]{Colors.NC} {message}")

def print_warning(message: str):
    print(f"{Colors.YELLOW}⚠️  [{time.strftime('%H:%M:%S')}]{Colors.NC} {message}")

def print_error(message: str):
    print(f"{Colors.RED}❌ [{time.strftime('%H:%M:%S')}]{Colors.NC} {message}")

def print_info(message: str):
    print(f"{Colors.CYAN}ℹ️  [{time.strftime('%H:%M:%S')}]{Colors.NC} {message}")

# Header
print(f"{Colors.PURPLE}================================={Colors.NC}")
print(f"{Colors.PURPLE}    Model Heritage Project{Colors.NC}")
print(f"{Colors.PURPLE}================================={Colors.NC}")
print("")

# Function to determine the correct Python executable path inside the venv
def get_python_executable(venv_name: str) -> str:
    """Returns the path to the python executable within a virtual environment."""
    if sys.platform == "win32":
        return os.path.join("..", venv_name, "Scripts", "python.exe")
    else: # Linux and macOS
        return os.path.join("..", venv_name, "bin", "python")

# Check if virtual environment exists
if not os.path.isdir(GLOBAL_VENV_NAME):
    print_error(f"Virtual environment '{GLOBAL_VENV_NAME}' not found!")
    print_info(f"Please create it first: python -m venv {GLOBAL_VENV_NAME}")
    sys.exit(1)

print_success(f"Virtual environment '{GLOBAL_VENV_NAME}' found")

backend_proc: Optional[subprocess.Popen] = None
frontend_proc: Optional[subprocess.Popen] = None

def cleanup():
    """Function to kill background processes on exit."""
    global backend_proc, frontend_proc
    print_status("Shutting down services...")

    if backend_proc and backend_proc.poll() is None:
        print_status(f"Stopping backend (PID: {backend_proc.pid})...")
        backend_proc.terminate()
        try:
            backend_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            backend_proc.kill()
        print_success("Backend stopped")
    else:
        print_info("Backend already stopped")

    if frontend_proc and frontend_proc.poll() is None:
        print_status(f"Stopping frontend (PID: {frontend_proc.pid})...")
        frontend_proc.terminate()
        try:
            frontend_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            frontend_proc.kill()
        print_success("Frontend stopped")
    else:
        print_info("Frontend already stopped")
    
    print("")
    print_success("All services stopped successfully")
    print(f"{Colors.PURPLE}Thanks for using Model Heritage Project!{Colors.NC}")
    sys.exit(0)

def signal_handler(sig, frame):
    """Handle Ctrl+C signal."""
    print("")
    print_warning("Ctrl+C detected, initiating shutdown...")
    cleanup()

# Trap Ctrl+C and call cleanup function
signal.signal(signal.SIGINT, signal_handler)

try:
    # --- Run Backend in background ---
    print_status("Starting backend server...")
    if not os.path.exists("model_heritage_backend/run_server.py"):
        print_error("Backend server file 'run_server.py' not found!")
        sys.exit(1)

    backend_venv_path = get_python_executable(GLOBAL_VENV_NAME)
    print_info("Activated virtual environment for backend")
    
    backend_proc = subprocess.Popen(
        [backend_venv_path, "run_server.py"],
        cwd="model_heritage_backend"
    )

    if backend_proc.poll() is None:
        print_success(f"Backend started successfully (PID: {backend_proc.pid})")
    else:
        print_error("Failed to start backend")
        cleanup()
        sys.exit(1)

    # --- Run Frontend in background ---
    print_status("Starting frontend server...")
    if not os.path.exists("model_heritage_frontend/package.json"):
        print_error("Frontend package.json not found!")
        cleanup()
        sys.exit(1)

    frontend_proc = subprocess.Popen(
        ["pnpm", "run", "dev", "--host"],
        cwd="model_heritage_frontend"
    )

    time.sleep(2)  # Give frontend time to start

    if frontend_proc.poll() is None:
        print_success(f"Frontend started successfully (PID: {frontend_proc.pid})")
    else:
        print_error("Failed to start frontend")
        cleanup()
        sys.exit(1)

    # Success message
    print("")
    print(f"{Colors.GREEN}🚀 ================================={Colors.NC}")
    print(f"{Colors.GREEN}    PROJECT IS RUNNING!{Colors.NC}")
    print(f"{Colors.GREEN}================================={Colors.NC}")
    print(f"{Colors.CYAN}📡 Backend:{Colors.NC}  http://localhost:5001")
    print(f"{Colors.CYAN}🌐 Frontend:{Colors.NC} http://localhost:5173")
    
    try:
        host_ip = subprocess.check_output("hostname -I | awk '{print $1}'", shell=True, text=True).strip()
        print(f"{Colors.CYAN}🌍 Network:{Colors.NC}  http://{host_ip}:5173")
    except Exception:
        print_warning("Could not determine network IP address.")

    print("")
    print_warning("Press Ctrl+C to stop both processes")
    print("")

    # Keep the script running until interrupted
    while backend_proc.poll() is None or frontend_proc.poll() is None:
        time.sleep(1)

except Exception as e:
    print_error(f"An unexpected error occurred: {e}")
    cleanup()
    sys.exit(1)
finally:
    # Ensure cleanup is called if the script exits unexpectedly
    if backend_proc or frontend_proc:
        cleanup()
