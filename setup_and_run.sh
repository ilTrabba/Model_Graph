#!/bin/bash

# Stop on any error
set -e

# Define the global virtual environment name
GLOBAL_VENV_NAME="ModelHeritageEnv"

# --- Global Virtual Environment Setup ---
echo "Creating and activating global virtual environment: $GLOBAL_VENV_NAME"
python3 -m venv "$GLOBAL_VENV_NAME"
source "$GLOBAL_VENV_NAME"/bin/activate

# --- Backend Setup ---
echo "Setting up backend..."
cd model_heritage_backend

# Generate requirements.txt for the backend
pip freeze > requirements.txt

pip install -r requirements.txt
pip install flask-cors  # Ensure flask-cors is installed

# Create database if it doesn't exist (with safe error handling)
echo "Checking database setup..."
python -c "
try:
    from src.main import app
    app.app_context().push()
    print('App context loaded successfully')
    try:
        from src.models.user import db
        db.create_all()
        print('Database OK')
    except ImportError:
        print('Database models not found - skipping database initialization')
    except Exception as e:
        print(f'Database setup error: {e} - continuing without database')
except ImportError as e:
    print(f'Main app import error: {e} - skipping database setup')
except Exception as e:
    print(f'Unexpected error: {e} - continuing anyway')
"

cd .. # Go back to project root

# --- Frontend Setup ---
echo "Setting up frontend..."
cd model_heritage_frontend
pnpm install

# Configure API URL for frontend
echo "VITE_API_URL=http://localhost:5001" > .env.local
cd .. # Go back to project root

echo "Setup complete. Now running the project..."

# --- Run Backend in background ---
echo "Starting backend..."
cd model_heritage_backend
source "../$GLOBAL_VENV_NAME"/bin/activate # Activate global venv
python run_server.py &
BACKEND_PID=$!
cd ..

# --- Run Frontend in background ---
echo "Starting frontend..."
cd model_heritage_frontend
source "../$GLOBAL_VENV_NAME"/bin/activate # Activate global venv (optional for pnpm, but good practice)
pnpm run dev --host &
FRONTEND_PID=$!
cd ..

echo "Project is running. Backend on http://localhost:5001, Frontend on http://localhost:5173"
echo "Press Ctrl+C to stop both processes."

# Function to kill background processes on exit
cleanup() {
    echo "Stopping backend (PID: $BACKEND_PID) and frontend (PID: $FRONTEND_PID)..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    echo "Processes stopped."
}

# Trap Ctrl+C and call cleanup function
trap cleanup SIGINT

# Keep the script running until interrupted
wait $BACKEND_PID
wait $FRONTEND_PID