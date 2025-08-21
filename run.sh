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