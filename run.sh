#!/bin/bash

# Define the global virtual environment name
GLOBAL_VENV_NAME="ModelHeritageEnv"

# Colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored messages
print_status() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✅ [$(date +'%H:%M:%S')]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️  [$(date +'%H:%M:%S')]${NC} $1"
}

print_error() {
    echo -e "${RED}❌ [$(date +'%H:%M:%S')]${NC} $1"
}

print_info() {
    echo -e "${CYAN}ℹ️  [$(date +'%H:%M:%S')]${NC} $1"
}

# Header
echo -e "${PURPLE}=================================${NC}"
echo -e "${PURPLE}    Model Heritage Project${NC}"
echo -e "${PURPLE}=================================${NC}"
echo ""

# Check if virtual environment exists
if [ ! -d "$GLOBAL_VENV_NAME" ]; then
    print_error "Virtual environment '$GLOBAL_VENV_NAME' not found!"
    print_info "Please create it first: python -m venv $GLOBAL_VENV_NAME"
    exit 1
fi

print_success "Virtual environment '$GLOBAL_VENV_NAME' found"

# --- Run Backend in background ---
print_status "Starting backend server..."
cd model_heritage_backend

if [ ! -f "run_server.py" ]; then
    print_error "Backend server file 'run_server.py' not found!"
    cd ..
    exit 1
fi

source "../$GLOBAL_VENV_NAME"/bin/activate
print_info "Activated virtual environment for backend"

python run_server.py &
BACKEND_PID=$!
cd ..

if kill -0 $BACKEND_PID 2>/dev/null; then
    print_success "Backend started successfully (PID: $BACKEND_PID)"
else
    print_error "Failed to start backend"
    exit 1
fi

# --- Run Frontend in background ---
print_status "Starting frontend server..."
cd model_heritage_frontend

if [ ! -f "package.json" ]; then
    print_error "Frontend package.json not found!"
    cd ..
    cleanup
    exit 1
fi

source "../$GLOBAL_VENV_NAME"/bin/activate
print_info "Activated virtual environment for frontend"

pnpm run dev --host &
FRONTEND_PID=$!
cd ..

sleep 2 # Give frontend time to start

if kill -0 $FRONTEND_PID 2>/dev/null; then
    print_success "Frontend started successfully (PID: $FRONTEND_PID)"
else
    print_error "Failed to start frontend"
    cleanup
    exit 1
fi

# Success message
echo ""
echo -e "${GREEN}🚀 =================================${NC}"
echo -e "${GREEN}    PROJECT IS RUNNING!${NC}"
echo -e "${GREEN}=================================${NC}"
echo -e "${CYAN}📡 Backend:${NC}  http://localhost:5001"
echo -e "${CYAN}🌐 Frontend:${NC} http://localhost:5173"
echo -e "${CYAN}🌍 Network:${NC}  http://$(hostname -I | awk '{print $1}'):5173"
echo ""
print_warning "Press Ctrl+C to stop both processes"
echo ""

# Function to kill background processes on exit
cleanup() {
    echo ""
    print_status "Shutting down services..."
    
    if kill -0 $BACKEND_PID 2>/dev/null; then
        print_status "Stopping backend (PID: $BACKEND_PID)..."
        kill $BACKEND_PID 2>/dev/null
        print_success "Backend stopped"
    else
        print_info "Backend already stopped"
    fi
    
    if kill -0 $FRONTEND_PID 2>/dev/null; then
        print_status "Stopping frontend (PID: $FRONTEND_PID)..."
        kill $FRONTEND_PID 2>/dev/null
        print_success "Frontend stopped"
    else
        print_info "Frontend already stopped"
    fi
    
    echo ""
    print_success "All services stopped successfully"
    echo -e "${PURPLE}Thanks for using Model Heritage Project!${NC}"
    exit 0
}

# Trap Ctrl+C and call cleanup function
trap cleanup SIGINT

# Keep the script running until interrupted
wait $BACKEND_PID
wait $FRONTEND_PID