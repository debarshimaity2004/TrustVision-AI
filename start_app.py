import subprocess
import sys
import os
import signal
import time

def terminate_processes(processes):
    print("\nShutting down all services...")
    for p in processes:
        try:
            p.terminate()
        except:
            pass
    print("Goodbye!")
    sys.exit(0)

if __name__ == "__main__":
    print("===========================================")
    print("   Starting TrustVision-AI Full Stack      ")
    print("===========================================")

    root_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_dir = os.path.join(root_dir, "Frontend")

    processes = []

    try:
        # 1. Start the Backend API (FastAPI)
        print("\n[>>] Starting Python Backend API on port 8000...")
        backend_process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "Backend.main:app", "--reload"],
            cwd=root_dir
        )
        processes.append(backend_process)

        # Give backend a moment to initialize the ML models
        time.sleep(3)

        # 2. Start the Frontend Application (Vite/React)
        print("\n[>>] Starting React Frontend on port 5173...")
        # We use shell=True on Windows for npm commands to resolve correctly
        frontend_process = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=frontend_dir,
            shell=True
        )
        processes.append(frontend_process)

        print("\n===========================================")
        print(" All services are running sequentially!")
        print(" Frontend UI: http://localhost:5173")
        print(" Backend API: http://localhost:8000")
        print(" Press Ctrl+C in this terminal to stop all.")
        print("===========================================\n")

        # Keep the main script alive to listen for Ctrl+C
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        terminate_processes(processes)
    except Exception as e:
        print(f"Failed to start services: {e}")
        terminate_processes(processes)
