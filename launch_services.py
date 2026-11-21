import os
import sys
import subprocess
import time
import webbrowser

ROOT_DIR = r"C:\Users\hy180\Downloads\ASD-Facial-CNN"
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
BACKEND_DIR = os.path.join(ROOT_DIR, "backend-spring")

CREATE_NEW_CONSOLE = 0x00000010

print("Launching 1. PyTorch FastAPI ML Engine on Port 8000...")
p_python = subprocess.Popen(
    ["cmd.exe", "/k", "title PyTorch FastAPI Engine (:8000) && python -m uvicorn src.api.server:app --host 127.0.0.1 --port 8000 --reload"],
    cwd=ROOT_DIR,
    creationflags=CREATE_NEW_CONSOLE
)
print(f"FastAPI started with PID: {p_python.pid}")

time.sleep(2)

print("Launching 2. Java Spring Boot 3 Gateway on Port 8080...")
p_spring = subprocess.Popen(
    ["cmd.exe", "/k", "title Spring Boot Gateway (:8080) && mvnw.cmd spring-boot:run -Dspring-boot.run.profiles=h2"],
    cwd=BACKEND_DIR,
    creationflags=CREATE_NEW_CONSOLE
)
print(f"Spring Boot started with PID: {p_spring.pid}")

time.sleep(2)

print("Launching 3. React 18 Frontend on Port 5173...")
p_frontend = subprocess.Popen(
    ["cmd.exe", "/k", "title React 18 Clinician Dashboard (:5173) && npm.cmd run dev"],
    cwd=FRONTEND_DIR,
    creationflags=CREATE_NEW_CONSOLE
)
print(f"Frontend started with PID: {p_frontend.pid}")

time.sleep(4)

print("Opening browser to http://localhost:5173/ ...")
webbrowser.open("http://localhost:5173/")
webbrowser.open("http://127.0.0.1:8000/docs")

print("All 3 services successfully launched in dedicated console windows!")
