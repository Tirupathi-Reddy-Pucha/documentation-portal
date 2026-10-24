@echo off
title M.A.C.H. Local Web Server
echo ========================================================
echo Starting M.A.C.H. Project Website & Documentation Portal
echo Server running at: http://localhost:8080/
echo ========================================================
start "" "http://localhost:8080/"
node server.js
pause
