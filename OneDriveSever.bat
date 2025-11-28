@echo off
call .\LearningColorConstancy_env\Scripts\activate.bat

cd src\OneDriveServer
python OneDriveServer.py

call deactivate 

pause