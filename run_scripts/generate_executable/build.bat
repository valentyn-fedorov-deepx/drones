@echo off
REM Usage: build.bat <run_script> <src_folder> <output_folder>
IF "%~3"=="" (
    echo Usage: %0 run_script src_folder output_folder
    exit /b 1
)

set "RUN_SCRIPT=%~1"
set "SRC_FOLDER=%~2"
set "OUTPUT_FOLDER=%~3"

echo Executing command:
echo pyinstaller --onefile --distpath "%OUTPUT_FOLDER%" --paths "%SRC_FOLDER%" "%RUN_SCRIPT%"

pyinstaller --onefile --distpath "%OUTPUT_FOLDER%" --paths "%SRC_FOLDER%" "%RUN_SCRIPT%"
IF ERRORLEVEL 1 (
    echo Error: PyInstaller did not complete successfully.
    pause
    exit /b 1
) ELSE (
    echo Executable successfully created in: %OUTPUT_FOLDER%
)
pause
