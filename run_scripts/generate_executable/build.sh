#!/bin/bash
# Usage: ./build.sh --run-script <run_script> --src-folder <src_folder> --output-folder <output_folder>

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --run-script) RUN_SCRIPT="$2"; shift ;;
        --src-folder) SRC_FOLDER="$2"; shift ;;
        --output-folder) OUTPUT_FOLDER="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Validate required parameters
if [ -z "$RUN_SCRIPT" ] || [ -z "$SRC_FOLDER" ] || [ -z "$OUTPUT_FOLDER" ]; then
    echo "Usage: $0 --run-script <run_script> --src-folder <src_folder> --output-folder <output_folder>"
    exit 1
fi

echo "Executing command:"
COMMAND="pyinstaller --onefile --distpath \"$OUTPUT_FOLDER\" --paths \"$SRC_FOLDER\" \"$RUN_SCRIPT\""
echo $COMMAND

# Run the PyInstaller command
eval $COMMAND
if [ $? -ne 0 ]; then
    echo "Error: PyInstaller did not complete successfully."
    exit 1
else
    echo "Executable successfully created in: $OUTPUT_FOLDER"
fi
