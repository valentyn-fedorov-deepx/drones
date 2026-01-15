import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Generate a standalone .exe from a Python run script using PyInstaller."
    )
    parser.add_argument(
        "--run-script", type=str, required=True,
        help="Path to the run script"
    )
    parser.add_argument(
        "--src-folder", type=str, required=True,
        help="Path to the folder containing the imported modules (e.g., dx_vyzai_people_track)"
    )
    parser.add_argument(
        "--output-folder", type=str, required=True,
        help="Destination folder where the .exe will be generated"
    )
    parser.add_argument(
        "--onefile", action="store_true",
        help="Create a one-file bundled executable"
    )
    parser.add_argument(
        "--pyinstaller-args", nargs="*", default=[],
        help="Additional arguments to pass to PyInstaller (e.g., --noconsole --icon=myicon.ico)"
    )

    args = parser.parse_args()

    python_executable_path = Path(sys.executable)
    pyinstaller_executable_path = python_executable_path.parent / "pyinstaller"

    print(f"Using pyinstaller from: {pyinstaller_executable_path}")

    # Build the PyInstaller command.
    command = [
        str(pyinstaller_executable_path),
        "--distpath", args.output_folder,
        "--paths", args.src_folder,
    ]

    # Add --onefile if specified
    if args.onefile:
        command.append("--onefile")

    # Add any additional PyInstaller arguments
    if args.pyinstaller_args:
        command.extend(args.pyinstaller_args)

    # Add the run script as the last argument
    command.append(args.run_script)

    print("Executing command:")
    print(" ".join(command))

    # Execute the command
    result = subprocess.run(command)

    if result.returncode != 0:
        print("Error: PyInstaller did not complete successfully.")
        sys.exit(result.returncode)
    else:
        print(f"Executable successfully created in: {args.output_folder}")


if __name__ == "__main__":
    main()
