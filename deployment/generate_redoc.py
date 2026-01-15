#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
import urllib.request
import urllib.error

def generate_redoc(openapi_file, output_dir=None, output_filename=None, standalone=True):
    """
    Generate ReDoc HTML documentation from an OpenAPI JSON file without any external tools.
    
    Args:
        openapi_file (str): Path to the OpenAPI JSON file
        output_dir (str, optional): Directory to save the generated HTML
        output_filename (str, optional): Name of the output HTML file
        standalone (bool): Whether to download the ReDoc script and bundle it
    """
    # Validate input file exists
    openapi_path = Path(openapi_file)
    if not openapi_path.exists():
        raise FileNotFoundError(f"OpenAPI file not found: {openapi_file}")
    
    # Validate it's a valid JSON file and load it
    try:
        with open(openapi_path, 'r', encoding='utf-8') as f:
            spec = json.load(f)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON file: {openapi_file}")
    
    # Set default output directory and filename if not provided
    if not output_dir:
        output_dir = os.getcwd()
    
    if not output_filename:
        output_filename = "redoc.html"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = Path(output_dir) / output_filename
    
    print(f"Generating ReDoc documentation from {openapi_file}...")
    
    # Get ReDoc script if standalone mode is requested
    redoc_script = ""
    if standalone:
        try:
            print("Downloading ReDoc standalone bundle...")
            redoc_url = "https://cdn.jsdelivr.net/npm/redoc@latest/bundles/redoc.standalone.js"
            with urllib.request.urlopen(redoc_url) as response:
                redoc_script = response.read().decode('utf-8')
            print("ReDoc script downloaded successfully.")
        except urllib.error.URLError as e:
            print(f"Warning: Could not download ReDoc script: {e}")
            print("Falling back to CDN version.")
            standalone = False
    
    # Create the HTML content with embedded ReDoc
    if standalone:
        html_template = f"""<!DOCTYPE html>
<html>
  <head>
    <title>API Documentation</title>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://fonts.googleapis.com/css?family=Montserrat:300,400,700|Roboto:300,400,700" rel="stylesheet">
    <style>
      body {{
        margin: 0;
        padding: 0;
      }}
    </style>
  </head>
  <body>
    <div id="redoc-container"></div>
    <script>
{redoc_script}
    </script>
    <script>
      const spec = {json.dumps(spec)};
      Redoc.init(spec, {{
        scrollYOffset: 50
      }}, document.getElementById('redoc-container'))
    </script>
  </body>
</html>
"""
    else:
        html_template = f"""<!DOCTYPE html>
<html>
  <head>
    <title>API Documentation</title>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://fonts.googleapis.com/css?family=Montserrat:300,400,700|Roboto:300,400,700" rel="stylesheet">
    <style>
      body {{
        margin: 0;
        padding: 0;
      }}
    </style>
  </head>
  <body>
    <div id="redoc-container"></div>
    <script src="https://cdn.jsdelivr.net/npm/redoc@latest/bundles/redoc.standalone.js"></script>
    <script>
      const spec = {json.dumps(spec)};
      Redoc.init(spec, {{
        scrollYOffset: 50
      }}, document.getElementById('redoc-container'))
    </script>
  </body>
</html>
"""
    
    # Write the HTML to a file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print(f"ReDoc documentation generated successfully: {output_path}")
    return str(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ReDoc documentation from OpenAPI JSON")
    parser.add_argument("openapi_file", help="Path to the OpenAPI JSON file")
    parser.add_argument("--output-dir", help="Directory to save the generated HTML")
    parser.add_argument("--output-filename", help="Name of the output HTML file")
    parser.add_argument("--no-standalone", action="store_false", dest="standalone", 
                        help="Use CDN for ReDoc script instead of bundling it")
    
    args = parser.parse_args()
    
    try:
        output_path = generate_redoc(
            args.openapi_file, 
            args.output_dir, 
            args.output_filename,
            args.standalone
        )
        print(f"You can view your API documentation by opening: {output_path}")
    except Exception as e:
        print(f"Error: {e}")