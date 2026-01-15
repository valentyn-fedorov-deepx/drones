Often we need to provide python executables for clients of our client, and this is why this script exists. Using pyinstaller we can archive the full python environment to a single executable file or directory that includes all needed dependencies. For this, you should use the build.py defined here. This script will use the python env from which you've started the script.

A couple of notes. The more dependencies, the more issues may come up in the build. What I've encountered so far is:
When building with norfair library, all its dependencies weren't bundled, so you will need to specify `--copy-metadata norfair`.   
When building tensorrt, the same thing happens, not all .dll files are copied to the build, and `--copy-metadata` doesn't help. But what helps is manually finding the .dll files in the `.env/lib/tensorrt*` libraries and copying them to the same directory where the generated executable is.

If you do not have direct access to the device you are building, you can use Docker with the correct image. For example, to build for jetpack 5, I've used `nvcr.io/nvidia/l4t-jetpack:r35.4.1` in a separate Docker profile.

`docker run --rm --privileged tonistiigi/binfmt --install arm64`     
`docker buildx create --use --name jetson || docker buildx use jetson`   
`docker buildx build --platform linux/arm64`   

A Dockerfile will change from project to project, but you certainly will need to install python.

`RUN apt-get update && apt-get install -y --no-install-recommends \`    
`    python3 python3-pip python3-venv python3-dev \`    
`    build-essential patchelf upx \`    
`    libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 \`  
`    && rm -rf /var/lib/apt/lists/*`
