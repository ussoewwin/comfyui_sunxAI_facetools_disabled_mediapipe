# comfyui_sunxAI_facetools

Face detection & restoration tools for ComfyUI by Sunx.ai

> [!NOTE]
> This projected was created with a [cookiecutter](https://github.com/Comfy-Org/cookiecutter-comfy-extension) template. It helps you start writing custom nodes without worrying about the Python setup.

## Quickstart

1. Install [ComfyUI](https://docs.comfy.org/get_started).
1. Install [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)
1. Look up this extension in ComfyUI-Manager. If you are installing manually, clone this repository under `ComfyUI/custom_nodes`.
1. Restart ComfyUI.

# Features

- **DetectFaces**: Detects faces in images and returns the face with the largest area
- **DetectFaceByIndex**: Detects faces by index, supporting left-to-right selection of specific faces (0=leftmost, 1=second, etc.), with gender filtering support (0=any gender, 1=male, 2=female). Gender detection is only enabled when needed for improved performance.
- **DetectFaceByGender**: Detects faces by gender and index, supporting filtering of male/female faces and selection in left-to-right order
- **CropFaces**: Crops detected face regions
- **WarpFaceBack**: Warps processed faces back to the original image
- **InstantID**: Face identity preservation functionality
- **ColorAdjust**: Face color adjustment
- **SaveImageWebsocket**: Saves images via WebSocket

## Installation Dependencies

This extension uses InsightFace for high-precision gender detection (replacing mediapipe for Python 3.13 compatibility). Models will be automatically downloaded on first use:

### Standard Installation (Python 3.10-3.12)

```bash
pip install insightface
```

### Python 3.13 Users (Windows Only)

**⚠️ Important**: The standard `pip install insightface` does not provide Python 3.13 wheels. Windows users running Python 3.13 must download the pre-built wheel from:

**https://huggingface.co/ussoewwin/Insightface_for_windows/tree/main**

Download the appropriate wheel file (`insightface-0.7.3-cp313-cp313-win_amd64.whl`) and install it directly:

```bash
pip install insightface-0.7.3-cp313-cp313-win_amd64.whl
```

**Note**: This is Windows-only. Linux/macOS Python 3.13 users will need to build from source or use Python 3.12 or earlier.

**Note**: InsightFace requires additional model files that will be automatically downloaded on first run.

### Gender Detection Features
- Uses InsightFace for high-precision gender recognition
- Supports both GPU and CPU modes with automatic device selection
- Provides age detection and confidence information
- Supports fallback mechanism (based on facial aspect ratio)
- Prevents division by zero errors to ensure stable operation

## Develop

To install the dev dependencies and pre-commit (will run the ruff hook), do:

```bash
cd comfyui_sunxAI_facetools
pip install -e .[dev]
pre-commit install
```

The `-e` flag above will result in a "live" install, in the sense that any changes you make to your node extension will automatically be picked up the next time you run ComfyUI.

## Publish to Github

Install Github Desktop or follow these [instructions](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) for ssh.

1. Create a Github repository that matches the directory name.
2. Push the files to Git
```
git add .
git commit -m "project scaffolding"
git push
```

## Writing custom nodes

An example custom node is located in [node.py](src/comfyui_sunxAI_facetools/nodes.py). To learn more, read the [docs](https://docs.comfy.org/essentials/custom_node_overview).


## Tests

This repo contains unit tests written in Pytest in the `tests/` directory. It is recommended to unit test your custom node.

- [build-pipeline.yml](.github/workflows/build-pipeline.yml) will run pytest and linter on any open PRs
- [validate.yml](.github/workflows/validate.yml) will run [node-diff](https://github.com/Comfy-Org/node-diff) to check for breaking changes

## Publishing to Registry

If you wish to share this custom node with others in the community, you can publish it to the registry. We've already auto-populated some fields in `pyproject.toml` under `tool.comfy`, but please double-check that they are correct.

You need to make an account on https://registry.comfy.org and create an API key token.

- [ ] Go to the [registry](https://registry.comfy.org). Login and create a publisher id (everything after the `@` sign on your registry profile).
- [ ] Add the publisher id into the pyproject.toml file.
- [ ] Create an api key on the Registry for publishing from Github. [Instructions](https://docs.comfy.org/registry/publishing#create-an-api-key-for-publishing).
- [ ] Add it to your Github Repository Secrets as `REGISTRY_ACCESS_TOKEN`.

A Github action will run on every git push. You can also run the Github action manually. Full instructions [here](https://docs.comfy.org/registry/publishing). Join our [discord](https://discord.com/invite/comfyorg) if you have any questions!

