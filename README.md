
# Immac Tools

A small collection of ComfyUI extension nodes created for personal use.

## Quick summary

- Package: `immac_tools`
- Purpose: ComfyUI extension providing utility nodes.

## Installation

This repository is intended to be used as a ComfyUI custom node. Install it by one of the following methods:

A) ComfyUI Manager (When/If published)

- Use ComfyUI's built-in extension/manager (if available) to add this repo directly by URL. The manager will clone the repository into the appropriate `custom_nodes` folder and keep it updated.

B) Clone into your ComfyUI `custom_nodes` folder

1. From your ComfyUI repository root (or wherever you keep custom nodes), clone this repo into the `custom_nodes` directory:

```bash
# Example (from ComfyUI repo root)
git clone https://github.com/Immac/immac_tools custom_nodes/immac_tools
```

2. Restart ComfyUI. The extension's `comfy_entrypoint` should be discovered and the nodes will appear in the node list.

## Nodes included

The repository includes the following nodes (see `src/immac_tools/nodes.py` for implementation):

- Concatenate Sigmas Node (`ConcatenateSigmasImmacTools` / display name: `Concatenate Sigmas Node`)
  - Category: Example
  - Inputs: `sigmas_1`, `sigmas_2` (sigma schedules as tensors or lists)
  - Outputs: concatenated sigma schedule
  - Behavior: concatenates two sigma schedules, tolerating None inputs and non-torch objects by converting them to tensors.

- Splice Sigmas At % (`SpliceSigmasAtImmacTools` / display name: `Splice Sigmas At %`)
  - Category: Example
  - Inputs: `sigmas_a`, `sigmas_b`, `splice` (float 0.0 - 1.0)
  - Outputs: `spliced_sigmas` (full), `first_part`, `second_part`
  - Behavior: picks a boundary on `sigmas_a` based on `splice` and combines `sigmas_a` and `sigmas_b` around that boundary. Accepts tensor or list inputs.

## License & contact

MIT — see `LICENSE`. For questions or issues open an issue at the repository: https://github.com/Immac/immac_tools
