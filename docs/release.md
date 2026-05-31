# Release Process

1. Run tests.

```bash
pytest
```

2. Export generator-only weights.

```bash
python scripts/export_inference_weights.py \
  --checkpoint path/to/best.pt \
  --output-dir release_assets \
  --include-fp16
```

3. Upload these files to GitHub Release `v0.1.0`.

```text
rgb2nir-pix2pix-generator-fp32.safetensors
rgb2nir-pix2pix-generator-fp16.safetensors
checksums.sha256
weights_manifest.json
```

4. Update `scripts/download_weights.py` with the exported SHA256 values before
publishing the release.
