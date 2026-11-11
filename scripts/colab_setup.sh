#!/usr/bin/env bash
# Install everything needed by notebooks/colab_end_to_end.ipynb.
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install --upgrade -r requirements-colab.txt

# IndicTransToolkit occasionally needs a source build (it ships Cython code).
if ! python -c "import IndicTransToolkit" >/dev/null 2>&1; then
  if ! python -m pip install "IndicTransToolkit==1.1.1"; then
    echo "Wheel install failed; building IndicTransToolkit from source ..."
    rm -rf /tmp/IndicTransToolkit
    git clone --depth 1 https://github.com/VarunGumma/IndicTransToolkit.git /tmp/IndicTransToolkit
    python -m pip install --editable /tmp/IndicTransToolkit --no-build-isolation
  fi
fi

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "Setup complete. Open notebooks/colab_end_to_end.ipynb."
