#!/usr/bin/env bash
# Zip the Claude skill for upload to claude.ai (Settings, Capabilities, Skills).
set -euo pipefail
cd "$(dirname "$0")/../skills"
python3 ../scripts/gen_docs.py >/dev/null
rm -f ../manimater-components.skill.zip
zip -rq ../manimater-components.skill.zip manimater-components -x '*/__pycache__/*'
echo "wrote manimater-components.skill.zip"
