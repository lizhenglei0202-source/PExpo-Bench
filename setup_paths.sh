#!/usr/bin/env bash
set -euo pipefail
package_root="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$package_root/analysis_outputs"
echo "Data paths are relative to this package. See REPRODUCE.md for environment setup."
