#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 crenaz
#
# Fetch high-accuracy Tesseract language models into a project-local tessdata/
# directory.
#
# Ubuntu's tesseract-ocr-* packages ship the standard `tessdata` models. Upstream
# also publishes `tessdata_best`, which is slower but measurably more accurate --
# on this project's sample Spanish page it scored 64.47% vs 54.92% average
# confidence. Keeping the models project-local means the pipeline does not depend
# on which language packs happen to be installed system-wide, and survives a fresh
# clone or a container build.
#
# Usage:
#   ./scripts/fetch_tessdata.sh          # download anything missing
#   ./scripts/fetch_tessdata.sh --force  # re-download everything

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TESSDATA_DIR="${REPO_ROOT}/tessdata"
BASE_URL="https://github.com/tesseract-ocr/tessdata_best/raw/main"

# eng+spa are the pipeline default (--lang spa+eng); osd enables orientation and
# script detection, which --psm 0 and the deskew work both rely on.
LANGS=(eng spa osd)

# A truncated or HTML-error download is the common failure here, so require a
# plausible floor rather than trusting exit status alone.
MIN_BYTES=1000000

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

mkdir -p "${TESSDATA_DIR}"
echo "Fetching tessdata_best models into ${TESSDATA_DIR}"

for lang in "${LANGS[@]}"; do
    target="${TESSDATA_DIR}/${lang}.traineddata"

    if [[ -f "${target}" && ${FORCE} -eq 0 ]]; then
        echo "  ${lang}: already present ($(du -h "${target}" | cut -f1)), skipping"
        continue
    fi

    echo -n "  ${lang}: downloading... "
    tmp="${target}.partial"
    if ! curl -sSL --fail --max-time 300 -o "${tmp}" "${BASE_URL}/${lang}.traineddata"; then
        rm -f "${tmp}"
        echo "FAILED (download error)"
        exit 1
    fi

    size=$(stat -c %s "${tmp}")
    if (( size < MIN_BYTES )); then
        rm -f "${tmp}"
        echo "FAILED (got ${size} bytes, expected at least ${MIN_BYTES})"
        exit 1
    fi

    mv "${tmp}" "${target}"
    echo "done ($(du -h "${target}" | cut -f1))"
done

echo
echo "Done. Verify with:"
echo "  TESSDATA_PREFIX=${TESSDATA_DIR} tesseract --list-langs"
