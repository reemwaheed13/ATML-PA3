#!/usr/bin/env bash
# Download the standard public COCO-2014 + VQAv2 files Task 2 needs, into DRIVE_ROOT/data.
# (handout p7-p8). Run once on Colab after colab/setup.ipynb has set DRIVE_ROOT.
#
#   DRIVE_ROOT=/content/drive/MyDrive/ATML-PA3 bash scripts/download_task2_data.sh
#
# Produces under $DRIVE_ROOT/data:
#   train2014/ val2014/                                  (images)
#   captions_{train,val}2014.json instances_{train,val}2014.json
#   v2_OpenEnded_mscoco_{train,val}2014_questions.json
#   v2_mscoco_{train,val}2014_annotations.json
#   v2_mscoco_{train,val}2014_complementary_pairs.json
# Idempotent: skips any target that already exists. Large (~19 GB images); images are optional if
# features are cached once, but the annotation JSONs are always needed for selection.
set -euo pipefail

: "${DRIVE_ROOT:?set DRIVE_ROOT (e.g. /content/drive/MyDrive/ATML-PA3) before running}"
DATA="${DRIVE_ROOT}/data"
mkdir -p "${DATA}"
cd "${DATA}"

COCO_IMG="http://images.cocodataset.org/zips"
COCO_ANN="http://images.cocodataset.org/annotations"
VQA="https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa"

# $1 = url, $2 = a sentinel path that exists after a successful unzip (skip if present)
fetch_unzip() {
  local url="$1" sentinel="$2" zip
  zip="$(basename "${url}")"
  if [[ -e "${sentinel}" ]]; then
    echo "[skip] ${sentinel} already present"
    return
  fi
  echo "[get ] ${url}"
  wget -q --show-progress -c "${url}" -O "${zip}"
  echo "[unzip] ${zip}"
  unzip -q -o "${zip}"
  rm -f "${zip}"
}

# --- images (large; comment these two out if you only cache features elsewhere) ---
fetch_unzip "${COCO_IMG}/train2014.zip" "train2014"
fetch_unzip "${COCO_IMG}/val2014.zip"   "val2014"

# --- COCO caption + instance annotations (unzip into annotations/, then flatten) ---
if [[ ! -e "captions_train2014.json" ]]; then
  fetch_unzip "${COCO_ANN}/annotations_trainval2014.zip" "annotations/captions_train2014.json"
  mv -n annotations/*.json "${DATA}/" || true
  rmdir annotations 2>/dev/null || true
fi

# --- VQAv2 questions / annotations / complementary pairs ---
fetch_unzip "${VQA}/v2_Questions_Train_mscoco.zip"          "v2_OpenEnded_mscoco_train2014_questions.json"
fetch_unzip "${VQA}/v2_Questions_Val_mscoco.zip"            "v2_OpenEnded_mscoco_val2014_questions.json"
fetch_unzip "${VQA}/v2_Annotations_Train_mscoco.zip"        "v2_mscoco_train2014_annotations.json"
fetch_unzip "${VQA}/v2_Annotations_Val_mscoco.zip"          "v2_mscoco_val2014_annotations.json"
fetch_unzip "${VQA}/v2_Complementary_Pairs_Train_mscoco.zip" "v2_mscoco_train2014_complementary_pairs.json"
fetch_unzip "${VQA}/v2_Complementary_Pairs_Val_mscoco.zip"  "v2_mscoco_val2014_complementary_pairs.json"

echo "[done] Task-2 data in ${DATA}"
