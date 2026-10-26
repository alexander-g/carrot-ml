#!/bin/bash

set -ex


if [[ "${GITHUB_ACTIONS:-}" != "true" ]]; then
  echo "This script is meant to run in github actions only. Will delete modified files."
  exit 0
fi


if [[ -z "${MODEL_DIR+x}" ]]; then
  echo "Error: MODEL_DIR is not set" >&2
  exit 1
fi


GIT="git -C ${MODEL_DIR}"



rc=0
# iterate over all tagged models in the model repo for backward compatibility
for tag in $($GIT tag); do
    echo $tag
    $GIT checkout --force $tag
    $GIT clean --force -d
    ls -halt $MODEL_DIR

    # only one modelfile
    python -c "import glob; assert len(glob.glob('./${MODEL_DIR}/*.pt.zip')) == 1"
    current_ptzip=$(ls $MODEL_DIR/*.pt.zip)

    python update-yolo.py --model=$current_ptzip || rc=1 && continue

    # use the most recent .pt.zip
    updated_ptzip=$(realpath $(ls -t -- $MODEL_DIR/*.pt.zip | head -n 1))

    # smoke test
    echo "TODO: add an actual image file with cells and treerings"
    python inference.py \
            --model=${updated_ptzip} \
            --input=tests/assets/black.png \
            --px-per-mm=1000 \
            --output=/tmp/DELETE || rc=1
    
    echo "*********"
done;






# again with the most recent model
$GIT checkout --force release-candidate || $GIT checkout --force main
$GIT clean --force -d

ls -halt $MODEL_DIR
# only one modelfile
python -c "import glob; assert len(glob.glob('./${MODEL_DIR}/*.pt.zip')) == 1"
current_ptzip=$(ls $MODEL_DIR/*.pt.zip)

python update-yolo.py --model=$current_ptzip

# use the most recent .pt.zip
updated_ptzip=$(realpath $(ls -t -- $MODEL_DIR/*.pt.zip | head -n 1))
# same with onnx
updated_onnx=$(realpath $(ls -t -- $MODEL_DIR/*.onnx | head -n 1))

# running twice to make sure the hash is deterministic
# copy previous ones first
copy_ptzip=/tmp/$(basename $updated_ptzip)
copy_onnx=/tmp/$(basename $updated_onnx)
mv $updated_ptzip $copy_ptzip && rm $updated_ptzip
mv $updated_onnx  $copy_onnx && rm $updated_onnx

python update-yolo.py --model=$current_ptzip

# use the most recent .pt.zip
updated_ptzip=$(realpath $(ls -t -- $MODEL_DIR/*.pt.zip | head -n 1))
updated_onnx=$(realpath $(ls -t -- $MODEL_DIR/*.onnx | head -n 1))


sha256sum $updated_ptzip $copy_ptzip
test "$(sha256sum $updated_ptzip | cut -d' ' -f1)" = \
    "$(sha256sum $copy_ptzip | cut -d' ' -f1)"

sha256sum $updated_onnx $copy_onnx
test "$(sha256sum $updated_onnx | cut -d' ' -f1)" = \
    "$(sha256sum $copy_onnx | cut -d' ' -f1)"


# smoke test
echo "TODO: add an actual image file with cells and treerings"
python inference.py \
        --model=${updated_ptzip} \
        --input=tests/assets/black.png \
        --px-per-mm=1000 \
        --output=/tmp/DELETE



exit $failed
