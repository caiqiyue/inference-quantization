#!/usr/bin/env bash
# Download and verify the official, unquantized Qwen3-8B checkpoint from ModelScope.
# Safe to rerun: valid files are skipped and incomplete files are resumed.
set -euo pipefail

destination="${1:-/home/qiyue.cai/host_workspace/models/Qwen3-8B}"
mkdir -p "$destination"

files=(
  ".gitattributes 088e25d6bceaa8943f4f292006210358292349cc36e7578f80c62fce75535f46"
  "LICENSE 832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e"
  "README.md 0f36caaff9c2516411a7738db384606263ba653c1e63e61d72f511606164d5a6"
  "config.json f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30"
  "configuration.json f888421726665e8a84b738eed42a64875aed79de8be7daade851ac8bf4c0cef9"
  "generation_config.json 2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2"
  "merges.txt 8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"
  "model-00001-of-00005.safetensors 31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f"
  "model-00002-of-00005.safetensors 5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282"
  "model-00003-of-00005.safetensors c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836"
  "model-00004-of-00005.safetensors b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a"
  "model-00005-of-00005.safetensors 20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff"
  "model.safetensors.index.json f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc"
  "tokenizer.json aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
  "tokenizer_config.json d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"
  "vocab.json ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"
)

for entry in "${files[@]}"; do
  read -r name expected_sha <<<"$entry"
  target="$destination/$name"
  if [[ -f "$target" ]] && [[ "$(sha256sum "$target" | awk '{print $1}')" == "$expected_sha" ]]; then
    echo "verified: $name"
    continue
  fi
  echo "downloading: $name"
  curl --fail --location --retry 8 --retry-all-errors --connect-timeout 15 \
    --continue-at - --output "$target" \
    "https://www.modelscope.cn/api/v1/models/Qwen/Qwen3-8B/repo?Revision=master&FilePath=$name"
  actual_sha="$(sha256sum "$target" | awk '{print $1}')"
  [[ "$actual_sha" == "$expected_sha" ]] || { echo "SHA-256 mismatch: $name" >&2; exit 1; }
  echo "verified: $name"
done

echo "Qwen3-8B is complete and verified at: $destination"
