#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
    echo "Usage: $0 HRX_V0.2.0_SOURCE BUILD_DIRECTORY" >&2
    exit 2
fi
src=$(cd "$(dirname "$0")/../.." && pwd)
sdk_src=$(cd "$1" && pwd)
mkdir -p "$2"
out=$(cd "$2" && pwd)
sdk="$out/sdk"
build="$out/build"

(
    cd "$sdk_src"
    python3.12 dev.py cmake setup
    source .venv/bin/activate
    cmake -S . -B "$out/sdk-build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
        -DBUILD_SHARED_LIBS=OFF -DIREE_ENABLE_WERROR_FLAG=OFF -DLIBHRX_BUILD=ON \
        -DLIBHRX_BUILD_HIP_BINDING=OFF -DLIBHRX_BUILD_PASSTHROUGH=OFF -DLIBHRX_BUILD_CTS=OFF
    cmake --build "$out/sdk-build" --target hrx hrx-info loomc_shared
    cmake --install "$out/sdk-build" --prefix "$sdk" --component HrxPublicDist
)

cmake -S "$src" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DGGML_HRX=ON -DCMAKE_PREFIX_PATH="$sdk" -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=OFF \
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_USE_PREBUILT_UI=OFF \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON '-DCMAKE_INSTALL_RPATH=$ORIGIN'
bin="$build/bin"
cp -a "$sdk"/lib/lib{hrx,loomc}.so* "$bin/"
programs=(llama-server llama-bench llama-perplexity test-backend-ops test-flash-attn-indexed-bias)
cmake --build "$build" --target "${programs[@]}"

stage=$(mktemp -d "$out/package.XXXXXXXX")
trap 'rm -rf -- "$stage"' EXIT
pkg="$stage/llama-hrx"
mkdir -p "$pkg/bin" "$pkg/scripts" "$pkg/licenses"
for program in "${programs[@]}"; do
    cp "$bin/$program" "$pkg/bin/"
done
cp -a "$bin"/lib{ggml,ggml-base,ggml-cpu,ggml-hrx,llama,llama-common,llama-server-impl,llama-bench-impl,llama-perplexity-impl,mtmd,hrx,loomc}.so* "$pkg/bin/"
cp "$src/scripts/serve-inkling2-dspark-hrx.sh" "$src/scripts/hrx/verify.sh" "$pkg/scripts/"
cp "$src/docs/backend/HRX.md" "$pkg/README.md"
cp "$src/LICENSE" "$pkg/licenses/llama.cpp"
cp "$sdk_src/LICENSE" "$pkg/licenses/hrx-system"
cp "$src"/licenses/LICENSE-* "$pkg/licenses/"
for dep in "$out"/sdk-build/_deps/*-src; do
    [[ -d $dep ]] || continue
    name=$(basename "$dep" -src)
    find "$dep" -maxdepth 1 -type f \( -iname 'license*' -o -iname 'copying*' -o -iname 'notice*' \) \
        -exec install -D -t "$pkg/licenses/$name" {} +
done
git -C "$src" rev-parse HEAD > "$pkg/REVISION"
getconf GNU_LIBC_VERSION > "$pkg/BUILD_LIBC"
tar -czf "$stage/llama-hrx-linux-x86_64.tar.gz" -C "$stage" llama-hrx
mv "$stage/llama-hrx-linux-x86_64.tar.gz" "$out/"
echo "$out/llama-hrx-linux-x86_64.tar.gz"
