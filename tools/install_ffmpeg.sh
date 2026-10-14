#!/usr/bin/env bash
# Install a static ffmpeg + ffprobe into ./.tools/current (no root required).
# musicvid finds it automatically; otherwise put ffmpeg on PATH.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p .tools
cd .tools

arch="$(uname -m)"
case "$(uname -s)" in
  Linux)
    url="https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
    [ "$arch" = "aarch64" ] && url="https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-arm64-static.tar.xz"
    ;;
  Darwin)
    echo "On macOS: brew install ffmpeg" >&2
    exit 1
    ;;
  *)
    echo "Unsupported platform: $(uname -s). Install ffmpeg and put it on PATH." >&2
    exit 1
    ;;
esac

echo "Downloading $url"
curl -fsSL -o ff.tar.xz "$url"
tar xf ff.tar.xz
rm -rf current
ln -sfn "$PWD/$(ls -d ffmpeg-*-static | head -1)" current
rm -f ff.tar.xz
./current/ffmpeg -version | head -1
echo "Installed into .tools/current"
