# Install FFmpeg

DexJoCo uses FFmpeg for video recording. Choose an installation method below, then verify that `ffmpeg` and `ffprobe` are available on your path.

## With sudo on Ubuntu or Debian

```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

## Without sudo on Linux

Download a shared build from [BtbN’s FFmpeg builds](https://github.com/BtbN/FFmpeg-Builds). The `gpl-shared` package includes the command-line programs and the FFmpeg shared libraries needed by Python video backends such as TorchCodec. Keep the package’s `bin/` and `lib/` directories together.

The following commands install FFmpeg 8.1 under `~/.local/opt/ffmpeg-8.1`. These builds require glibc 2.28 or newer and Linux 4.18 or newer.

```bash
(
    set -eu

    case "$(uname -m)" in
        x86_64) ffmpeg_target=linux64 ;;
        aarch64|arm64) ffmpeg_target=linuxarm64 ;;
        *) echo "Choose a build for your architecture from the release page."; exit 1 ;;
    esac

    ffmpeg_tmp=$(mktemp -d)
    trap 'rm -rf "$ffmpeg_tmp"' EXIT

    curl -fL "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-${ffmpeg_target}-gpl-shared-8.1.tar.xz" \
        -o "$ffmpeg_tmp/ffmpeg.tar.xz"
    mkdir -p "$HOME/.local/opt/ffmpeg-8.1"
    tar -xJf "$ffmpeg_tmp/ffmpeg.tar.xz" \
        -C "$HOME/.local/opt/ffmpeg-8.1" --strip-components=1
)
```

Expose the programs and shared libraries to the current shell:

```bash
export PATH="$HOME/.local/opt/ffmpeg-8.1/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/opt/ffmpeg-8.1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

For future Bash sessions, also add these lines to `~/.bashrc`. For Zsh, use `~/.zshrc`. The `latest` URL follows ongoing builds of the 8.1 branch.

## Verify the installation

```bash
command -v ffmpeg
ffmpeg -version
command -v ffprobe
ffprobe -version
```

For other platforms and builds, see the [FFmpeg download page](https://ffmpeg.org/download.html).
