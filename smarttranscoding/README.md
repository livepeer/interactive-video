### Requirements

Project requires libavcodec (ffmpeg) and friends. See `install_ffmpeg.sh` . Running this script will install everything in `~/compiled`. In order to build the project, the dependent libraries will need to be discoverable by pkg-config and golang. If you installed everything with `install_ffmpeg.sh` , then run `export PKG_CONFIG_PATH=~/compiled/lib/pkgconfig:$PKG_CONFIG_PATH` so the deps are picked up.
  
  remark: For rapid quality assurance we offer use of burned in subtitle. To use this ffmpeg should be built with --enable-libass
 
To build the project, golang needs to be installed.

https://golang.org/doc/install

### Build 

check PKG_CONFIG_PATH,CGO_CFLAGS,CGO_LDFLAGS environment values.

export PKG_CONFIG_PATH="${PKG_CONFIG_PATH:-}:$HOME/compiled/lib/pkgconfig"

export CGO_CFLAGS="-I$HOME/compiled/include"

export CGO_LDFLAGS="-L$HOME/compiled/lib"

```
git clone https://github.com/livepeer/interactive-video.git 

cd interactive-video/smarttranscoding

go build cmd/example/smarttranscoding.go

```

### Running the smarttranscoding API server

```
./smarttranscoding -addr <ip:port>
```