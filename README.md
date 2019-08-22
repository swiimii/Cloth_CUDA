Simple cloth simulation using CUDA

This is the same thing as
[my other repo](https://github.com/stevenBorisko/Cloth)
but using CUDA and a lot less code. CUDA and OpenGL interop got a little too
deep in the graphics weeds, so I decided to stay away and just bring the
particle data back to the host whenever rendering a frame. This is, of course,
saying no to a great reduction in both device memory traffic and graphic
rendering speed, but I am focused more on CUDA than OpenGL.

The execution flow is fairly simple...

Two Threads:
- CUDA: Call CUDA kernel until OpenGL thread is ready to render a frame. At that
point, it swaps the "read" and "write" particle data buffers so OpenGL can
render the next frame.
- OpenGL: Render particle positions using GL\_POINTS or GL\_LINES depending on
the graphics settings.

## Compiling

```
mkdir build
cd build
cmake ..
make
```

## Running

### Automagic

`./sampleRun.sh`

### Manual

`./generateMesh [ARGUMENTS] | ./build/bin/cloth [OPTIONS]`

Give both those commands a separate run with `--help` to get some actual
instructions. Look at `sampleRun.sh` for a good starting point for paramter
tweaking.

There are a few keys on which the graphics window will act:
- `b` toggles bindings
- `c` toggles colors
- `q` and `ESC` exit the program

## Performance

Rendered using an Nvidia GeForce 840M on my laptop I bought in 2015. 

Particles in mesh | Milliseconds per 1000 time steps
----------------- | --------------------------------
64 | 30.157
256 | 41.538
1024 | 114.09
4096 | 403.27
16384 | 1287.0
65536 | 5821.1
262144 | 23327

The complexity of this program is clearly linear. Each particle added to the
mesh appends about 100 microseconds per 1000 time steps.

## Gallery

The following videos are 1600-particle meshes with 64 time steps per frame.
Doing some quick math with the stats above, that is about 100 frames per second.
Keep in mind though that all hundred frames making it through `recordmydesktop`
and several `ffmpeg` cropping commands is unlikely, but the computation is what
counts!
- [Color](https://vimeo.com/355227231): The blue represents relaxed bindings
while red bindings are more stressed.
- [Visibility](https://vimeo.com/355227223): Same simulation. This is just to
show that the graphics can be changed at runtime.
