Simple cloth simulation using CUDA

This is the same thing as
[my other repo](https://github.com/stevenBorisko/Cloth)
but using CUDA and a lot less code. CUDA and OpenGL interop got a little too
deep in the graphics weeds, so I decided to stay away and just bring the
particle data back to the host whenever rendering a frame.

The execution flow is fairly simple...

Two Threads:
- CUDA: Call CUDA kernel until OpenGL thread is ready to render a frame. At that
point, it swaps the "read" and "write" particle data buffers so OpenGL can
render the next frame.
- OpenGL: Render particle positions using GL\_POINTS.

## Gallery

65536 (256x256) particle mesh. 
- [Real time screen recording](https://vimeo.com/354576797)
rendered using an Nvidia GeForce 840M on my laptop I bought in 2015. 
- [Same recording sped up 10x](https://vimeo.com/354577104)
to make it look less like it is on the Moon.
