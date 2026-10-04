# client_doca/ — reference only

A GPUNetIO client set-up (flow + cyclic RX queue, GPU TX queue with wait-on-time, dmabuf buffer)
derived from the TVG suite. It is **not built and was not used** in any measurement: every network
run uses the AF_PACKET client in `client/`. It is kept because the streaming generator
(`stream/gen_host.c`, `stream/gen_kernel.cu`) derives its send path from these files.
