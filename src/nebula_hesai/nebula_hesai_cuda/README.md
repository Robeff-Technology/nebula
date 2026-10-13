# nebula_hesai_cuda

OT128 (Pandar128E4X) decoder that generates the points on the GPU, for `nebula_hesai`'s
`use_cuda_decoder` parameter.

The per-packet bookkeeping stays on the CPU and is the same code as in `HesaiDecoder`: packet
parsing, functional safety, packet loss and IMU decoding, and the scan cutter, which runs once per
block. Instead of converting each return to a point, the CPU records per block which channels fall
into which scan. When a scan is cut, one kernel (`src/cuda_scan_builder.cu`) converts all of the
scan's packets, and a stable compaction puts the points in the CPU decoder's order: block group,
channel, return.

The GPU output is meant to be identical to the CPU decoder's, not just close:

- The angle tables (azimuth, sine, cosine per raw azimuth and channel) are filled by calling the
  CPU angle corrector, so the GPU's own trigonometry is never used.
- Firing time offsets and return types are tabulated from the sensor definition's own functions.
- The kernel repeats the CPU arithmetic in the same types (distance in double, then float) and is
  compiled with `--fmad=false`; the host code is ISO C++ without fused multiply-adds.

## Use

`nebula_hesai` builds the support when this package and `cuda_blackboard` are found. With
`use_cuda_decoder: true` and an OT128, `pandar_points` is published through cuda_blackboard (GPU
memory); plain `PointCloud2` subscribers still get a host copy through cuda_blackboard's
compatibility path. Remap `pandar_points/cuda` along with `pandar_points`. Load the consumer (e.g.
`autoware_cuda_pointcloud_preprocessor` with `use_cuda_blackboard_input: true`) into the same
container, or the cloud still crosses the host.

The node is created without node-wide intra-process in this mode (cuda_blackboard's negotiation
uses transient_local topics); this is read from the node's parameter overrides, i.e. pass
`use_cuda_decoder` as a parameter when loading the component.

Not supported: the blockage mask and the downsample mask filters, and models other than the OT128.
With those, the node logs why and decodes on the CPU, still publishing through cuda_blackboard.
