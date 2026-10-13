// Copyright 2026 Robeff Technology
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

// Plain data shared by the host side of the CUDA decoder and its kernels. Nothing here may pull in
// nebula or ROS headers: the kernels are compiled by nvcc.

#include <cstddef>
#include <cstdint>

namespace nebula::drivers::cuda
{

/// Pandar128E4X (OT128): 2 blocks of 128 channels per packet, at most 2 returns, 0.01 deg azimuth
constexpr int k_n_channels = 128;
constexpr int k_n_blocks = 2;
constexpr int k_max_returns = 2;
constexpr int k_n_azimuths = 360 * 100;
constexpr int k_channel_mask_words = k_n_channels / 32;

/// The 11 return modes of the Hesai protocol, 0x33 (SINGLE_FIRST) to 0x3d / 0x3e
constexpr int k_first_return_mode = 0x33;
constexpr int k_n_return_modes = 0x3e - k_first_return_mode + 1;

/// Time offset table index: n_returns (1, 2) x operational state (high resolution, other) x
/// azimuth state (0 to 3)
constexpr int k_n_time_tables = 2 * 2 * 4;

/// Same memory layout as nebula::drivers::PointXYZIRCAEDT (NebulaPoint); checked on the host side.
struct alignas(16) CudaPoint
{
  float x;
  float y;
  float z;
  std::uint8_t intensity;
  std::uint8_t return_type;
  std::uint16_t channel;
  float azimuth;
  float elevation;
  float distance;
  std::uint32_t time_stamp;
};

/// The parts of one packet the kernel reads, copied out of the packet on the host.
struct CudaPacket
{
  /// Meters per distance unit (header.dis_unit / 1000), in double as in the CPU decoder
  double dis_unit;
  /// Raw block azimuths, in 0.01 deg
  std::uint16_t azimuth[k_n_blocks];
  /// Raw return mode from the packet tail
  std::uint8_t return_mode;
  /// Per block: index of the firing time offset table (see k_n_time_tables)
  std::uint8_t time_table_index[k_n_blocks];
  std::uint8_t reserved[3];
  /// Unit3B: distance (uint16, little endian) and reflectivity, per block and channel
  std::uint8_t units[k_n_blocks][k_n_channels][3];
};

/// A group of returns (one block in single return mode, two in dual return mode) whose points go
/// into the scan being decoded, with the channels that belong to it.
struct CudaBlockGroup
{
  /// Index of the packet in the scan's packet array
  std::uint32_t packet_index;
  std::uint8_t start_block;
  std::uint8_t n_blocks;
  std::uint8_t reserved[2];
  /// Packet timestamp minus scan timestamp, in ns
  std::int64_t packet_to_scan_ns;
  /// Bit c set: channel c is in the FoV and in this scan
  std::uint32_t channel_mask[k_channel_mask_words];
};

/// Constant inputs of the kernel, computed once on the host from the sensor definition and its
/// calibration (so they match the CPU decoder bit for bit).
struct CudaDecoderTables
{
  /// [k_n_azimuths * k_n_channels]: corrected azimuth (rad) and its sine and cosine
  const float * azimuth_rad;
  const float * sin_azimuth;
  const float * cos_azimuth;
  /// [k_n_channels]
  const float * elevation_rad;
  const float * sin_elevation;
  const float * cos_elevation;
  /// [k_n_time_tables * k_n_blocks * k_n_channels]: firing time of the point relative to its packet
  const std::int32_t * time_offset_ns;
  /// [k_n_return_modes * k_max_returns * 2 (not strongest, strongest)]
  const std::uint8_t * return_type;
};

struct CudaDecoderParameters
{
  /// ReturnType::IDENTICAL
  std::uint8_t return_type_identical;
  /// The sensor's own range limits (float, as in the sensor definition)
  float sensor_min_range;
  float sensor_max_range;
  /// The configured range limits and dual return threshold (double, as in the configuration)
  double min_range;
  double max_range;
  double dual_return_distance_threshold;
};

}  // namespace nebula::drivers::cuda
