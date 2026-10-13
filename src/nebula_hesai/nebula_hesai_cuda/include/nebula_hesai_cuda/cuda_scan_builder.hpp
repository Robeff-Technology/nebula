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

#include "nebula_hesai_cuda/cuda_types.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nebula::drivers::cuda
{

/// A contiguous run of packets in host memory (pinned, see CudaScanBuilder::allocate_host)
struct PacketSpan
{
  const CudaPacket * data{nullptr};
  std::size_t size{0};
};

/// Turns the packets and block groups of one scan into points on the GPU, in the order the CPU
/// decoder appends them: block group, then channel, then return.
class CudaScanBuilder
{
public:
  /// @param azimuth_rad, sin_azimuth, cos_azimuth [k_n_azimuths * k_n_channels]
  /// @param elevation_rad, sin_elevation, cos_elevation [k_n_channels]
  /// @param time_offset_ns [k_n_time_tables * k_n_blocks * k_n_channels]
  /// @param return_type [k_n_return_modes * k_max_returns * 2]
  CudaScanBuilder(
    const std::vector<float> & azimuth_rad, const std::vector<float> & sin_azimuth,
    const std::vector<float> & cos_azimuth, const std::vector<float> & elevation_rad,
    const std::vector<float> & sin_elevation, const std::vector<float> & cos_elevation,
    const std::vector<std::int32_t> & time_offset_ns, const std::vector<std::uint8_t> & return_type,
    const CudaDecoderParameters & parameters);
  ~CudaScanBuilder();

  CudaScanBuilder(const CudaScanBuilder &) = delete;
  CudaScanBuilder & operator=(const CudaScanBuilder &) = delete;

  /// Pinned host memory, so the per-scan copies to the GPU are plain DMA transfers
  static CudaPacket * allocate_host(std::size_t n_packets);
  static void free_host(CudaPacket * packets);

  /// @brief Decode one scan. The packets are given as up to two spans (a ring buffer that wrapped
  /// around); a block group's packet_index counts from the start of the first span.
  /// @return The number of points, in device_points(). Returns after the stream has finished, so
  /// the result can be read and the host packets reused right away.
  std::size_t build(
    PacketSpan first, PacketSpan second, const std::vector<CudaBlockGroup> & block_groups);

  /// Valid until the next build()
  [[nodiscard]] const CudaPoint * device_points() const { return points_; }
  [[nodiscard]] cudaStream_t stream() const { return stream_; }

private:
  template <typename T>
  static void reserve(T *& buffer, std::size_t & capacity, std::size_t n);

  cudaStream_t stream_{nullptr};
  CudaDecoderParameters parameters_{};

  float * azimuth_rad_{nullptr};
  float * sin_azimuth_{nullptr};
  float * cos_azimuth_{nullptr};
  float * elevation_rad_{nullptr};
  float * sin_elevation_{nullptr};
  float * cos_elevation_{nullptr};
  std::int32_t * time_offset_ns_{nullptr};
  std::uint8_t * return_type_{nullptr};

  CudaPacket * packets_{nullptr};
  std::size_t packets_capacity_{0};
  CudaBlockGroup * block_groups_{nullptr};
  std::size_t block_groups_capacity_{0};
  /// One slot per (block group, channel, return), and whether it holds a point
  CudaPoint * candidates_{nullptr};
  std::size_t candidates_capacity_{0};
  std::uint8_t * flags_{nullptr};
  std::size_t flags_capacity_{0};
  CudaPoint * points_{nullptr};
  std::size_t points_capacity_{0};
  std::uint8_t * cub_temp_{nullptr};
  std::size_t cub_temp_capacity_{0};
  int * n_selected_{nullptr};
  int * host_n_selected_{nullptr};
};

}  // namespace nebula::drivers::cuda
