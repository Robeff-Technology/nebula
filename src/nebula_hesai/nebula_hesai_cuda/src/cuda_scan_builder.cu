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

// Compiled with --fmad=false: the point coordinates must come out of the same float operations as
// in the CPU decoder (HesaiDecoder::convert_returns), without fused multiply-adds.

#include "nebula_hesai_cuda/cuda_scan_builder.hpp"

#include <cub/device/device_select.cuh>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace nebula::drivers::cuda
{

namespace
{

void check(cudaError_t error, const char * what)
{
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
  }
}

template <typename T>
T * upload(const std::vector<T> & host, std::size_t expected_size, const char * what)
{
  if (host.size() != expected_size) {
    throw std::invalid_argument(std::string("CudaScanBuilder: wrong size of ") + what);
  }
  T * device = nullptr;
  check(cudaMalloc(&device, host.size() * sizeof(T)), what);
  check(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), what);
  return device;
}

/// One thread per (block group, channel). Mirrors HesaiDecoder::convert_returns for one channel,
/// without the blockage mask and downsample filters (the host side refuses those configurations).
__global__ void decode_kernel(
  const CudaPacket * __restrict__ packets, const CudaBlockGroup * __restrict__ block_groups,
  int n_block_groups, CudaDecoderTables tables, CudaDecoderParameters parameters,
  CudaPoint * __restrict__ candidates, std::uint8_t * __restrict__ flags)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n_block_groups * k_n_channels) {
    return;
  }

  const int group_id = index / k_n_channels;
  const int channel_id = index % k_n_channels;
  const std::size_t out = static_cast<std::size_t>(index) * k_max_returns;
  for (int r = 0; r < k_max_returns; ++r) {
    flags[out + r] = 0;
  }

  const CudaBlockGroup & group = block_groups[group_id];
  if (((group.channel_mask[channel_id / 32] >> (channel_id % 32)) & 1U) == 0) {
    return;
  }

  const CudaPacket & packet = packets[group.packet_index];
  const int n_blocks = group.n_blocks;
  const std::uint32_t raw_azimuth = packet.azimuth[group.start_block];
  const int return_mode = static_cast<int>(packet.return_mode) - k_first_return_mode;
  if (
    raw_azimuth >= static_cast<std::uint32_t>(k_n_azimuths) || return_mode < 0 ||
    return_mode >= k_n_return_modes) {
    return;
  }

  std::uint16_t raw_distance[k_max_returns];
  std::uint8_t reflectivity[k_max_returns];
  for (int r = 0; r < n_blocks; ++r) {
    const std::uint8_t * unit = packet.units[group.start_block + r][channel_id];
    raw_distance[r] = static_cast<std::uint16_t>(unit[0] | (unit[1] << 8));
    reflectivity[r] = unit[2];
  }

  const std::size_t angle_index =
    static_cast<std::size_t>(raw_azimuth) * k_n_channels + channel_id;

  for (int r = 0; r < n_blocks; ++r) {
    bool point_is_valid = raw_distance[r] != 0;

    // get_distance(): uint16 times the double distance unit, stored as float
    const float distance = static_cast<float>(raw_distance[r] * packet.dis_unit);

    if (
      distance < parameters.sensor_min_range || parameters.sensor_max_range < distance ||
      distance < parameters.min_range || parameters.max_range < distance) {
      point_is_valid = false;
    }

    // HesaiSensor::get_return_type: identical returns first, then the sensor's per-mode mapping
    bool is_duplicate = false;
    bool is_strongest = true;
    for (int i = 0; i < n_blocks; ++i) {
      if (i == r) {
        continue;
      }
      if (raw_distance[r] == raw_distance[i] && reflectivity[r] == reflectivity[i]) {
        is_duplicate = true;
      }
      if (reflectivity[r] < reflectivity[i]) {
        is_strongest = false;
      }
    }
    const std::uint8_t return_type =
      is_duplicate
        ? parameters.return_type_identical
        : tables.return_type[(return_mode * k_max_returns + r) * 2 + (is_strongest ? 1 : 0)];

    // Keep only the last of multiple identical points
    if (return_type == parameters.return_type_identical && r != n_blocks - 1) {
      point_is_valid = false;
    }

    // Keep only the last (if any) of multiple points that are too close
    if (r != n_blocks - 1) {
      for (int i = 0; i < n_blocks; ++i) {
        if (i == r) {
          continue;
        }
        const float other_distance = static_cast<float>(raw_distance[i] * packet.dis_unit);
        if (fabsf(other_distance - distance) < parameters.dual_return_distance_threshold) {
          point_is_valid = false;
          break;
        }
      }
    }

    if (!point_is_valid) {
      continue;
    }

    const int block_id = group.start_block + r;
    const int time_table = packet.time_table_index[block_id];
    const std::int64_t point_to_packet_offset_ns =
      tables.time_offset_ns[(time_table * k_n_blocks + block_id) * k_n_channels + channel_id];
    const std::int64_t point_to_scan_offset_ns =
      group.packet_to_scan_ns + point_to_packet_offset_ns;

    CudaPoint point;
    point.distance = distance;
    point.intensity = reflectivity[r];
    point.time_stamp =
      point_to_scan_offset_ns < 0 ? 0U : static_cast<std::uint32_t>(point_to_scan_offset_ns);
    point.return_type = return_type;
    point.channel = static_cast<std::uint16_t>(channel_id);

    const float xy_distance = distance * tables.cos_elevation[channel_id];
    point.x = xy_distance * tables.sin_azimuth[angle_index];
    point.y = xy_distance * tables.cos_azimuth[angle_index];
    point.z = distance * tables.sin_elevation[channel_id];
    point.azimuth = tables.azimuth_rad[angle_index];
    point.elevation = tables.elevation_rad[channel_id];

    candidates[out + r] = point;
    flags[out + r] = 1;
  }
}

}  // namespace

CudaScanBuilder::CudaScanBuilder(
  const std::vector<float> & azimuth_rad, const std::vector<float> & sin_azimuth,
  const std::vector<float> & cos_azimuth, const std::vector<float> & elevation_rad,
  const std::vector<float> & sin_elevation, const std::vector<float> & cos_elevation,
  const std::vector<std::int32_t> & time_offset_ns, const std::vector<std::uint8_t> & return_type,
  const CudaDecoderParameters & parameters)
: parameters_(parameters)
{
  constexpr std::size_t n_angles = static_cast<std::size_t>(k_n_azimuths) * k_n_channels;
  check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreate");
  azimuth_rad_ = upload(azimuth_rad, n_angles, "azimuth_rad");
  sin_azimuth_ = upload(sin_azimuth, n_angles, "sin_azimuth");
  cos_azimuth_ = upload(cos_azimuth, n_angles, "cos_azimuth");
  elevation_rad_ = upload(elevation_rad, k_n_channels, "elevation_rad");
  sin_elevation_ = upload(sin_elevation, k_n_channels, "sin_elevation");
  cos_elevation_ = upload(cos_elevation, k_n_channels, "cos_elevation");
  time_offset_ns_ =
    upload(time_offset_ns, k_n_time_tables * k_n_blocks * k_n_channels, "time_offset_ns");
  return_type_ = upload(return_type, k_n_return_modes * k_max_returns * 2, "return_type");
  check(cudaMalloc(&n_selected_, sizeof(int)), "cudaMalloc n_selected");
  check(cudaMallocHost(&host_n_selected_, sizeof(int)), "cudaMallocHost n_selected");
}

CudaScanBuilder::~CudaScanBuilder()
{
  if (stream_) {
    cudaStreamSynchronize(stream_);
  }
  for (void * p :
       {static_cast<void *>(azimuth_rad_), static_cast<void *>(sin_azimuth_),
        static_cast<void *>(cos_azimuth_), static_cast<void *>(elevation_rad_),
        static_cast<void *>(sin_elevation_), static_cast<void *>(cos_elevation_),
        static_cast<void *>(time_offset_ns_), static_cast<void *>(return_type_),
        static_cast<void *>(packets_), static_cast<void *>(block_groups_),
        static_cast<void *>(candidates_), static_cast<void *>(flags_),
        static_cast<void *>(points_), static_cast<void *>(cub_temp_),
        static_cast<void *>(n_selected_)}) {
    cudaFree(p);
  }
  cudaFreeHost(host_n_selected_);
  if (stream_) {
    cudaStreamDestroy(stream_);
  }
}

CudaPacket * CudaScanBuilder::allocate_host(std::size_t n_packets)
{
  CudaPacket * packets = nullptr;
  check(cudaMallocHost(&packets, n_packets * sizeof(CudaPacket)), "cudaMallocHost packets");
  return packets;
}

void CudaScanBuilder::free_host(CudaPacket * packets)
{
  cudaFreeHost(packets);
}

template <typename T>
void CudaScanBuilder::reserve(T *& buffer, std::size_t & capacity, std::size_t n)
{
  if (n <= capacity) {
    return;
  }
  // Grow with headroom: scans vary slightly in size, reallocating every scan would stall
  const std::size_t new_capacity = n + n / 4;
  cudaFree(buffer);  // implicitly synchronizes
  buffer = nullptr;
  capacity = 0;
  check(cudaMalloc(&buffer, new_capacity * sizeof(T)), "cudaMalloc");
  capacity = new_capacity;
}

std::size_t CudaScanBuilder::build(
  PacketSpan first, PacketSpan second, const std::vector<CudaBlockGroup> & block_groups)
{
  if (block_groups.empty()) {
    return 0;
  }

  const std::size_t n_packets = first.size + second.size;
  const std::size_t n_candidates = block_groups.size() * k_n_channels * k_max_returns;

  reserve(packets_, packets_capacity_, n_packets);
  reserve(block_groups_, block_groups_capacity_, block_groups.size());
  reserve(candidates_, candidates_capacity_, n_candidates);
  reserve(flags_, flags_capacity_, n_candidates);
  reserve(points_, points_capacity_, n_candidates);

  check(
    cudaMemcpyAsync(
      packets_, first.data, first.size * sizeof(CudaPacket), cudaMemcpyHostToDevice, stream_),
    "copy packets");
  if (second.size > 0) {
    check(
      cudaMemcpyAsync(
        packets_ + first.size, second.data, second.size * sizeof(CudaPacket),
        cudaMemcpyHostToDevice, stream_),
      "copy packets");
  }
  check(
    cudaMemcpyAsync(
      block_groups_, block_groups.data(), block_groups.size() * sizeof(CudaBlockGroup),
      cudaMemcpyHostToDevice, stream_),
    "copy block groups");

  const CudaDecoderTables tables{azimuth_rad_,   sin_azimuth_,   cos_azimuth_,
                                 elevation_rad_, sin_elevation_, cos_elevation_,
                                 time_offset_ns_, return_type_};
  const int n_threads = static_cast<int>(block_groups.size()) * k_n_channels;
  constexpr int threads_per_block = 256;
  const int n_thread_blocks = (n_threads + threads_per_block - 1) / threads_per_block;
  decode_kernel<<<n_thread_blocks, threads_per_block, 0, stream_>>>(
    packets_, block_groups_, static_cast<int>(block_groups.size()), tables, parameters_,
    candidates_, flags_);
  check(cudaGetLastError(), "decode_kernel");

  // Stable compaction: keeps the (block group, channel, return) order of the CPU decoder
  std::size_t temp_bytes = 0;
  check(
    cub::DeviceSelect::Flagged(
      nullptr, temp_bytes, candidates_, flags_, points_, n_selected_,
      static_cast<int>(n_candidates), stream_),
    "DeviceSelect::Flagged (size)");
  reserve(cub_temp_, cub_temp_capacity_, temp_bytes);
  check(
    cub::DeviceSelect::Flagged(
      cub_temp_, temp_bytes, candidates_, flags_, points_, n_selected_,
      static_cast<int>(n_candidates), stream_),
    "DeviceSelect::Flagged");

  check(
    cudaMemcpyAsync(host_n_selected_, n_selected_, sizeof(int), cudaMemcpyDeviceToHost, stream_),
    "copy point count");
  check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
  return static_cast<std::size_t>(*host_n_selected_);
}

}  // namespace nebula::drivers::cuda
