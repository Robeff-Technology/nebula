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

#include "nebula_hesai_cuda/hesai_cuda_decoder.hpp"

#include "nebula_hesai_cuda/cuda_scan_builder.hpp"

#include <nebula_core_common/nebula_common.hpp>
#include <nebula_core_common/point_types.hpp>
#include <nebula_core_common/util/stopwatch.hpp>
#include <nebula_core_decoders/angles.hpp>
#include <nebula_core_decoders/scan_cutter.hpp>
#include <nebula_hesai_decoders/decoders/hesai_packet.hpp>
#include <nebula_hesai_decoders/decoders/pandar_128e4x.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nebula::drivers
{

namespace
{

using SensorT = Pandar128E4X;
using packet_t = SensorT::packet_t;
using unit_t = packet_t::body_t::block_t::unit_t;

static_assert(packet_t::n_channels == cuda::k_n_channels);
static_assert(packet_t::n_blocks == cuda::k_n_blocks);
static_assert(packet_t::max_returns == cuda::k_max_returns);
static_assert(packet_t::degree_subdivisions * 360 == cuda::k_n_azimuths);
static_assert(sizeof(unit_t) == 3, "CudaPacket stores Unit3B as 3 bytes");
static_assert(
  sizeof(cuda::CudaPacket::units[0]) == sizeof(packet_t::body_t::block_t::units),
  "CudaPacket must hold one block's units as they are in the packet");

// The kernel writes CudaPoint; the wrapper publishes it as NebulaPoint
static_assert(sizeof(cuda::CudaPoint) == sizeof(NebulaPoint));
static_assert(offsetof(cuda::CudaPoint, x) == offsetof(NebulaPoint, x));
static_assert(offsetof(cuda::CudaPoint, y) == offsetof(NebulaPoint, y));
static_assert(offsetof(cuda::CudaPoint, z) == offsetof(NebulaPoint, z));
static_assert(offsetof(cuda::CudaPoint, intensity) == offsetof(NebulaPoint, intensity));
static_assert(offsetof(cuda::CudaPoint, return_type) == offsetof(NebulaPoint, return_type));
static_assert(offsetof(cuda::CudaPoint, channel) == offsetof(NebulaPoint, channel));
static_assert(offsetof(cuda::CudaPoint, azimuth) == offsetof(NebulaPoint, azimuth));
static_assert(offsetof(cuda::CudaPoint, elevation) == offsetof(NebulaPoint, elevation));
static_assert(offsetof(cuda::CudaPoint, distance) == offsetof(NebulaPoint, distance));
static_assert(offsetof(cuda::CudaPoint, time_stamp) == offsetof(NebulaPoint, time_stamp));

/// Pandar128E4X's operational state for high resolution mode (private in the sensor definition).
/// Every other state uses the static firing time table.
constexpr uint8_t k_operational_state_high_resolution = 0;

int time_table_index(int n_returns, uint8_t operational_state, uint8_t azimuth_state)
{
  const int returns_index = n_returns == 1 ? 0 : 1;
  const int state_index = operational_state == k_operational_state_high_resolution ? 0 : 1;
  return (returns_index * 2 + state_index) * 4 + (azimuth_state & 0b11);
}

}  // namespace

struct HesaiCudaDecoder128E4X::Impl
{
  /// Packets kept for the scans that are not cut yet. One 10 Hz scan is 1800 packets in single
  /// return and 3600 in dual return mode; two scans can be open at the cut.
  static constexpr uint64_t k_ring_size = 16384;
  static constexpr std::chrono::seconds k_log_interval{5};

  struct Frame
  {
    uint64_t scan_timestamp_ns{0};
    /// Ring slot of the first packet a block group of this scan refers to
    uint64_t first_slot{0};
    std::vector<cuda::CudaBlockGroup> block_groups;
  };

  std::shared_ptr<const HesaiSensorConfiguration> sensor_configuration;
  std::shared_ptr<loggers::Logger> logger;
  SensorT sensor{};
  SensorT::angle_corrector_t angle_corrector;
  ScanCutter<packet_t::n_channels, float> scan_cutter;

  std::shared_ptr<FunctionalSafetyDecoderTypedBase<packet_t>> functional_safety_decoder;
  std::shared_ptr<PacketLossDetectorTypedBase<packet_t>> packet_loss_detector;
  std::shared_ptr<ImuDecoderTypedBase<packet_t>> imu_decoder;

  pointcloud_callback_t pointcloud_callback;
  cuda_pointcloud_callback_t cuda_pointcloud_callback;

  std::unique_ptr<cuda::CudaScanBuilder> builder;
  cuda::CudaPacket * ring{nullptr};
  uint64_t next_slot{0};
  uint64_t current_slot{0};
  uint64_t current_packet_timestamp_ns{0};
  std::array<Frame, 2> frames;

  packet_t packet{};
  uint64_t callback_time_ns{0};
  bool did_scan_complete{false};
  size_t current_block_id{0};

  uint64_t malformed_packet_count{0};
  std::chrono::steady_clock::time_point last_malformed_packet_log{};
  std::chrono::steady_clock::time_point last_overflow_log{};

  Impl(
    const std::shared_ptr<const HesaiSensorConfiguration> & config,
    const std::shared_ptr<const HesaiCalibrationConfiguration> & calibration,
    const std::shared_ptr<loggers::Logger> & logger_in)
  : sensor_configuration(config),
    logger(logger_in),
    angle_corrector(calibration),
    scan_cutter(
      2 * M_PIf, deg2rad(config->cut_angle), deg2rad(config->cloud_min_angle),
      deg2rad(config->cloud_max_angle), [this](uint8_t buffer_index) { on_scan_complete(buffer_index); },
      [this](uint8_t buffer_index) { on_set_timestamp(buffer_index); })
  {
    builder = make_builder();
    ring = cuda::CudaScanBuilder::allocate_host(k_ring_size);
  }

  ~Impl() { cuda::CudaScanBuilder::free_host(ring); }

  Impl(const Impl &) = delete;
  Impl & operator=(const Impl &) = delete;

  /// Builds the kernel's lookup tables from the same functions the CPU decoder calls per point
  std::unique_ptr<cuda::CudaScanBuilder> make_builder()
  {
    constexpr size_t n_angles = static_cast<size_t>(cuda::k_n_azimuths) * cuda::k_n_channels;
    std::vector<float> azimuth_rad(n_angles);
    std::vector<float> sin_azimuth(n_angles);
    std::vector<float> cos_azimuth(n_angles);
    for (uint32_t azimuth = 0; azimuth < static_cast<uint32_t>(cuda::k_n_azimuths); ++azimuth) {
      for (uint32_t channel = 0; channel < static_cast<uint32_t>(cuda::k_n_channels); ++channel) {
        const auto angles = angle_corrector.get_corrected_angle_data(azimuth, channel);
        const size_t index = static_cast<size_t>(azimuth) * cuda::k_n_channels + channel;
        azimuth_rad[index] = angles.azimuth_rad;
        sin_azimuth[index] = angles.sin_azimuth;
        cos_azimuth[index] = angles.cos_azimuth;
      }
    }

    std::vector<float> elevation_rad(cuda::k_n_channels);
    std::vector<float> sin_elevation(cuda::k_n_channels);
    std::vector<float> cos_elevation(cuda::k_n_channels);
    for (uint32_t channel = 0; channel < static_cast<uint32_t>(cuda::k_n_channels); ++channel) {
      const auto angles = angle_corrector.get_corrected_angle_data(0, channel);
      elevation_rad[channel] = angles.elevation_rad;
      sin_elevation[channel] = angles.sin_elevation;
      cos_elevation[channel] = angles.cos_elevation;
    }

    // Firing time of each point relative to its packet, per (returns, operational state, azimuth
    // state, block, channel), from the sensor definition itself
    std::vector<int32_t> time_offset_ns(cuda::k_n_time_tables * cuda::k_n_blocks * cuda::k_n_channels);
    for (int n_returns : {1, 2}) {
      for (uint8_t operational_state : {k_operational_state_high_resolution, uint8_t{1}}) {
        for (uint8_t azimuth_state = 0; azimuth_state < 4; ++azimuth_state) {
          for (uint32_t block_id = 0; block_id < cuda::k_n_blocks; ++block_id) {
            packet_t probe{};
            probe.tail.return_mode = n_returns == 1 ? hesai_packet::return_mode::SINGLE_STRONGEST
                                                    : hesai_packet::return_mode::DUAL_LAST_STRONGEST;
            probe.tail.operational_state = operational_state;
            probe.tail.azimuth_state = static_cast<uint16_t>(azimuth_state << (14 - block_id * 2));
            const int table = time_table_index(n_returns, operational_state, azimuth_state);
            for (uint32_t channel = 0; channel < static_cast<uint32_t>(cuda::k_n_channels);
                 ++channel) {
              time_offset_ns[(table * cuda::k_n_blocks + block_id) * cuda::k_n_channels + channel] =
                sensor.get_packet_relative_point_time_offset(block_id, channel, probe);
            }
          }
        }
      }
    }

    // Return type per (return mode, return index, strongest or not), from the sensor definition
    std::vector<uint8_t> return_type(cuda::k_n_return_modes * cuda::k_max_returns * 2, 0);
    for (int mode_index = 0; mode_index < cuda::k_n_return_modes; ++mode_index) {
      const auto mode =
        static_cast<hesai_packet::return_mode::ReturnMode>(cuda::k_first_return_mode + mode_index);
      int n_returns = 0;
      try {
        n_returns = hesai_packet::get_n_returns(mode);
      } catch (const std::runtime_error &) {
        continue;
      }
      if (n_returns > cuda::k_max_returns) {
        continue;
      }
      for (int return_idx = 0; return_idx < n_returns; ++return_idx) {
        for (int strongest = 0; strongest < 2; ++strongest) {
          // Distinct distances (no identical returns); return_idx is the strongest or the weakest
          std::array<unit_t, cuda::k_max_returns> units{};
          std::vector<const unit_t *> return_units;
          for (int i = 0; i < n_returns; ++i) {
            units[i].distance = static_cast<uint16_t>(100 + i);
            const bool is_this = i == return_idx;
            units[i].reflectivity = (is_this == (strongest == 1)) ? 200 : 100;
            return_units.push_back(&units[i]);
          }
          return_type[(mode_index * cuda::k_max_returns + return_idx) * 2 + strongest] =
            static_cast<uint8_t>(sensor.get_return_type(mode, return_idx, return_units));
        }
      }
    }

    cuda::CudaDecoderParameters parameters{};
    parameters.return_type_identical = static_cast<uint8_t>(ReturnType::IDENTICAL);
    parameters.sensor_min_range = SensorT::min_range;
    parameters.sensor_max_range = SensorT::max_range;
    parameters.min_range = sensor_configuration->min_range;
    parameters.max_range = sensor_configuration->max_range;
    parameters.dual_return_distance_threshold = sensor_configuration->dual_return_distance_threshold;

    return std::make_unique<cuda::CudaScanBuilder>(
      azimuth_rad, sin_azimuth, cos_azimuth, elevation_rad, sin_elevation, cos_elevation,
      time_offset_ns, return_type, parameters);
  }

  bool parse_packet(const std::vector<uint8_t> & raw_packet)
  {
    if (raw_packet.size() < sizeof(packet_t)) {
      // See HesaiDecoder::parse_packet: report the first, then at most once per interval
      ++malformed_packet_count;
      const auto now = std::chrono::steady_clock::now();
      if (now - last_malformed_packet_log >= k_log_interval) {
        std::stringstream message;
        message << "Packet size mismatch: " << raw_packet.size()
                << " | Expected at least: " << sizeof(packet_t) << " (" << malformed_packet_count
                << " undecodable packet(s) since last report)";
        logger->error(message.str());
        last_malformed_packet_log = now;
        malformed_packet_count = 0;
      }
      return false;
    }

    std::memcpy(&packet, raw_packet.data(), sizeof(packet_t));

    // As in HesaiDecoder::parse_packet: get_n_returns() would throw and end the process
    if (!hesai_packet::is_known_return_mode(packet.tail.return_mode)) {
      ++malformed_packet_count;
      const auto now = std::chrono::steady_clock::now();
      if (now - last_malformed_packet_log >= k_log_interval) {
        std::stringstream message;
        message << "Unknown return mode 0x" << std::hex << static_cast<int>(packet.tail.return_mode)
                << std::dec << " (" << malformed_packet_count
                << " undecodable packet(s) since last report)";
        logger->error(message.str());
        last_malformed_packet_log = now;
        malformed_packet_count = 0;
      }
      return false;
    }
    return true;
  }

  /// Copies the parts of the current packet the kernel needs into the next ring slot
  void store_packet(int n_returns)
  {
    current_slot = next_slot++;

    // A scan whose packets would be overwritten (the scan was never cut, e.g. the sensor stopped
    // spinning) cannot be decoded any more; drop what was collected for it.
    for (auto & frame : frames) {
      if (!frame.block_groups.empty() && current_slot - frame.first_slot >= k_ring_size) {
        frame.block_groups.clear();
        const auto now = std::chrono::steady_clock::now();
        if (now - last_overflow_log >= k_log_interval) {
          logger->error("Scan longer than the packet buffer, dropping its points");
          last_overflow_log = now;
        }
      }
    }

    cuda::CudaPacket & slot = ring[current_slot % k_ring_size];
    slot.dis_unit = hesai_packet::get_dis_unit(packet);
    slot.return_mode = packet.tail.return_mode;
    for (size_t block_id = 0; block_id < packet_t::n_blocks; ++block_id) {
      slot.azimuth[block_id] = static_cast<uint16_t>(packet.body.blocks[block_id].get_azimuth());
      slot.time_table_index[block_id] = static_cast<uint8_t>(time_table_index(
        n_returns, packet.tail.operational_state, packet.tail.get_azimuth_state(block_id)));
      std::memcpy(
        slot.units[block_id], packet.body.blocks[block_id].units, sizeof(slot.units[block_id]));
    }
    current_packet_timestamp_ns = sensor.get_packet_timestamp_ns(packet);
  }

  /// Takes the place of HesaiDecoder::convert_returns: records which channels of this group of
  /// returns go into which scan
  void stage_returns(
    size_t start_block_id, size_t n_blocks,
    const typename decltype(scan_cutter)::State & scan_state)
  {
    for (uint8_t buffer_index = 0; buffer_index < frames.size(); ++buffer_index) {
      cuda::CudaBlockGroup group{};
      bool any_channel = false;
      for (size_t channel_id = 0; channel_id < packet_t::n_channels; ++channel_id) {
        if (
          scan_state.channels_in_fov[channel_id] &&
          scan_state.channel_buffer_indices[channel_id] == buffer_index) {
          group.channel_mask[channel_id / 32] |= 1U << (channel_id % 32);
          any_channel = true;
        }
      }
      if (!any_channel) {
        continue;
      }

      auto & frame = frames[buffer_index];
      if (frame.block_groups.empty()) {
        frame.first_slot = current_slot;
      }
      group.packet_index = static_cast<uint32_t>(current_slot - frame.first_slot);
      group.start_block = static_cast<uint8_t>(start_block_id);
      group.n_blocks = static_cast<uint8_t>(n_blocks);
      group.packet_to_scan_ns = static_cast<int64_t>(current_packet_timestamp_ns) -
                                static_cast<int64_t>(frame.scan_timestamp_ns);
      frame.block_groups.push_back(group);
    }
  }

  void on_scan_complete(uint8_t buffer_index)
  {
    did_scan_complete = true;

    auto & frame = frames[buffer_index];
    constexpr uint64_t nanoseconds_per_second = 1'000'000'000ULL;
    double scan_timestamp_s =
      static_cast<double>(frame.scan_timestamp_ns / nanoseconds_per_second) +
      (static_cast<double>(frame.scan_timestamp_ns % nanoseconds_per_second) / 1e9);

    size_t n_points = 0;
    if (!frame.block_groups.empty()) {
      // The scan's packets run from its first slot to the current packet, possibly wrapping
      const uint64_t n_packets = next_slot - frame.first_slot;
      const uint64_t start = frame.first_slot % k_ring_size;
      const uint64_t first_size = std::min(n_packets, k_ring_size - start);
      n_points = builder->build(
        {ring + start, first_size}, {ring, n_packets - first_size}, frame.block_groups);
    }

    util::Stopwatch stopwatch;
    if (cuda_pointcloud_callback) {
      cuda_pointcloud_callback(
        CudaDecodedScan{builder->device_points(), n_points, builder->stream()}, scan_timestamp_s);
    }
    if (pointcloud_callback) {
      auto pointcloud = std::make_shared<NebulaPointCloud>();
      pointcloud->resize(n_points);
      if (n_points > 0) {
        cudaError_t error = cudaMemcpyAsync(
          pointcloud->data(), builder->device_points(), n_points * sizeof(NebulaPoint),
          cudaMemcpyDeviceToHost, builder->stream());
        if (error == cudaSuccess) {
          error = cudaStreamSynchronize(builder->stream());
        }
        if (error != cudaSuccess) {
          throw std::runtime_error(std::string("cudaMemcpy scan: ") + cudaGetErrorString(error));
        }
      }
      pointcloud_callback(pointcloud, scan_timestamp_s);
    }
    callback_time_ns += stopwatch.elapsed_ns();

    frame.block_groups.clear();
  }

  void on_set_timestamp(uint8_t buffer_index)
  {
    auto & frame = frames[buffer_index];
    frame.scan_timestamp_ns = sensor.get_packet_timestamp_ns(packet);
    frame.scan_timestamp_ns +=
      sensor.get_earliest_point_time_offset_for_block(current_block_id, packet);
  }

  PacketDecodeResult unpack(const std::vector<uint8_t> & raw_packet)
  {
    util::Stopwatch decode_watch;
    callback_time_ns = 0;
    did_scan_complete = false;

    if (!parse_packet(raw_packet)) {
      return {PerformanceCounters{decode_watch.elapsed_ns()}, DecodeError::PACKET_PARSE_FAILED};
    }

    if (packet_loss_detector) {
      packet_loss_detector->update(packet);
    }
    if (imu_decoder) {
      imu_decoder->update(packet);
    }
    if (functional_safety_decoder) {
      functional_safety_decoder->update(packet);
    }

    const size_t n_returns = hesai_packet::get_n_returns(packet.tail.return_mode);
    if (n_returns > packet_t::n_blocks) {
      return {PerformanceCounters{decode_watch.elapsed_ns()}, DecodeError::PACKET_PARSE_FAILED};
    }

    store_packet(static_cast<int>(n_returns));

    for (size_t block_id = 0; block_id < packet_t::n_blocks; block_id += n_returns) {
      auto block_azimuth = packet.body.blocks[block_id].get_azimuth();
      auto channel_azimuths_out = angle_corrector.get_corrected_azimuths(block_azimuth);
      current_block_id = block_id;
      const auto & scan_state = scan_cutter.step(channel_azimuths_out);

      if (scan_state.does_block_intersect_fov()) {
        stage_returns(block_id, n_returns, scan_state);
      }
    }

    uint64_t decode_duration_ns = decode_watch.elapsed_ns();

    PacketMetadata metadata;
    metadata.packet_timestamp_ns = sensor.get_packet_timestamp_ns(packet);
    metadata.did_scan_complete = did_scan_complete;
    return {PerformanceCounters{decode_duration_ns - callback_time_ns}, metadata};
  }
};

std::optional<std::string> HesaiCudaDecoder128E4X::unsupported_reason(
  const HesaiSensorConfiguration & config)
{
  if (config.sensor_model != SensorModel::HESAI_PANDAR128_E4X) {
    return "the CUDA decoder only supports the OT128 (Pandar128E4X)";
  }
  if (config.downsample_mask_path) {
    return "the downsample mask filter is not supported by the CUDA decoder";
  }
  if (config.blockage_mask_horizontal_bin_size_mdeg) {
    return "the blockage mask is not supported by the CUDA decoder";
  }
  return std::nullopt;
}

HesaiCudaDecoder128E4X::HesaiCudaDecoder128E4X(
  const std::shared_ptr<const HesaiSensorConfiguration> & sensor_configuration,
  const std::shared_ptr<const HesaiCalibrationConfigurationBase> & calibration_configuration,
  const std::shared_ptr<loggers::Logger> & logger, FunctionalSafetyDecoderBase::alive_cb_t alive_cb,
  FunctionalSafetyDecoderBase::stuck_cb_t stuck_cb,
  FunctionalSafetyDecoderBase::status_cb_t status_cb, PacketLossDetectorBase::lost_cb_t lost_cb,
  ImuDecoderBase::imu_cb_t imu_cb)
{
  if (!sensor_configuration) {
    throw std::runtime_error("HesaiCudaDecoder128E4X needs a sensor configuration");
  }
  if (auto reason = unsupported_reason(*sensor_configuration)) {
    throw std::runtime_error(*reason);
  }
  auto calibration =
    std::dynamic_pointer_cast<const HesaiCalibrationConfiguration>(calibration_configuration);
  if (!calibration) {
    throw std::runtime_error("HesaiCudaDecoder128E4X needs a HesaiCalibrationConfiguration");
  }

  impl_ = std::make_unique<Impl>(sensor_configuration, calibration, logger);

  // As in HesaiDriver::initialize_decoder
  if constexpr (hesai_packet::HasFunctionalSafety<packet_t>::value) {
    auto decoder =
      std::make_shared<FunctionalSafetyDecoder<packet_t>>(sensor_configuration->rotation_speed);
    decoder->set_alive_callback(std::move(alive_cb));
    decoder->set_stuck_callback(std::move(stuck_cb));
    decoder->set_status_callback(std::move(status_cb));
    impl_->functional_safety_decoder = decoder;
  }
  if constexpr (hesai_packet::HasPacketLossDetection<packet_t>::value) {
    auto detector = std::make_shared<PacketLossDetector<packet_t>>();
    detector->set_lost_callback(std::move(lost_cb));
    impl_->packet_loss_detector = detector;
  }
  if constexpr (hesai_packet::HasImu<packet_t>::value) {
    auto decoder = std::make_shared<ImuDecoder<packet_t>>();
    decoder->set_imu_callback(std::move(imu_cb));
    impl_->imu_decoder = decoder;
  }
}

HesaiCudaDecoder128E4X::~HesaiCudaDecoder128E4X() = default;

void HesaiCudaDecoder128E4X::set_pointcloud_callback(pointcloud_callback_t callback)
{
  impl_->pointcloud_callback = std::move(callback);
}

void HesaiCudaDecoder128E4X::set_cuda_pointcloud_callback(cuda_pointcloud_callback_t callback)
{
  impl_->cuda_pointcloud_callback = std::move(callback);
}

PacketDecodeResult HesaiCudaDecoder128E4X::unpack(const std::vector<uint8_t> & packet)
{
  return impl_->unpack(packet);
}

}  // namespace nebula::drivers
