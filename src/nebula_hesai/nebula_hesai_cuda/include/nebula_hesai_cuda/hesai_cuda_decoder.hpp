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

#include <nebula_core_common/loggers/logger.hpp>
#include <nebula_hesai_common/hesai_common.hpp>
#include <nebula_hesai_decoders/decoders/functional_safety.hpp>
#include <nebula_hesai_decoders/decoders/hesai_scan_decoder.hpp>
#include <nebula_hesai_decoders/decoders/imu_decoder.hpp>
#include <nebula_hesai_decoders/decoders/packet_loss_detector.hpp>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace nebula::drivers
{

/// A decoded scan in GPU memory, in the NebulaPoint (PointXYZIRCAEDT) layout. Valid only during the
/// callback; the work on `stream` has finished when the callback runs.
struct CudaDecodedScan
{
  const cuda::CudaPoint * points{nullptr};
  std::size_t n_points{0};
  cudaStream_t stream{nullptr};
};

/// @brief OT128 (Pandar128E4X) decoder that generates the points on the GPU.
///
/// The packet bookkeeping stays on the CPU and is the same code as in HesaiDecoder: packet parsing,
/// functional safety, packet loss and IMU decoding, and the scan cutter, which runs once per block.
/// Per block, only the channels that fall into each scan are recorded. When a scan is cut, one
/// kernel turns all its packets into points, in the CPU decoder's order and with its arithmetic,
/// and the scan is handed out in GPU memory.
///
/// Not supported (see unsupported_reason()): the blockage mask and the downsample mask filters.
class HesaiCudaDecoder128E4X : public HesaiScanDecoder
{
public:
  using cuda_pointcloud_callback_t =
    std::function<void(const CudaDecodedScan & scan, double timestamp_s)>;

  /// @return Why this configuration cannot be decoded on the GPU, or nullopt if it can
  static std::optional<std::string> unsupported_reason(const HesaiSensorConfiguration & config);

  HesaiCudaDecoder128E4X(
    const std::shared_ptr<const HesaiSensorConfiguration> & sensor_configuration,
    const std::shared_ptr<const HesaiCalibrationConfigurationBase> & calibration_configuration,
    const std::shared_ptr<loggers::Logger> & logger,
    FunctionalSafetyDecoderBase::alive_cb_t alive_cb = nullptr,
    FunctionalSafetyDecoderBase::stuck_cb_t stuck_cb = nullptr,
    FunctionalSafetyDecoderBase::status_cb_t status_cb = nullptr,
    PacketLossDetectorBase::lost_cb_t lost_cb = nullptr, ImuDecoderBase::imu_cb_t imu_cb = nullptr);
  ~HesaiCudaDecoder128E4X() override;

  /// Scans in host memory (copied back from the GPU), e.g. for comparing with the CPU decoder
  void set_pointcloud_callback(pointcloud_callback_t callback) override;

  /// Scans in GPU memory
  void set_cuda_pointcloud_callback(cuda_pointcloud_callback_t callback);

  PacketDecodeResult unpack(const std::vector<uint8_t> & packet) override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace nebula::drivers
