// Copyright 2026 TIER IV, Inc.
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

#include "nebula_hesai_decoders/decoders/hesai_packet.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <optional>
#include <utility>

namespace nebula::drivers
{

/// @brief One IMU measurement, converted to ROS-standard (REP-103) units
struct ImuSample
{
  /// @brief Timestamp of the packet that carried this sample, in nanoseconds since epoch
  uint64_t timestamp_ns;
  /// @brief Linear acceleration in m/s^2
  double acceleration_x;
  double acceleration_y;
  double acceleration_z;
  /// @brief Angular velocity in rad/s
  double angular_velocity_x;
  double angular_velocity_y;
  double angular_velocity_z;
};

class ImuDecoderBase
{
public:
  using imu_cb_t = std::function<void(const ImuSample & sample)>;

  virtual ~ImuDecoderBase() = default;

  virtual void set_imu_callback(imu_cb_t /* on_imu */) {}
};

template <typename PacketT>
class ImuDecoderTypedBase : public ImuDecoderBase
{
public:
  virtual void update(const PacketT & /* packet */) {}
};

/**
 * @brief Interprets the IMU data in the packet tail of Hesai pointcloud packets for supported
 * sensor models.
 *
 * @tparam PacketT A packet definition that has an IMU section in its tail.
 */
template <typename PacketT>
class ImuDecoder : public ImuDecoderTypedBase<PacketT>
{
  using imu_data_t = decltype(std::declval<PacketT>().tail.imu);

  /// @brief Header flag signaling that the optional IMU section is present in the packet
  static constexpr uint8_t flag_imu_present = 0x02;

  static constexpr double g_to_m_s2 = 9.80665;
  static constexpr double deg_to_rad = M_PI / 180.;

public:
  using imu_cb_t = ImuDecoderBase::imu_cb_t;

  void set_imu_callback(imu_cb_t on_imu) override { on_imu_ = std::move(on_imu); }

  void update(const PacketT & packet) override
  {
    if (!on_imu_) return;
    if (!(packet.header.flags & flag_imu_present)) return;

    const imu_data_t & imu = packet.tail.imu;

    // The sensor sends its latest IMU sample with every packet, at a much higher packet rate than
    // the IMU sampling rate. Only forward new samples.
    if (last_data_ && std::memcmp(&*last_data_, &imu, sizeof(imu_data_t)) == 0) {
      return;
    }
    last_data_.emplace(imu);

    ImuSample sample;
    sample.timestamp_ns = hesai_packet::get_timestamp_ns(packet);
    sample.acceleration_x = imu.get_acceleration_x_g() * g_to_m_s2;
    sample.acceleration_y = imu.get_acceleration_y_g() * g_to_m_s2;
    sample.acceleration_z = imu.get_acceleration_z_g() * g_to_m_s2;
    sample.angular_velocity_x = imu.get_angular_velocity_x_deg_s() * deg_to_rad;
    sample.angular_velocity_y = imu.get_angular_velocity_y_deg_s() * deg_to_rad;
    sample.angular_velocity_z = imu.get_angular_velocity_z_deg_s() * deg_to_rad;

    on_imu_(sample);
  }

private:
  std::optional<imu_data_t> last_data_{};
  imu_cb_t on_imu_;
};

}  // namespace nebula::drivers
