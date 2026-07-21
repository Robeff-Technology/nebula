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

#include "nebula_hesai_decoders/decoders/hesai_packet.hpp"
#include "nebula_hesai_decoders/decoders/imu_decoder.hpp"
#include "nebula_hesai_decoders/decoders/pandar_128e4x.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <vector>

using nebula::drivers::ImuDecoder;
using nebula::drivers::ImuSample;
using nebula::drivers::hesai_packet::Packet128E4X;

namespace
{

constexpr uint8_t flag_imu_present = 0x02;

Packet128E4X make_packet_with_imu()
{
  Packet128E4X packet{};
  packet.header.flags = flag_imu_present;
  packet.tail.date_time.year = 126;  // 2026
  packet.tail.date_time.month = 7;
  packet.tail.date_time.day = 16;
  packet.tail.timestamp = 123456;  // microseconds past date_time

  packet.tail.imu.acceleration_unit = 1000;      // = 0.001 g/LSB
  packet.tail.imu.angular_velocity_unit = 1000;  // = 0.01 (deg/s)/LSB
  packet.tail.imu.timestamp = 40;                // = 1 ms
  packet.tail.imu.x_axis_acceleration = 1000;    // = 1 g
  packet.tail.imu.y_axis_acceleration = -500;    // = -0.5 g
  packet.tail.imu.z_axis_acceleration = 0;
  packet.tail.imu.x_axis_angular_velocity = 9000;   // = 90 deg/s
  packet.tail.imu.y_axis_angular_velocity = -9000;  // = -90 deg/s
  packet.tail.imu.z_axis_angular_velocity = 0;
  return packet;
}

}  // namespace

TEST(ImuDecoderTest, ConvertsToSiUnits)
{
  auto packet = make_packet_with_imu();

  ImuDecoder<Packet128E4X> decoder;
  std::optional<ImuSample> sample;
  decoder.set_imu_callback([&](const ImuSample & s) { sample = s; });
  decoder.update(packet);

  ASSERT_TRUE(sample.has_value());
  EXPECT_EQ(sample->timestamp_ns, nebula::drivers::hesai_packet::get_timestamp_ns(packet));
  EXPECT_DOUBLE_EQ(sample->acceleration_x, 9.80665);
  EXPECT_DOUBLE_EQ(sample->acceleration_y, -0.5 * 9.80665);
  EXPECT_DOUBLE_EQ(sample->acceleration_z, 0.);
  EXPECT_DOUBLE_EQ(sample->angular_velocity_x, M_PI / 2);
  EXPECT_DOUBLE_EQ(sample->angular_velocity_y, -M_PI / 2);
  EXPECT_DOUBLE_EQ(sample->angular_velocity_z, 0.);
}

TEST(ImuDecoderTest, DeduplicatesRepeatedSamples)
{
  auto packet = make_packet_with_imu();

  ImuDecoder<Packet128E4X> decoder;
  std::vector<ImuSample> samples;
  decoder.set_imu_callback([&](const ImuSample & s) { samples.push_back(s); });

  // The same IMU sample arriving in multiple packets is only forwarded once,
  // even if the packet timestamp advances
  decoder.update(packet);
  packet.tail.timestamp += 100;
  decoder.update(packet);
  EXPECT_EQ(samples.size(), 1U);

  // A new sample (identical values, newer IMU timestamp) is forwarded
  packet.tail.imu.timestamp += 200;  // 5 ms at 25 us/LSB
  decoder.update(packet);
  EXPECT_EQ(samples.size(), 2U);

  // Changed measurement values are forwarded
  packet.tail.imu.z_axis_acceleration = 1000;
  decoder.update(packet);
  ASSERT_EQ(samples.size(), 3U);
  EXPECT_DOUBLE_EQ(samples.back().acceleration_z, 9.80665);
}

TEST(ImuDecoderTest, IgnoresPacketsWithoutImuFlag)
{
  auto packet = make_packet_with_imu();
  packet.header.flags = 0x00;

  ImuDecoder<Packet128E4X> decoder;
  std::vector<ImuSample> samples;
  decoder.set_imu_callback([&](const ImuSample & s) { samples.push_back(s); });
  decoder.update(packet);

  EXPECT_TRUE(samples.empty());
}
