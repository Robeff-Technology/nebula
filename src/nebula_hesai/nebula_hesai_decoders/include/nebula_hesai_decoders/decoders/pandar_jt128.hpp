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
#include "nebula_hesai_decoders/decoders/hesai_sensor.hpp"
#include "nebula_hesai_decoders/decoders/pandar_128e3x.hpp"

#include <cstdint>
#include <vector>

namespace nebula::drivers
{

namespace hesai_packet
{

#pragma pack(push, 1)

// The JT128 packet layout was determined by capturing live packets from a real sensor and
// checking which combination of unit size and tail layout produced a valid `return_mode` byte at
// the expected offset (the sensor was configured with `return_mode: LastStrongest`, which only
// this layout decodes into a valid `ReturnMode::DUAL_LAST_STRONGEST` value):
//  - Unlike Pandar128/OT128 (`Packet128E3X`, which uses 3-byte `Unit3B`), the JT128 uses 4-byte
//    `Unit4B` per-point units.
//  - Unlike Pandar128/OT128, the JT128 packet's header `flags` byte has the "functional safety"
//    bit unset, and the packet does not contain a `FunctionalSafety128E3X` section at all.
// Everything else (header, tail layout/fields) matched `Header12B`/`Tail128E3X` exactly.
struct PacketJT128 : public PacketBase<2, 128, 2, 100>
{
  using body_t = BodyWithCrc<Block<Unit4B, PacketJT128::n_channels>, PacketJT128::n_blocks>;
  Header12B header;
  body_t body;
  Tail128E3X tail;
};

#pragma pack(pop)

}  // namespace hesai_packet

/// @brief The JT128 sensor.
///
/// Confirmed against HesaiLidar_SDK_2.0 (`libhesai/UdpParser/include/udp1_4_parser.h`, the
/// `STR_OTHER`/JT128 branches, cross-checked with a live packet capture):
///  - Packet framing: see `PacketJT128` above.
///  - FoV: elevation -4.4 to +90.5 deg, azimuth resolution 0.4 deg / elevation resolution 0.73 deg
///    (`default_remake_config` for `STR_OTHER` in the vendor SDK).
///  - Block timing offset: `OTHER_BLOCK_NS_OFFSET1 + OTHER_BLOCK_NS_OFFSET2 * ((n_blocks -
///    block_id - 1) / n_returns)` (vendor SDK `PandarN::OTHER_BLOCK_NS_OFFSET{1,2}`, used in the
///    `STR_OTHER` branch of `Udp1_4Parser::ComputeXYZI`).
///
/// Per-channel firing time correction: unlike OT128/128E3X (whose per-channel offset depends on
/// azimuth state and near/far distance), the vendor SDK's `STR_OTHER`/JT128 branch of
/// `Udp1_4Parser::GetFiretimes` uses a single fixed nanosecond constant per channel, loaded from an
/// external "firetimes" file and added directly to the point timestamp
/// (`timestamp += block_ns_offset + GetFiretimes(...)`, no dependency on motor speed, operational
/// mode, or distance for this sensor). The values below are taken verbatim from Hesai's own
/// "JT128_Firetime Correction File.csv" (vendored at
/// `HesaiLidar_ROS_2.0/src/driver/HesaiLidar_SDK_2.0/correction/firetime_correction/`), converted
/// from microseconds to nanoseconds. The vendor SDK separately uses the same per-channel value to
/// apply a small azimuth correction (to account for sensor rotation during the firing delay,
/// `Udp1_4Parser::GetFiretimesCorrection`) -- Nebula's angle-correction architecture has no hook for
/// a motor-speed-dependent per-channel azimuth shift for ANY Hesai sensor (not just JT128), so this
/// is not applied; only the point timestamp is corrected here.
///
/// NOT verified / not implemented:
///  - `get_return_type` ordering: the vendor SDK's `STR_OTHER` branch does not implement any
///    explicit return-type/duplicate classification (that is Nebula's own abstraction), so there
///    is no reference to confirm whether JT128 needs the same first/last flip that OT128/128E3X
///    need. Left as the `HesaiSensor` base class default (no flip) absent evidence either way.
///
/// Confirmed on real hardware (2026-07-16, PC as PTP grandmaster over 1588v2/UDP, PTP locked with
/// offsetFromMaster/meanPathDelay both 0): the JT128 populates the packet's `date_time` field
/// (nominally UTC, per both this protocol's field naming and the vendor SDK's own `m_u8UTC` name)
/// with TAI time instead, i.e. it does not apply the PTP `currentUtcOffset` correction. This makes
/// its timestamp exactly (to the millisecond, over repeated live samples) 37s ahead of true UTC --
/// exactly the current TAI-UTC leap-second offset. `get_packet_timestamp_ns` below corrects for
/// this by subtracting the TAI-UTC offset appropriate for the packet's own date (so old recordings
/// decode correctly too), looked up from `tai_minus_utc_seconds`.
class PandarJT128 : public HesaiSensor<hesai_packet::PacketJT128>
{
private:
  // Vendor SDK: `namespace PandarN { OTHER_BLOCK_NS_OFFSET1 = -1888000; OTHER_BLOCK_NS_OFFSET2 =
  // -111000; }` in `libhesai/UdpProtocol/udp_protocol_v1_4.h`.
  static constexpr int32_t other_block_ns_offset1 = -1'888'000;
  static constexpr int32_t other_block_ns_offset2 = -111'000;

  /// @brief TAI-UTC offset (leap seconds) in effect from a given Unix time (UTC), per IERS
  /// Bulletin C. Update this table if IERS ever schedules another leap second (none since
  /// 2017-01-01, and the leap second is slated for discontinuation by 2035).
  [[nodiscard]] static int32_t tai_minus_utc_seconds(int64_t utc_unix_time_s)
  {
    struct LeapSecondEntry
    {
      int64_t effective_unix_time_utc;
      int32_t offset_s;
    };
    static constexpr LeapSecondEntry table[] = {
      {63072000, 10},    // 1972-01-01
      {78796800, 11},    // 1972-07-01
      {94694400, 12},    // 1973-01-01
      {126230400, 13},   // 1974-01-01
      {157766400, 14},   // 1975-01-01
      {189302400, 15},   // 1976-01-01
      {220924800, 16},   // 1977-01-01
      {252460800, 17},   // 1978-01-01
      {283996800, 18},   // 1979-01-01
      {315532800, 19},   // 1980-01-01
      {362793600, 20},   // 1981-07-01
      {394329600, 21},   // 1982-07-01
      {425865600, 22},   // 1983-07-01
      {489024000, 23},   // 1985-07-01
      {567993600, 24},   // 1988-01-01
      {631152000, 25},   // 1990-01-01
      {662688000, 26},   // 1991-01-01
      {709948800, 27},   // 1992-07-01
      {741484800, 28},   // 1993-07-01
      {773020800, 29},   // 1994-07-01
      {820454400, 30},   // 1996-01-01
      {867715200, 31},   // 1997-07-01
      {915148800, 32},   // 1999-01-01
      {1136073600, 33},  // 2006-01-01
      {1230768000, 34},  // 2009-01-01
      {1341100800, 35},  // 2012-07-01
      {1435708800, 36},  // 2015-07-01
      {1483228800, 37},  // 2017-01-01
    };

    int32_t offset = 0;
    for (const auto & entry : table) {
      if (utc_unix_time_s < entry.effective_unix_time_utc) break;
      offset = entry.offset_s;
    }
    return offset;
  }

  // Per-channel firing time offset in nanoseconds, verbatim from Hesai's
  // "JT128_Firetime Correction File.csv" (channel N is index N-1), converted from microseconds.
  static constexpr int32_t channel_firetime_offset_ns[128] = {
    95180,  23240,  98220,  20200, 101260,  17160, 104300,  14120, 77280,  92140,  74240,  89100,
    71200,  86060,  68160,  83020, 50260,  11080,  47220,   8040, 44180,   5000,  41140,   1960,
    65120, 105820,  62080, 102780, 59040,  99740,  56000,  96700, 38100,  24760,  35060,  21720,
    32020,  18680,  28980,  15640, 78800,  93660,  75760,  90620, 72720,  87580,  69680,  84540,
    51780,  12600,  48740,   9560, 45700,   6520,  42660,   3480, 66640, 103540,  63600, 100500,
    60560,  97460,  57520,  94420, 39620,  22480,  36580,  19440, 33540,  16400,  30500,  13360,
    76520,  91380,  73480,  88340, 70440,  85300,  67400,  82260, 49500,  10320,  46460,   7280,
    43420,   4240,  40380,   1200, 64360, 105060,  61320, 102020, 58280,  98980,  55240,  95940,
    37340,  24000,  34300,  20960, 31260,  17920,  28220,  14880, 78040,  92900,  75000,  89860,
    71960,  86820,  68920,  83780, 51020,  11840,  47980,   8800, 44940,   5760,  41900,   2720,
    65880,  62840,  59800,  56760, 38860,  35820,  32780,  29740,
  };

public:
  static constexpr float min_range = 0.1;
  static constexpr float max_range = 230.0;
  static constexpr size_t max_scan_buffer_points = 691200;
  static constexpr FieldOfView<int32_t, MilliDegrees> fov_mdeg{{0, 360'000}, {-4'400, 90'500}};
  static constexpr AnglePair<int32_t, MilliDegrees> peak_resolution_mdeg{400, 730};

  int get_packet_relative_point_time_offset(
    uint32_t block_id, uint32_t channel_id, const packet_t & packet) override
  {
    auto n_returns = hesai_packet::get_n_returns(packet.tail.return_mode);
    int n_blocks = static_cast<int>(packet_t::n_blocks);
    int block_id_int = static_cast<int>(block_id);
    int block_offset_ns =
      other_block_ns_offset1 + other_block_ns_offset2 * ((n_blocks - block_id_int - 1) / n_returns);
    return block_offset_ns + channel_firetime_offset_ns[channel_id];
  }

  [[nodiscard]] uint64_t get_packet_timestamp_ns(const packet_t & packet) const override
  {
    uint64_t raw_ns = hesai_packet::get_timestamp_ns(packet);
    int32_t offset_s = tai_minus_utc_seconds(static_cast<int64_t>(raw_ns / 1'000'000'000ULL));
    return raw_ns - static_cast<uint64_t>(offset_s) * 1'000'000'000ULL;
  }
};

}  // namespace nebula::drivers
