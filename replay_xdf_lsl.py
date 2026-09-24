#!/usr/bin/env python3
"""
replay_xdf_lsl.py -- Replay a recorded .xdf EEG file as a live LSL stream.

The point of this script is to make an already-recorded file behave exactly
like a running amplifier, so that every piece of downstream code you write
(2 s windowing, peak alpha frequency, TMS triggering) can be developed and
debugged against reproducible data and then moved to the real amplifier by
changing nothing but the stream name.

Typical use
-----------
    # 1. See what is inside the file
    python replay_xdf_lsl.py mydata.xdf --list

    # 2. Replay stream #0 in real time as an LSL stream called "EEGReplay"
    python replay_xdf_lsl.py mydata.xdf --stream-index 0

    # 3. Replay forever (handy while you tune the analysis loop)
    python replay_xdf_lsl.py mydata.xdf --stream-index 0 --loop

    # 4. Replay at 4x speed for a quick offline sanity check
    python replay_xdf_lsl.py mydata.xdf --stream-index 0 --speed 4

Requirements
------------
    pip install pylsl pyxdf numpy
(Linux may additionally need liblsl: conda install -c conda-forge liblsl,
 or grab the .deb from github.com/sccn/liblsl/releases)
"""

import argparse
import os
import sys
import time

import numpy as np


# --------------------------------------------------------------------------
# XDF loading / inspection
# --------------------------------------------------------------------------

def load_xdf_streams(path):
    """Load all streams from an XDF file."""
    import pyxdf
    if not os.path.exists(path):
        sys.exit(f"File not found: {path}")
    print(f"Loading {path} ... (large files can take a while)")
    # dejitter_timestamps=True fits a linear clock to each stream's timestamps,
    # which removes the transport jitter present in the original recording.
    streams, _header = pyxdf.load_xdf(path, dejitter_timestamps=True)
    print(f"Found {len(streams)} stream(s).\n")
    return streams


def stream_summary(stream):
    """Pull the fields we care about out of a pyxdf stream dict."""
    info = stream["info"]
    ts = stream["time_stamps"]
    data = stream["time_series"]

    def first(key, default=""):
        v = info.get(key, [default])
        return v[0] if v else default

    n_samples = len(ts)
    nominal = float(first("nominal_srate", 0) or 0)
    duration = (ts[-1] - ts[0]) if n_samples > 1 else 0.0
    effective = (n_samples - 1) / duration if duration > 0 else 0.0

    return {
        "name": first("name"),
        "type": first("type"),
        "n_channels": int(first("channel_count", 0) or 0),
        "srate_nominal": nominal,
        "srate_effective": effective,
        "n_samples": n_samples,
        "duration": duration,
        "dtype": type(data).__name__,
        "is_numeric": isinstance(data, np.ndarray),
    }


def channel_labels(stream, n_channels):
    """Extract channel labels from the XDF metadata, with a fallback."""
    labels = []
    try:
        desc = stream["info"]["desc"][0]
        chans = desc["channels"][0]["channel"]
        for ch in chans:
            lbl = ch.get("label", [""])[0]
            labels.append(lbl if lbl else None)
    except (KeyError, TypeError, IndexError):
        pass
    if len(labels) != n_channels or any(l is None for l in labels):
        labels = [f"Ch{i+1}" for i in range(n_channels)]
    return labels


def channel_units(stream, n_channels, default="microvolts"):
    units = []
    try:
        chans = stream["info"]["desc"][0]["channels"][0]["channel"]
        for ch in chans:
            u = ch.get("unit", [""])[0]
            units.append(u if u else default)
    except (KeyError, TypeError, IndexError):
        pass
    if len(units) != n_channels:
        units = [default] * n_channels
    return units


def print_stream_table(streams):
    print(f"{'idx':<4} {'name':<22} {'type':<10} {'ch':>4} {'srate':>8} "
          f"{'eff.srate':>10} {'samples':>10} {'dur(s)':>9}")
    print("-" * 82)
    for i, s in enumerate(streams):
        d = stream_summary(s)
        print(f"{i:<4} {d['name'][:22]:<22} {d['type'][:10]:<10} "
              f"{d['n_channels']:>4} {d['srate_nominal']:>8.1f} "
              f"{d['srate_effective']:>10.3f} {d['n_samples']:>10} "
              f"{d['duration']:>9.1f}")
    print()
    for i, s in enumerate(streams):
        d = stream_summary(s)
        if d["n_channels"]:
            labs = channel_labels(s, d["n_channels"])
            preview = ", ".join(labs[:12]) + (" ..." if len(labs) > 12 else "")
            print(f"  [{i}] channels: {preview}")
    print()


def pick_stream(streams, index=None, name=None, stype=None):
    """Choose which stream to replay."""
    if index is not None:
        if not 0 <= index < len(streams):
            sys.exit(f"--stream-index {index} out of range (0..{len(streams)-1})")
        return streams[index]

    candidates = []
    for i, s in enumerate(streams):
        d = stream_summary(s)
        if name and d["name"] != name:
            continue
        if stype and d["type"].lower() != stype.lower():
            continue
        if not name and not stype:
            # auto mode: numeric, regularly sampled, more than one channel
            if d["is_numeric"] and d["srate_nominal"] > 0 and d["n_channels"] > 0:
                candidates.append((i, s))
            continue
        candidates.append((i, s))

    if not candidates:
        sys.exit("No matching stream. Run with --list to see what is in the file.")
    if len(candidates) > 1:
        print("Multiple candidate streams:")
        for i, s in candidates:
            d = stream_summary(s)
            print(f"  [{i}] {d['name']} ({d['type']}, {d['n_channels']} ch)")
        sys.exit("Be specific: use --stream-index, --stream-name or --stream-type.")
    return candidates[0][1]


# --------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------

def prepare_data(stream, srate_override=None, channels=None, scale=1.0):
    """Return (data[n_samples, n_channels] float32, srate, labels, units)."""
    d = stream_summary(stream)
    data = stream["time_series"]

    if not isinstance(data, np.ndarray):
        sys.exit("Selected stream is not numeric (looks like a marker stream). "
                 "Pick the EEG stream instead.")

    data = np.asarray(data)
    if data.ndim == 1:
        data = data[:, None]

    srate = srate_override or d["srate_nominal"]
    if not srate:
        sys.exit("Stream has no nominal sampling rate; pass --srate 500.")

    labels = channel_labels(stream, data.shape[1])
    units = channel_units(stream, data.shape[1])

    if channels:
        idx = []
        for c in channels:
            if c.isdigit():
                idx.append(int(c))
            elif c in labels:
                idx.append(labels.index(c))
            else:
                sys.exit(f"Unknown channel '{c}'. Available: {labels}")
        data = data[:, idx]
        labels = [labels[i] for i in idx]
        units = [units[i] for i in idx]

    if scale != 1.0:
        data = data * scale

    # Check for recording gaps -- these exist if the amplifier was paused.
    ts = np.asarray(stream["time_stamps"])
    if len(ts) > 2:
        dt = np.diff(ts)
        gaps = np.where(dt > 5.0 / srate)[0]
        if len(gaps):
            print(f"  NOTE: {len(gaps)} gap(s) > 5 samples in the original "
                  f"timestamps (largest {dt[gaps].max():.2f} s). This replay "
                  f"streams the samples back-to-back at a constant rate and "
                  f"will not reproduce those gaps.")

    # float32 is what liblsl wants for cf_float32 and is plenty for EEG in uV.
    data = np.ascontiguousarray(data, dtype=np.float32)
    return data, float(srate), labels, units


# --------------------------------------------------------------------------
# The replay loop
# --------------------------------------------------------------------------

def build_outlet(name, stype, labels, units, srate, source_id, chunk_size):
    from pylsl import StreamInfo, StreamOutlet

    info = StreamInfo(
        name=name,
        type=stype,
        channel_count=len(labels),
        nominal_srate=srate,
        channel_format="float32",
        source_id=source_id,
    )
    chans = info.desc().append_child("channels")
    for lbl, unit in zip(labels, units):
        ch = chans.append_child("channel")
        ch.append_child_value("label", lbl)
        ch.append_child_value("unit", unit)
        ch.append_child_value("type", "EEG")
    info.desc().append_child_value("manufacturer", "replay_xdf_lsl")

    # chunk_size here is the outlet's transmission granularity. Setting it to
    # the same chunk we push keeps liblsl from re-buffering on our behalf.
    return StreamOutlet(info, chunk_size=chunk_size, max_buffered=360)


def precise_sleep(deadline, spin_margin=0.0015):
    """Sleep until `deadline` (a time.perf_counter() value).

    time.sleep() alone oversleeps by a millisecond or more depending on the OS
    scheduler, which at 500 Hz means whole samples of jitter. So we sleep to
    just short of the deadline and busy-wait the rest.
    """
    remaining = deadline - time.perf_counter()
    if remaining > spin_margin:
        time.sleep(remaining - spin_margin)
    while time.perf_counter() < deadline:
        pass


def replay(data, srate, outlet, chunk_samples, speed=1.0, loop=False,
           report_every=5.0):
    from pylsl import local_clock

    n_total, n_ch = data.shape
    chunk_dur = chunk_samples / srate / speed

    print(f"\nStreaming {n_total} samples x {n_ch} ch at {srate:g} Hz "
          f"(speed {speed}x)")
    print(f"Chunk: {chunk_samples} samples = "
          f"{1000*chunk_samples/srate:.1f} ms of data, pushed every "
          f"{1000*chunk_dur:.1f} ms wall clock")
    print(f"Total duration: {n_total/srate/speed:.1f} s"
          + (" per pass, looping" if loop else ""))
    print("Ctrl-C to stop.\n")

    pass_no = 0
    worst_late = 0.0
    try:
        while True:
            pass_no += 1
            t0_perf = time.perf_counter()
            t0_lsl = local_clock()
            pos = 0
            next_report = report_every
            late_sum, late_n = 0.0, 0

            while pos < n_total:
                end = min(pos + chunk_samples, n_total)

                # Wait until the moment the *last* sample of this chunk would
                # have been acquired by a real amplifier.
                due_perf = t0_perf + (end / srate) / speed
                precise_sleep(due_perf)

                late = time.perf_counter() - due_perf
                late_sum += late
                late_n += 1
                worst_late = max(worst_late, late)

                # Deterministic timestamp for the last sample in the chunk.
                # liblsl back-dates the earlier samples using nominal_srate,
                # so the consumer sees a perfectly regular time base -- the
                # same thing a hardware-clocked amplifier gives you.
                stamp = t0_lsl + ((end - 1) / srate) / speed

                outlet.push_chunk(data[pos:end], stamp)
                pos = end

                elapsed = time.perf_counter() - t0_perf
                if elapsed >= next_report:
                    mean_late = 1000 * late_sum / max(late_n, 1)
                    print(f"  t={elapsed:6.1f}s  {pos:>9}/{n_total} samples  "
                          f"mean lateness {mean_late:5.2f} ms  "
                          f"worst {1000*worst_late:5.2f} ms  "
                          f"consumers={outlet.have_consumers()}")
                    next_report += report_every
                    late_sum, late_n = 0.0, 0

            total = time.perf_counter() - t0_perf
            print(f"Pass {pass_no} done in {total:.2f} s "
                  f"(expected {n_total/srate/speed:.2f} s, "
                  f"effective rate {n_total/total:.2f} Hz)")

            if not loop:
                break
            print("Looping...\n")

    except KeyboardInterrupt:
        print("\nStopped by user.")


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Replay an XDF recording as a live LSL stream.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("xdf", help="path to the .xdf file")
    p.add_argument("--list", action="store_true",
                   help="print the streams in the file and exit")

    sel = p.add_argument_group("stream selection")
    sel.add_argument("--stream-index", type=int, default=None)
    sel.add_argument("--stream-name", default=None)
    sel.add_argument("--stream-type", default=None,
                     help="e.g. EEG")
    sel.add_argument("--channels", nargs="+", default=None,
                     help="subset of channels, by label or 0-based index")

    out = p.add_argument_group("output stream")
    out.add_argument("--name", default="EEGReplay",
                     help="LSL stream name to advertise (default: EEGReplay)")
    out.add_argument("--type", dest="out_type", default="EEG")
    out.add_argument("--source-id", default=None)

    tim = p.add_argument_group("timing")
    tim.add_argument("--srate", type=float, default=None,
                     help="override the nominal sampling rate")
    tim.add_argument("--chunk-ms", type=float, default=20.0,
                     help="chunk size in ms of data (default 20 = 10 samples "
                          "at 500 Hz). Smaller = lower latency, more overhead.")
    tim.add_argument("--speed", type=float, default=1.0,
                     help="playback speed multiplier (1.0 = real time)")
    tim.add_argument("--loop", action="store_true",
                     help="restart from the beginning when the file ends")
    tim.add_argument("--wait-for-consumer", action="store_true",
                     help="don't start streaming until something connects")
    tim.add_argument("--scale", type=float, default=1.0,
                     help="multiply the data (e.g. 1e6 if stored in volts)")

    args = p.parse_args()

    streams = load_xdf_streams(args.xdf)
    if args.list:
        print_stream_table(streams)
        return

    print_stream_table(streams)
    stream = pick_stream(streams, args.stream_index,
                         args.stream_name, args.stream_type)
    data, srate, labels, units = prepare_data(
        stream, args.srate, args.channels, args.scale)

    chunk_samples = max(1, int(round(args.chunk_ms * srate / 1000.0)))
    source_id = args.source_id or f"xdfreplay_{os.path.basename(args.xdf)}"

    print(f"\nSelected: {stream_summary(stream)['name']} -> "
          f"advertising as '{args.name}' ({args.out_type})")
    print(f"  {data.shape[1]} channels: {', '.join(labels)}")
    print(f"  {data.shape[0]} samples @ {srate:g} Hz "
          f"= {data.shape[0]/srate:.1f} s")
    print(f"  amplitude range: {data.min():.2f} to {data.max():.2f} {units[0]}")

    outlet = build_outlet(args.name, args.out_type, labels, units,
                          srate, source_id, chunk_samples)

    if args.wait_for_consumer:
        print("\nWaiting for a consumer to connect...")
        while not outlet.have_consumers():
            time.sleep(0.1)
        print("Consumer connected.")
    else:
        # Give liblsl a moment to announce itself on the network.
        time.sleep(0.5)

    replay(data, srate, outlet, chunk_samples,
           speed=args.speed, loop=args.loop)


if __name__ == "__main__":
    main()
