#!/usr/bin/env python3
"""
list_lsl_streams.py -- Show every LSL stream currently on the network.

Use this to find the exact stream name your amplifier advertises, so you can
pass it to paf_realtime.py --name. Stream names are case-sensitive and often
include a device serial number, so read it off here rather than guessing.

Usage:
    python list_lsl_streams.py
    python list_lsl_streams.py --timeout 5 --watch
"""

import argparse
import time


def describe(info, inlet_labels=True):
    """Print one stream, and its channel labels if we can fetch them."""
    print(f"  name      : {info.name()}")
    print(f"  type      : {info.type()}")
    print(f"  channels  : {info.channel_count()}")
    print(f"  srate     : {info.nominal_srate():g} Hz"
          + ("  (irregular / event stream)" if info.nominal_srate() == 0 else ""))
    print(f"  format    : {info.channel_format()}")
    print(f"  source_id : {info.source_id()}")
    print(f"  host      : {info.hostname()}")

    if not inlet_labels or info.nominal_srate() == 0:
        print()
        return

    # Channel labels live in the full info, which requires opening an inlet.
    try:
        from pylsl import StreamInlet
        inlet = StreamInlet(info, max_buflen=1)
        full = inlet.info(timeout=3.0)
        ch = full.desc().child("channels").child("channel")
        labels = []
        for _ in range(full.channel_count()):
            labels.append(ch.child_value("label"))
            ch = ch.next_sibling()
        inlet.close_stream()
        if labels and any(labels):
            shown = ", ".join(l for l in labels if l)
            print(f"  labels    : {shown}")
        else:
            print("  labels    : (none advertised)")
    except Exception as e:
        print(f"  labels    : (could not read: {e})")
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timeout", type=float, default=3.0,
                   help="seconds to wait for streams to announce themselves")
    p.add_argument("--watch", action="store_true",
                   help="keep re-scanning until Ctrl-C")
    p.add_argument("--no-labels", action="store_true",
                   help="skip opening inlets to read channel labels")
    args = p.parse_args()

    from pylsl import resolve_streams

    try:
        while True:
            print(f"\nScanning for {args.timeout:g} s ...\n")
            streams = resolve_streams(wait_time=args.timeout)
            if not streams:
                print("  No LSL streams found.")
                print("  - Is the amplifier's LSL app running?")
                print("  - Same network / no VPN or firewall in the way?")
                print("  - For replay: is replay_xdf_lsl.py running?")
            else:
                print(f"Found {len(streams)} stream(s):\n")
                for i, s in enumerate(streams):
                    print(f"[{i}]")
                    describe(s, inlet_labels=not args.no_labels)
            if not args.watch:
                break
            time.sleep(2.0)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
