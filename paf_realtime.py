#!/usr/bin/env python3
"""
paf_realtime.py -- Real-time peak alpha frequency (PAF) and amplitude from an
LSL EEG stream.

Consumes the stream produced by replay_xdf_lsl.py (or, later, your real
amplifier -- change nothing but --name), maintains a sliding window, and on
every hop estimates:

    paf_peak   peak alpha frequency by argmax of the 1/f-corrected spectrum,
               refined by parabolic interpolation  [Hz]
    paf_cog    peak alpha frequency by centre of gravity of the 1/f-corrected
               spectrum -- more stable, slightly biased low  [Hz]
    amp_uv     amplitude of the alpha oscillation, i.e. the A in A*sin(2*pi*f*t)
               [microvolts]
    rms_uv     RMS amplitude in the alpha band around the peak  [microvolts]
    snr_db     how far the peak stands above the 1/f background  [dB]
    exponent   slope of the aperiodic background in log-log space

Results are printed, optionally logged to CSV, and pushed out as a second LSL
stream so your TMS trigger logic can be a separate process that subscribes to
this one.

Validate before you trust it:
    python paf_realtime.py --selftest

Typical use (with replay_xdf_lsl.py running in another terminal):
    python paf_realtime.py --name EEGReplay
    python paf_realtime.py --name EEGReplay --plot
    python paf_realtime.py --name EEGReplay --channels O1 O2 Pz --csv paf.csv

Requirements:  pip install pylsl numpy scipy   (matplotlib only for --plot)
"""

import argparse
import csv
import sys
import time
from collections import deque

import numpy as np
from scipy import signal

# Channels used by default, if present in the montage. PAF is a posterior
# phenomenon -- including frontal channels drags the estimate around.
DEFAULT_POSTERIOR = ["O1", "O2", "OZ", "PO3", "PO4", "POZ", "PO7", "PO8",
                     "P3", "P4", "PZ", "P7", "P8"]

STATUS_CODES = {"OK": 0.0, "NO_PEAK": 1.0, "ARTIFACT": 2.0, "FILLING": 3.0}


# --------------------------------------------------------------------------
# Ring buffer
# --------------------------------------------------------------------------

class RingBuffer:
    """Most recent N samples, oldest first when you call .get()."""

    def __init__(self, n_samples, n_channels, dtype=np.float32):
        self.buf = np.zeros((n_samples, n_channels), dtype=dtype)
        self.n = n_samples
        self.written = 0
        self.pos = 0

    def push(self, chunk):
        m = chunk.shape[0]
        if m >= self.n:
            self.buf[:] = chunk[-self.n:]
            self.pos = 0
            self.written += m
            return
        end = self.pos + m
        if end <= self.n:
            self.buf[self.pos:end] = chunk
        else:
            split = self.n - self.pos
            self.buf[self.pos:] = chunk[:split]
            self.buf[:end - self.n] = chunk[split:]
        self.pos = end % self.n
        self.written += m

    @property
    def full(self):
        return self.written >= self.n

    def get(self):
        if not self.full:
            return None
        return np.concatenate((self.buf[self.pos:], self.buf[:self.pos]), axis=0)


# --------------------------------------------------------------------------
# Spectral estimation
# --------------------------------------------------------------------------

def next_pow2(n):
    return 1 << (int(n) - 1).bit_length()


def compute_psd(x, fs, taper="hann", nfft=None, nw=2.0, k=3):
    """One-sided PSD in units^2/Hz.

    x    : (n_samples, n_channels), already detrended
    Zero-padding does not change total power, it only interpolates the
    spectrum so the peak can be located between bins. Verified: a 10 uV
    sinusoid integrates to exactly 50 uV^2 at any nfft.
    """
    n = x.shape[0]
    if nfft is None:
        nfft = next_pow2(8 * n)

    if taper == "multitaper":
        tapers = signal.windows.dpss(n, nw, k)
        psd = None
        for w in tapers:
            X = np.fft.rfft(x * w[:, None], n=nfft, axis=0)
            p = np.abs(X) ** 2 / (fs * np.sum(w ** 2))
            psd = p if psd is None else psd + p
        psd /= len(tapers)
    else:
        w = signal.windows.hann(n, sym=False)
        X = np.fft.rfft(x * w[:, None], n=nfft, axis=0)
        psd = np.abs(X) ** 2 / (fs * np.sum(w ** 2))

    # One-sided: fold negative frequencies into the positive ones.
    if nfft % 2 == 0:
        psd[1:-1] *= 2
    else:
        psd[1:] *= 2

    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    return freqs, psd


def fit_aperiodic(freqs, psd, fit_range=(2.0, 40.0), exclude=(6.0, 15.0)):
    """Fit the 1/f background as a straight line in log10-log10 space.

    Returns (psd_fit_linear, exponent). The alpha band is excluded from the
    fit so the oscillation does not pull the background up underneath itself.
    """
    m = ((freqs >= fit_range[0]) & (freqs <= fit_range[1])
         & ~((freqs >= exclude[0]) & (freqs <= exclude[1])))
    if m.sum() < 10:
        return np.full_like(psd, np.nan), np.nan
    lf = np.log10(freqs[m])
    lp = np.log10(np.maximum(psd[m], 1e-20))
    slope, intercept = np.polyfit(lf, lp, 1)
    safe_f = np.maximum(freqs, freqs[1] if len(freqs) > 1 else 1e-6)
    return 10.0 ** (intercept + slope * np.log10(safe_f)), slope


def parabolic_refine(freqs, y, i):
    """Sub-bin peak location by fitting a parabola to the 3 points around i."""
    if i <= 0 or i >= len(y) - 1:
        return freqs[i]
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    denom = y0 - 2 * y1 + y2
    if denom == 0:
        return freqs[i]
    delta = 0.5 * (y0 - y2) / denom
    delta = float(np.clip(delta, -1.0, 1.0))
    return freqs[i] + delta * (freqs[1] - freqs[0])


def estimate_paf(freqs, psd, band=(7.0, 13.0), min_snr_db=6.0,
                 smooth_hz=0.5, halfwidth=1.0):
    """Estimate PAF and alpha amplitude from a single averaged PSD.

    psd : 1-D array, the PSD averaged across the selected channels.
    """
    out = {"paf_peak": np.nan, "paf_cog": np.nan, "amp_uv": np.nan,
           "rms_uv": np.nan, "bandpower": np.nan, "peak_psd": np.nan,
           "snr_db": np.nan, "exponent": np.nan, "status": "NO_PEAK"}

    psd_fit, exponent = fit_aperiodic(freqs, psd)
    out["exponent"] = exponent
    if not np.isfinite(exponent):
        return out, None

    # Oscillatory part: what is left after the 1/f background is removed.
    resid = psd - psd_fit

    df = freqs[1] - freqs[0]
    nsm = max(3, int(round(smooth_hz / df)) | 1)
    if nsm < len(resid):
        resid_s = signal.savgol_filter(resid, nsm, 2)
    else:
        resid_s = resid

    bmask = (freqs >= band[0]) & (freqs <= band[1])
    if bmask.sum() < 5:
        return out, resid_s
    bidx = np.flatnonzero(bmask)
    local = resid_s[bmask]
    j = int(np.argmax(local))
    i = bidx[j]

    # A peak pinned to the edge of the search band is not a peak, it is the
    # band edge clipping a slope that continues outside. Reject it.
    if j == 0 or j == len(local) - 1:
        return out, resid_s

    peak_psd = psd[i]
    bg = psd_fit[i]
    snr_db = 10.0 * np.log10(max(peak_psd, 1e-20) / max(bg, 1e-20))
    out["peak_psd"] = float(peak_psd)
    out["snr_db"] = float(snr_db)

    if snr_db < min_snr_db or local[j] <= 0:
        return out, resid_s

    out["paf_peak"] = float(parabolic_refine(freqs, resid_s, i))

    # Centre of gravity on the 1/f-corrected power. Clipping at zero keeps
    # negative residual (background noise) from dragging the centroid.
    wts = np.clip(local, 0.0, None)
    if wts.sum() > 0:
        out["paf_cog"] = float(np.sum(freqs[bmask] * wts) / wts.sum())

    # Amplitude: integrate the oscillatory power in a window around the peak.
    f0 = out["paf_peak"]
    amask = (freqs >= f0 - halfwidth) & (freqs <= f0 + halfwidth)
    bp = float(np.sum(np.clip(resid[amask], 0.0, None)) * df)
    out["bandpower"] = bp
    out["rms_uv"] = float(np.sqrt(max(bp, 0.0)))
    # For a sinusoid, power = A^2/2, so A = sqrt(2 * power).
    out["amp_uv"] = float(np.sqrt(2.0 * max(bp, 0.0)))
    out["status"] = "OK"
    return out, resid_s


def analyse_window(win, fs, taper, nfft, band, reject_pp, min_good,
                   min_snr_db, halfwidth):
    """Full per-window pipeline: detrend, reject, PSD, average, estimate."""
    x = signal.detrend(win, axis=0, type="linear")

    pp = x.max(axis=0) - x.min(axis=0)
    good = pp <= reject_pp
    n_good = int(good.sum())
    if n_good < min_good:
        return ({"status": "ARTIFACT", "n_good": n_good, "paf_peak": np.nan,
                 "paf_cog": np.nan, "amp_uv": np.nan, "rms_uv": np.nan,
                 "bandpower": np.nan, "peak_psd": np.nan, "snr_db": np.nan,
                 "exponent": np.nan}, None, None)

    freqs, psd = compute_psd(x[:, good], fs, taper=taper, nfft=nfft)
    psd_mean = psd.mean(axis=1)
    res, resid = estimate_paf(freqs, psd_mean, band, min_snr_db,
                              halfwidth=halfwidth)
    res["n_good"] = n_good
    return res, freqs, psd_mean


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

def synth_eeg(n_samples, n_ch, fs, f_alpha, amp, seed=0, exponent=1.5,
              bg_uv=25.0):
    """1/f background plus a narrowband alpha oscillation, per channel."""
    r = np.random.default_rng(seed)
    n = int(next_pow2(n_samples * 2))
    f = np.fft.rfftfreq(n, 1.0 / fs)
    f[0] = f[1]
    out = np.zeros((n_samples, n_ch), dtype=np.float64)
    t = np.arange(n_samples) / fs
    for c in range(n_ch):
        X = (r.normal(size=len(f)) + 1j * r.normal(size=len(f))) * f ** (-exponent / 2)
        bg = np.fft.irfft(X, n=n)[:n_samples]
        bg *= bg_uv / bg.std()
        out[:, c] = bg + amp * np.sin(2 * np.pi * f_alpha * t + r.uniform(0, 2 * np.pi))
    return out


def selftest(fs=500.0, window=2.0, hop=0.25, f_alpha=10.3, amp=12.0,
             n_ch=7, duration=60.0, taper="hann", band=(7.0, 13.0),
             min_snr=6.0):
    print("=" * 68)
    print("SELF-TEST -- synthetic 1/f EEG with a known alpha oscillation")
    print("=" * 68)
    print(f"  truth: f_alpha = {f_alpha} Hz, amplitude = {amp} uV, "
          f"{n_ch} channels, 1/f exponent = -1.5")
    print(f"  window {window} s, hop {hop} s, taper {taper}\n")

    n_total = int(duration * fs)
    data = synth_eeg(n_total, n_ch, fs, f_alpha, amp, seed=7)
    nwin = int(window * fs)
    nhop = int(hop * fs)
    nfft = next_pow2(8 * nwin)

    pk, cg, am, sn, pr = [], [], [], [], []
    for start in range(0, n_total - nwin, nhop):
        win = data[start:start + nwin]
        t0 = time.perf_counter()
        res, _, _ = analyse_window(win, fs, taper, nfft, band,
                                   reject_pp=1e9, min_good=1,
                                   min_snr_db=min_snr, halfwidth=1.0)
        pr.append((time.perf_counter() - t0) * 1000)
        if res["status"] == "OK":
            pk.append(res["paf_peak"]); cg.append(res["paf_cog"])
            am.append(res["amp_uv"]); sn.append(res["snr_db"])

    pk, cg, am, sn, pr = map(np.array, (pk, cg, am, sn, pr))
    n = len(pk)
    print(f"  {n} windows analysed, peak detected in {100*n/max(1,len(pr)):.0f}%\n")
    print(f"{'metric':<16}{'mean':>10}{'sd':>9}{'bias':>10}{'|err|<0.2Hz':>13}")
    print("-" * 58)
    print(f"{'paf_peak (Hz)':<16}{pk.mean():>10.3f}{pk.std():>9.3f}"
          f"{pk.mean()-f_alpha:>+10.3f}{100*np.mean(np.abs(pk-f_alpha)<0.2):>12.0f}%")
    print(f"{'paf_cog  (Hz)':<16}{cg.mean():>10.3f}{cg.std():>9.3f}"
          f"{cg.mean()-f_alpha:>+10.3f}{100*np.mean(np.abs(cg-f_alpha)<0.2):>12.0f}%")
    print(f"{'amp_uv   (uV)':<16}{am.mean():>10.3f}{am.std():>9.3f}"
          f"{am.mean()-amp:>+10.3f}"
          f"{100*np.mean(np.abs(am-amp)/amp<0.1):>12.0f}%  (within 10%)")
    print(f"{'snr_db   (dB)':<16}{sn.mean():>10.3f}{sn.std():>9.3f}")
    print(f"\n  processing time per window: mean {pr.mean():.2f} ms, "
          f"max {pr.max():.2f} ms  (budget = {1000*hop:.0f} ms)")

    # False-positive check: the same pipeline on 1/f noise with NO alpha
    # must stay silent, otherwise a TMS trigger would fire on nothing.
    noise = synth_eeg(n_total, n_ch, fs, f_alpha, 0.0, seed=99)
    nfp = ntot = 0
    for start in range(0, n_total - nwin, nhop):
        r, _, _ = analyse_window(noise[start:start + nwin], fs, taper, nfft,
                                 band, 1e9, 1, min_snr, 1.0)
        ntot += 1
        nfp += (r["status"] == "OK")
    fp_rate = 100.0 * nfp / max(ntot, 1)
    print(f"\n  false positives on alpha-free 1/f noise "
          f"(min_snr = {min_snr} dB): {fp_rate:.0f}%")

    ok = (abs(pk.mean() - f_alpha) < 0.1 and abs(am.mean() - amp) / amp < 0.15
          and pr.mean() < 1000 * hop * 0.5 and fp_rate < 5.0)
    print(f"\n  RESULT: {'PASS' if ok else 'CHECK THE NUMBERS ABOVE'}")
    print("=" * 68)
    return ok


# --------------------------------------------------------------------------
# LSL plumbing
# --------------------------------------------------------------------------

def get_inlet_and_labels(name, stype, timeout=10.0):
    from pylsl import resolve_byprop, StreamInlet, proc_clocksync

    if name:
        print(f"Looking for stream '{name}' ...")
        found = resolve_byprop("name", name, 1, timeout=timeout)
    else:
        print(f"Looking for a stream of type '{stype}' ...")
        found = resolve_byprop("type", stype, 1, timeout=timeout)
    if not found:
        sys.exit("No stream found. Is replay_xdf_lsl.py running?")

    inlet = StreamInlet(found[0], max_buflen=60, max_chunklen=0,
                        processing_flags=proc_clocksync)
    info = inlet.info(timeout=5.0)          # full info, including channel desc
    n_ch = info.channel_count()
    fs = info.nominal_srate()

    labels = []
    try:
        ch = info.desc().child("channels").child("channel")
        for _ in range(n_ch):
            labels.append(ch.child_value("label"))
            ch = ch.next_sibling()
    except Exception:
        pass
    if len(labels) != n_ch or any(not l for l in labels):
        labels = [f"Ch{i+1}" for i in range(n_ch)]

    print(f"Connected to '{info.name()}': {n_ch} ch @ {fs:g} Hz")
    return inlet, fs, labels


def make_paf_outlet(hop, source_id="paf_realtime"):
    from pylsl import StreamInfo, StreamOutlet
    names = ["paf_peak", "paf_cog", "amp_uv", "rms_uv",
             "snr_db", "bandpower", "n_good", "status"]
    info = StreamInfo("PAF", "PAF", len(names), 1.0 / hop, "float32", source_id)
    chans = info.desc().append_child("channels")
    for nm in names:
        c = chans.append_child("channel")
        c.append_child_value("label", nm)
    return StreamOutlet(info), names


def resolve_channels(labels, requested):
    """Case-insensitive channel selection with a posterior default."""
    lut = {l.upper(): i for i, l in enumerate(labels)}
    if requested:
        idx = []
        for c in requested:
            if c.upper() in lut:
                idx.append(lut[c.upper()])
            elif c.isdigit() and int(c) < len(labels):
                idx.append(int(c))
            else:
                sys.exit(f"Channel '{c}' not in stream. Available: {labels}")
        return idx
    idx = [lut[c] for c in DEFAULT_POSTERIOR if c in lut]
    if not idx:
        print("  WARNING: no standard posterior channels found; using all.")
        idx = list(range(len(labels)))
    return idx


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Real-time peak alpha frequency from an LSL EEG stream.")
    p.add_argument("--selftest", action="store_true",
                   help="validate the estimator on synthetic data and exit")

    src = p.add_argument_group("source")
    src.add_argument("--name", default="EEGReplay")
    src.add_argument("--type", dest="stype", default="EEG")
    src.add_argument("--channels", nargs="+", default=None,
                     help="channels to analyse (default: posterior montage)")
    src.add_argument("--ref", default="average",
                     choices=["average", "none"],
                     help="re-referencing across all channels (default average)")

    ana = p.add_argument_group("analysis")
    ana.add_argument("--window", type=float, default=2.0)
    ana.add_argument("--hop", type=float, default=0.25)
    ana.add_argument("--band", type=float, nargs=2, default=[7.0, 13.0],
                     metavar=("LO", "HI"))
    ana.add_argument("--taper", default="hann", choices=["hann", "multitaper"])
    ana.add_argument("--pad", type=int, default=8,
                     help="zero-pad factor for peak localisation (default 8)")
    ana.add_argument("--highpass", type=float, default=1.0,
                     help="streaming high-pass in Hz, 0 to disable")
    ana.add_argument("--reject-pp", type=float, default=150.0,
                     help="reject a channel if peak-to-peak exceeds this (uV)")
    ana.add_argument("--min-good", type=int, default=2,
                     help="minimum clean channels needed to emit an estimate")
    ana.add_argument("--min-snr", type=float, default=6.0,
                     help="minimum peak prominence over the 1/f background, "
                          "in dB. Measured on synthetic data: 6.0 gives zero "
                          "false positives on alpha-free noise while still "
                          "detecting a 3 uV oscillation 100%% of the time. At "
                          "2.0, pure noise reports a peak 73%% of the time.")
    ana.add_argument("--halfwidth", type=float, default=1.0,
                     help="half-width around the peak for amplitude (Hz)")
    ana.add_argument("--median", type=int, default=5,
                     help="running median length for the reported PAF")

    outg = p.add_argument_group("output")
    outg.add_argument("--csv", default=None, help="log every estimate to CSV")
    outg.add_argument("--plot", action="store_true",
                      help="live spectrum + PAF plot (adds jitter; off for "
                           "closed-loop use)")
    outg.add_argument("--no-lsl-out", action="store_true",
                      help="do not publish the PAF stream")
    outg.add_argument("--duration", type=float, default=0.0)

    args = p.parse_args()

    if args.selftest:
        sys.exit(0 if selftest(taper=args.taper, window=args.window,
                               hop=args.hop, band=tuple(args.band),
                               min_snr=args.min_snr) else 1)

    from pylsl import local_clock

    inlet, fs, labels = get_inlet_and_labels(args.name, args.stype)
    n_ch_in = len(labels)
    sel = resolve_channels(labels, args.channels)
    print(f"  analysing {len(sel)} channel(s): "
          f"{', '.join(labels[i] for i in sel)}")
    print(f"  reference: {args.ref}")

    nwin = int(round(args.window * fs))
    nfft = next_pow2(args.pad * nwin)
    df = fs / nfft
    print(f"  window {args.window:g} s = {nwin} samples -> native resolution "
          f"{fs/nwin:.3f} Hz, interpolated to {df:.3f} Hz (nfft={nfft})")
    print(f"  band {args.band[0]}-{args.band[1]} Hz, hop {args.hop:g} s, "
          f"taper {args.taper}\n")

    ring = RingBuffer(nwin, n_ch_in)

    # Streaming high-pass. Filter state persists across chunks so there is no
    # per-window edge transient -- this is the causal equivalent of filtering
    # the whole recording first.
    sos, zi = None, None
    if args.highpass > 0:
        sos = signal.butter(4, args.highpass, btype="highpass", fs=fs,
                            output="sos")
        zi = None  # initialised from the first sample

    outlet = names = None
    if not args.no_lsl_out:
        outlet, names = make_paf_outlet(args.hop)
        print(f"  publishing LSL stream 'PAF' ({len(names)} ch): "
              f"{', '.join(names)}\n")

    csv_f = csv_w = None
    if args.csv:
        csv_f = open(args.csv, "w", newline="")
        csv_w = csv.writer(csv_f)
        csv_w.writerow(["t_lsl", "t_rel", "status", "paf_peak", "paf_cog",
                        "paf_median", "amp_uv", "rms_uv", "snr_db",
                        "bandpower", "exponent", "n_good", "proc_ms"])

    plotter = None
    if args.plot:
        plotter = LivePlot(args.band)

    hist = deque(maxlen=max(1, args.median))
    print(f"{'t(s)':>7} {'status':>9} {'PAF_pk':>8} {'PAF_cog':>8} "
          f"{'med':>7} {'amp_uV':>8} {'SNR_dB':>7} {'1/f':>6} "
          f"{'good':>5} {'ms':>6}")
    print("-" * 82)

    t_start = time.perf_counter()
    next_hop = t_start + args.hop
    newest_ts = None
    n_emit = 0
    proc_times = deque(maxlen=400)

    try:
        while True:
            chunk, stamps = inlet.pull_chunk(timeout=0.02, max_samples=4096)
            if stamps:
                arr = np.asarray(chunk, dtype=np.float64)
                if sos is not None:
                    if zi is None:
                        zi = (signal.sosfilt_zi(sos)[:, :, None]
                              * arr[0][None, None, :])
                    arr, zi = signal.sosfilt(sos, arr, axis=0, zi=zi)
                ring.push(arr.astype(np.float32))
                newest_ts = stamps[-1]

            now = time.perf_counter()
            if now < next_hop:
                continue
            next_hop += args.hop
            if next_hop < time.perf_counter():
                next_hop = time.perf_counter() + args.hop

            elapsed = now - t_start
            if not ring.full:
                print(f"{elapsed:7.1f} {'FILLING':>9}   "
                      f"buffer {100*ring.written/nwin:5.1f}%", end="\r")
                continue

            t0 = time.perf_counter()
            win = ring.get().astype(np.float64)
            if args.ref == "average":
                win = win - win.mean(axis=1, keepdims=True)
            res, freqs, psd_mean = analyse_window(
                win[:, sel], fs, args.taper, nfft, tuple(args.band),
                args.reject_pp, args.min_good, args.min_snr, args.halfwidth)
            proc_ms = (time.perf_counter() - t0) * 1000.0
            proc_times.append(proc_ms)

            if res["status"] == "OK":
                hist.append(res["paf_peak"])
            med = float(np.median(hist)) if hist else np.nan

            n_emit += 1
            if n_emit % max(1, int(round(1.0 / args.hop))) == 0:
                def f(v, w=8, d=3):
                    return f"{v:>{w}.{d}f}" if np.isfinite(v) else f"{'--':>{w}}"
                print(f"{elapsed:7.1f} {res['status']:>9} "
                      f"{f(res['paf_peak'])} {f(res['paf_cog'])} "
                      f"{f(med,7,3)} {f(res['amp_uv'],8,2)} "
                      f"{f(res['snr_db'],7,2)} {f(res['exponent'],6,2)} "
                      f"{res['n_good']:>5} {proc_ms:>6.1f}")

            if outlet is not None:
                vec = [res["paf_peak"], res["paf_cog"], res["amp_uv"],
                       res["rms_uv"], res["snr_db"], res["bandpower"],
                       float(res["n_good"]), STATUS_CODES[res["status"]]]
                vec = [0.0 if not np.isfinite(v) else float(v) for v in vec]
                # Stamp with the time of the newest sample in the window, so a
                # downstream trigger knows which data this refers to.
                outlet.push_sample(vec, newest_ts or local_clock())

            if csv_w is not None:
                csv_w.writerow([newest_ts, round(elapsed, 4), res["status"],
                                res["paf_peak"], res["paf_cog"], med,
                                res["amp_uv"], res["rms_uv"], res["snr_db"],
                                res["bandpower"], res["exponent"],
                                res["n_good"], round(proc_ms, 3)])

            if plotter is not None and freqs is not None:
                plotter.update(freqs, psd_mean, res, elapsed)

            if args.duration and elapsed > args.duration:
                break

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        if csv_f:
            csv_f.close()
            print(f"Wrote {args.csv}")
        if proc_times:
            pt = np.array(proc_times)
            print(f"Processing: mean {pt.mean():.2f} ms, max {pt.max():.2f} ms "
                  f"(budget {1000*args.hop:.0f} ms)")


# --------------------------------------------------------------------------

class LivePlot:
    """Spectrum with the detected peak, plus PAF over time."""

    def __init__(self, band, history=240):
        import matplotlib
        import matplotlib.pyplot as plt
        self.plt = plt
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(9, 7))
        self.band = band
        self.t_hist = deque(maxlen=history)
        self.p_hist = deque(maxlen=history)
        self.a_hist = deque(maxlen=history)

        self.ln_psd, = self.ax1.semilogy([], [], lw=1.2, label="PSD")
        self.ln_fit, = self.ax1.semilogy([], [], lw=1.0, ls="--", label="1/f fit")
        self.vline = self.ax1.axvline(np.nan, color="r", lw=1.2, label="PAF")
        self.ax1.axvspan(band[0], band[1], alpha=0.12, color="orange")
        self.ax1.set_xlim(1, 45)
        self.ax1.set_xlabel("Frequency (Hz)")
        self.ax1.set_ylabel("PSD (uV$^2$/Hz)")
        self.ax1.legend(loc="upper right", fontsize=8)
        self.ax1.grid(alpha=0.3)

        self.ln_paf, = self.ax2.plot([], [], ".-", ms=3, lw=0.8)
        self.ax2.set_ylim(band[0], band[1])
        self.ax2.set_xlabel("Time (s)")
        self.ax2.set_ylabel("PAF (Hz)")
        self.ax2.grid(alpha=0.3)
        self.ax2b = self.ax2.twinx()
        self.ln_amp, = self.ax2b.plot([], [], ".-", ms=3, lw=0.8,
                                      color="green", alpha=0.6)
        self.ax2b.set_ylabel("Amplitude (uV)", color="green")
        self.fig.tight_layout()
        self._i = 0

    def update(self, freqs, psd, res, t):
        self._i += 1
        if self._i % 2:          # redraw at half the hop rate
            return
        m = (freqs >= 1) & (freqs <= 45)
        self.ln_psd.set_data(freqs[m], psd[m])
        fit, _ = fit_aperiodic(freqs, psd)
        self.ln_fit.set_data(freqs[m], fit[m])
        self.ax1.set_ylim(max(psd[m].min() * 0.5, 1e-4), psd[m].max() * 3)
        if np.isfinite(res["paf_peak"]):
            self.vline.set_xdata([res["paf_peak"]] * 2)
            self.ax1.set_title(
                f"PAF {res['paf_peak']:.2f} Hz   "
                f"amp {res['amp_uv']:.1f} uV   SNR {res['snr_db']:.1f} dB")
            self.t_hist.append(t)
            self.p_hist.append(res["paf_peak"])
            self.a_hist.append(res["amp_uv"])
        else:
            self.ax1.set_title(f"no alpha peak ({res['status']})")
        if self.t_hist:
            self.ln_paf.set_data(self.t_hist, self.p_hist)
            self.ln_amp.set_data(self.t_hist, self.a_hist)
            self.ax2.set_xlim(self.t_hist[0], max(self.t_hist[-1], 1))
            self.ax2b.set_ylim(0, max(self.a_hist) * 1.3 + 1e-6)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self.plt.pause(0.001)


if __name__ == "__main__":
    main()
