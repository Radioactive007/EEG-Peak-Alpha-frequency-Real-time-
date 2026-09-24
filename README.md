# EEG Peak Alpha Frequency Pipeline

Real-time estimation of peak alpha frequency (PAF) and alpha amplitude from an
LSL EEG stream. Built so that the same analysis code runs unchanged against
either a recorded file or a live amplifier.

---

For testing in real time using existing data, use this
"
Terminal 1:  **python replay_xdf_lsl.py P03.xdf --stream-index 2 --loop**
Terminal 2:**  python paf_realtime.py --name EEGReplay --plot    **         %  Do this first for checking using dummy data ( python paf_realtime.py --selftest )
"
  In Terminal 1, P03.xdf is file name, which should be in the same folder, from where you open the terminal (it works better, if all the files are in the same folder)
  Secondly, I have taken stream Index as 2, because in xdf, index 2 is my EEG stream  # Terminal 1 — see what's in the file, then stream it
  For checking your stream name -( python replay_xdf_lsl.py P03.xdf --list ) You can write your file name here, and it will list the streams, based on that you can choose the stream id

  Prerequisites - ( pip install pylsl pyxdf numpy )
                  ( pip install scipy matplotlib  )

 For live streaming real data
 Use this just to see the number of streams in
 "
 Terminal 1:** python list_lsl_streams.py **            % then
           **  python paf_realtime.py --name menteve_usb_hid_001_eeg --plot  **    % in the same terminal
 
  For continuously saving the data final script:
** python paf_realtime.py --name menteve_usb_hid_001_eeg --csv logs\sub-P03_paf.csv --plot **     % This will also save the data


Final Code
  ** python src\paf_realtime.py --name menteve_usb_hid_001_eeg --csv logs\Sub-P03_paf.csv --plot **     %(same as last code, just the directory is different)


















## Folder layout

```
eeg-paf/
├── README.md                  this file
├── requirements.txt
│
├── src/
│   ├── replay_xdf_lsl.py      recorded .xdf  ->  live LSL stream
│   ├── paf_realtime.py        LSL stream     ->  PAF + amplitude
│   └── list_lsl_streams.py    show what is on the network
│
├── data/
│   ├── raw/                   original recordings   (P03.xdf ...)
│   └── live/                  new recordings from LabRecorder
│
├── logs/                      CSV output from paf_realtime.py
│
└── archive/
    └── check_stream.py        one-time timing diagnostic, kept for reference
```

Create it in PowerShell:

```powershell
cd C:\Users\Pranj\Desktop
mkdir eeg-paf\src, eeg-paf\data\raw, eeg-paf\data\live, eeg-paf\logs, eeg-paf\archive
```

Then move the scripts into `src\`, your `.xdf` files into `data\raw\`, and
`check_stream.py` into `archive\`.

Always run commands from the project root (`eeg-paf`), not from inside `src`.
That keeps every path in this file correct.

---

## Naming conventions

Pick these now; renaming later is painful.

| Thing | Pattern | Example |
|---|---|---|
| Raw recording | `sub-<ID>_ses-<N>_task-<name>_eeg.xdf` | `sub-P03_ses-01_task-rest_eeg.xdf` |
| PAF log | `sub-<ID>_ses-<N>_task-<name>_paf.csv` | `sub-P03_ses-01_task-rest_paf.csv` |
| LSL stream (replay) | `EEGReplay` | fixed, so commands never change |
| LSL stream (output) | `PAF` | fixed |

Keep the subject ID identical across the `.xdf` and its `.csv` so they pair up
automatically later.

---

## Setup (once)

```powershell
pip install -r requirements.txt
python src\paf_realtime.py --selftest
```

The self-test must print **PASS** before you trust any real number. It checks
the estimator against synthetic data with a known alpha frequency, and checks
that alpha-free noise does *not* produce a false peak.

---

## Mode A — replay a recorded file (development)

Two terminals, both opened at the project root.

**Terminal 1 — the data source.** Leave running; this is your fake amplifier.

```powershell
python src\replay_xdf_lsl.py data\raw\sub-P03_ses-01_task-rest_eeg.xdf --stream-index 2 --loop
```

**Terminal 2 — the analysis.**

```powershell
python src\paf_realtime.py --name EEGReplay --plot
```

With CSV logging instead of plotting:

```powershell
python src\paf_realtime.py --name EEGReplay --csv logs\sub-P03_ses-01_task-rest_paf.csv --no-lsl-out
```

---

## Mode B — live amplifier (real experiment)

`replay_xdf_lsl.py` is **not used**. The amplifier replaces it.

**Terminal 1 — the amplifier's own LSL application.** Start it however your
device requires. It publishes the EEG stream.

**Terminal 2 — find the exact stream name.** Names are case-sensitive.

```powershell
python src\list_lsl_streams.py
```

**Terminal 3 — the analysis.** Substitute the name you just read.

```powershell
python src\paf_realtime.py --name menteve_usb_hid_001_eeg --plot
```

**Terminal 4 — LabRecorder (recommended).** Record *both* the raw EEG stream
and the `PAF` stream into one `.xdf`. They share the LSL clock, so the PAF
values stay aligned to the raw data and you can re-check any estimate offline.
Save to `data\live\`.

That is the only difference between Mode A and Mode B: one flag, `--name`.

---

## Validating live data

Before believing anything, run an **eyes-open / eyes-closed** block:

- 30 s eyes open, 30 s eyes closed, repeated 3 times
- Alpha amplitude should roughly double with eyes closed
- PAF should stay within about 0.5 Hz across both

If alpha does not rise with eyes closed, the problem is electrodes,
referencing, or channel selection, not the analysis.

---

## Parameters worth knowing

| Flag | Default | Notes |
|---|---|---|
| `--channels` | posterior auto | O1 O2 Po3 Po4 P3 p4 Pz on this montage |
| `--window` | 2.0 s | longer = better resolution, slower response |
| `--hop` | 0.25 s | how often an estimate is produced |
| `--band` | 7–13 Hz | alpha search range |
| `--taper` | hann | `multitaper` smooths a split alpha peak |
| `--min-snr` | 6.0 dB | validated: 0% false positives, 100% detection at 3 uV |
| `--reject-pp` | 150 uV | raise if the `good` column sits low |
| `--median` | 5 | running median for the console `med` column only |

`paf_peak` is the tallest point in the alpha bump. `paf_cog` is its centre of
gravity — more stable when alpha has two sub-peaks, slightly biased low.
Choose one before looking at results and keep it.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `No such file or directory` | Wrong folder, or the file saved as `.py.txt` |
| `No stream found` | Source not running, or wrong `--name` |
| Constant `ARTIFACT` | `--reject-pp` too low for this data |
| Constant `NO_PEAK` | No real alpha, or wrong channels, or bad electrodes |
| PAF jumping 7–12 Hz | Split alpha peak; try `--taper multitaper` or use `paf_cog` |
| Lag climbing steadily | Consumer slower than real time; raise `--hop` |
This script
