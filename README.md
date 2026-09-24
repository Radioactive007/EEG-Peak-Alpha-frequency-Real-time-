1- For testing in real time using existing data, use this
"
Terminal 1:  python replay_xdf_lsl.py P03.xdf --stream-index 2 --loop
Terminal 2:  python paf_realtime.py --name EEGReplay --plot             %  Do this first for checking using dummy data ( python paf_realtime.py --selftest )
"
  In Terminal 1, P03.xdf is file name, which should be in the same folder, from where you open the terminal (it works better, if all the files are in the same folder)
  Secondly, I have taken stream Index as 2, because in xdf, index 2 is my EEG stream  # Terminal 1 — see what's in the file, then stream it
  For checking your stream name -( python replay_xdf_lsl.py P03.xdf --list ) You can write your file name here, and it will list the streams, based on that you can choose the stream id

  Prerequisites - ( pip install pylsl pyxdf numpy )
                  ( pip install scipy matplotlib  )

 
** For live streaming real data**
 Use this just to see the number of streams in
 "
 Terminal 1: python list_lsl_streams.py             % then
             python paf_realtime.py --name menteve_usb_hid_001_eeg --plot      % in the same terminal
 
  For continuously saving the data final script:
   python paf_realtime.py --name menteve_usb_hid_001_eeg --csv logs\sub-P03_paf.csv --plot     % This will also save the data

 ** Final Code**
 
python src\paf_realtime.py --name menteve_usb_hid_001_eeg --csv logs\Sub-P03_paf.csv --plot   %(same as last code, just the directory is different)
