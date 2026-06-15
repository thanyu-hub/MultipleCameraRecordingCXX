# MultiCam Capture GUI

## build

```bash
cmake -S . -B build
cmake --build build -j
```

## run

```bash
./build/multicam_capture_gui --output June28-Ponca --row 13
```

For windowed mode:

```bash
./build/multicam_capture_gui --output June28-Ponca --row 13 --windowed
```

Qt5 Widgets and OpenCV are required.

On Ubuntu-like systems, install the C++ development packages first:

```bash
sudo apt install cmake g++ qtbase5-dev libopencv-dev
```

Keyboard controls: `Enter`/`Space` capture, `R` reconnect, `F` fullscreen,
`Q`/`Esc` exit. Full-resolution PNGs and `session.log` are written under
`~/record_data/captures/YYYYMMDD_HHMMSS_<output>-row<row>/`, for example
`~/record_data/captures/20260608_150337_June28-Ponca-row13/`. File names stay
camera-oriented, such as `image_000000_cam0_192.168.0.20.png`. If keyboard
capture is inactive, the window shows `CLICK WINDOW TO ENABLE KEYBOARD CAPTURE`.

The GUI config file is `~/.config/multicam_capture/config.json`. If it does
not exist, the app creates a default config for cameras `192.168.0.20` through
`192.168.0.23`. If this file exists, verify the RTSP URLs before recording.
The output label and row must be entered on every launch with `-o`/`--output`
and `--row`.
Bad or stale camera streams are shown in red and are not saved as successes.
Set a camera `url` to an empty string to disable that camera slot. Disabled
slots are skipped and do not count as failed captures.
The lab default RTSP credentials are built in and are applied to the default
camera URLs without being stored in the generated config. Override them with
`MULTICAM_RTSP_USER` and `MULTICAM_RTSP_PASSWORD` when needed. Credentials are
redacted from logs.
