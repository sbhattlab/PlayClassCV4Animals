# Frame loading -- IMPORTANT

**H.264 frame seeking is broken.** `cv2.CAP_PROP_POS_FRAMES` seeking returns wrong frames for H.264-encoded videos (only accurate at keyframes/I-frames; inter-coded P/B-frames return incorrect data).

**Rule: Always use `load_video_frames_sequential()` from `src/io.py` for frame-accurate loading.** Never use seek-based frame loading (`load_video_frames_range`) unless explicitly requested by the user. When writing new code that loads video frames, default to sequential loading.

Two frame-loading functions exist in `src/io.py`:

- `load_video_frames_range` -- Uses `CAP_PROP_POS_FRAMES` seeking. **Fast but unreliable for H.264.**
- `load_video_frames_sequential` -- Reads sequentially from frame 0. **Slower but frame-accurate** for all codecs.

**Known impact**: The SAM3 tracking pipeline (`run_tracker.py`) uses `load_video_frames_range` to load each chunk's frames, so chunks after chunk 0 may start from slightly wrong frames due to seek inaccuracy. Tracking results appear fine in practice (SAM3 is robust to small frame offsets), but this should be fixed.
