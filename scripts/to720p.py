"""Downscale a clip to 720p (1280x720) with audio preserved, via PyAV (libx264 + aac).

Corpus-prep utility: the capture hardware records 720p, but our test footage is 4K. This makes
a native-720p file so the pipeline is exercised at the real target resolution (FS1: >= 720p).
Video is spatially rescaled and re-encoded; audio is re-encoded (resampled to the encoder's
format) so A/V timing is preserved and measurable (FS2). Source frame rate is kept.

    python to720p.py IN.mp4 OUT.mp4 [--seconds N]   # --seconds trims for a quick smoke test
"""
import argparse
import av
from fractions import Fraction

LONG_EDGE = 1280  # 720p landscape -> 1280x720


def transcode(src, dst, seconds=None):
    inp = av.open(src)
    ivs = inp.streams.video[0]
    ias = inp.streams.audio[0] if inp.streams.audio else None

    w, h = ivs.codec_context.width, ivs.codec_context.height
    scale = LONG_EDGE / max(w, h)
    ow, oh = (int(round(w * scale)) // 2) * 2, (int(round(h * scale)) // 2) * 2  # even dims for yuv420p

    out = av.open(dst, "w")
    rate = ivs.average_rate or Fraction(30, 1)
    ovs = out.add_stream("libx264", rate=rate)
    ovs.width, ovs.height = ow, oh
    ovs.pix_fmt = "yuv420p"
    ovs.options = {"crf": "23", "preset": "veryfast"}

    oas = resampler = None
    if ias is not None:
        sr = ias.codec_context.sample_rate
        oas = out.add_stream("aac", rate=sr)
        oas.options = {"b": "128k"}
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=sr)

    def within(frame, stream):
        if seconds is None or frame.pts is None:
            return True
        return float(frame.pts * stream.time_base) <= seconds

    stop_v = stop_a = False
    for packet in inp.demux(*( [ivs] + ([ias] if ias else []) )):
        if packet.dts is None:
            continue
        for frame in packet.decode():
            if isinstance(frame, av.VideoFrame):
                if not within(frame, ivs):
                    stop_v = True
                    continue
                nf = frame.reformat(width=ow, height=oh, format="yuv420p")
                for p in ovs.encode(nf):
                    out.mux(p)
            elif oas is not None and isinstance(frame, av.AudioFrame):
                if not within(frame, ias):
                    stop_a = True
                    continue
                for rf in resampler.resample(frame):
                    for p in oas.encode(rf):
                        out.mux(p)
        if seconds is not None and stop_v and (ias is None or stop_a):
            break

    for p in ovs.encode():
        out.mux(p)
    if oas is not None:
        for p in oas.encode():
            out.mux(p)
    out.close()
    inp.close()
    print(f"wrote {dst}  {ow}x{oh}  rate={float(rate):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--seconds", type=float, default=None)
    a = ap.parse_args()
    transcode(a.src, a.dst, a.seconds)
