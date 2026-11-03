"""Optional local RMS meter. Never stores audio; not a scream classifier."""
import time
import numpy as np

class AudioMeter:
    def __init__(self):
        self.stream = None
        self.db = -100.0
        self.updated = 0.0

    def start(self):
        import sounddevice as sd
        self.stream = sd.InputStream(channels=1, callback=self._receive)
        try:
            self.stream.start()
        except Exception:
            self.stream.close()
            self.stream = None
            raise

    def _receive(self, data, frames, timing, status):
        if status:
            self.updated = 0.0
            return
        self.db = float(20*np.log10(max(float(np.sqrt(np.mean(data**2))), 1e-5)))
        self.updated = time.monotonic()

    def loud(self):
        return self.stream is not None and time.monotonic()-self.updated < .5 and self.db > -18

    def close(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
