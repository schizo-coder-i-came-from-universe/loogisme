import pyaudio
import numpy as np

# Parameters
format = pyaudio.paInt16  # Audio format
kanal = 1  # Mono audio
RATE = 48000  # Sampling rate
frames_za_bufer = 2048  # Frames per buffer
sekundy = 2  # Duration to record

# Initialize PyAudio
audio = pyaudio.PyAudio()
CONCERT_PITCH = 440
Vsechny_Noty = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
Relevanti_Noty = ["E3", "A2", "D3", "G3", "B3", "E4"]

#def Nejblizsi_Nota(pitch):
 # i = int(np.round(np.log2(pitch/CONCERT_PITCH)*12))
 # closest_note = Vsechny_Noty[i % 12] + str(4 + (i + 9) // 12)
  #closest_pitch = round(CONCERT_PITCH*2**(i/12),2)
  #return closest_note, closest_pitch

# Open stream
stream = audio.open(format=format, channels=kanal,
                    rate=RATE, input=True,
                    frames_per_buffer=frames_za_bufer)
def Nejblizsi_Relevanti(pitch):
  i = int(np.round(np.log2(pitch/CONCERT_PITCH)*12))
  closest_note = Vsechny_Noty[i % 12] + str(4 + (i + 9) // 12)
  closest_pitch = round(CONCERT_PITCH*2**(i/12),2)
  if closest_note in Relevanti_Noty:
    return closest_note, closest_pitch
  elif closest_pitch <= 96.205:
        return "E3", 82.41
  elif 96.20 < closest_pitch <= 128.41:
        return "A2", 110.00
  elif 128.41 < closest_pitch <= 171.41:
        return "D3", 146.83
  elif 171.41 < closest_pitch <= 221.47:
        return "G3", 196.00
  elif 221.47 < closest_pitch <= 288.29:
        return "B3", 246.94
  else:
        return "E4", 329.63




def Lazeni(pf,nejbliz):
    if abs(pf - nejbliz) < 0.11:
        return "In tune"
    elif pf < nejbliz:
        return f"Too low by {nejbliz - pf} Hz"
    else:
        return f"Too high by {pf - nejbliz} Hz"
def harmonic_product_spectrum(fft_values, downsample_factor=5):
    hps = np.copy(fft_values)
    for h in range(2, downsample_factor + 1):
        decimated = np.copy(fft_values[::h])
        hps[:len(decimated)] *= decimated
    return hps

print("Listening...")

try:
    while True:
        # Read data
        frames = []
        for _ in range(0, int(RATE / frames_za_bufer * sekundy)):
            data = stream.read(frames_za_bufer)
            frames.append(np.frombuffer(data, dtype=np.int16))

        # Convert frames to numpy array
        frames = np.hstack(frames)

        # Apply window function
        window = np.hanning(len(frames))
        frames = frames * window

        # Zero-padding
        frames = np.pad(frames, (0, len(frames)), 'constant')

        # Perform FFT
        frequencies = np.fft.fftfreq(len(frames), 1.0 / RATE)
        fft_values = np.fft.fft(frames)

        hps = harmonic_product_spectrum(fft_values)
        peak_freq = round(abs(frequencies[np.argmax(hps)]), 3)

        print(f"Peak frequency: {peak_freq} Hz")
        nejbliz = Nejblizsi_Relevanti(peak_freq)
        print(f"Closest note: {nejbliz}")
        print(Lazeni(peak_freq,nejbliz[1]))
except KeyboardInterrupt:
    print("Stopped listening.")

finally:
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

