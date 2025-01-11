import pyaudio
import numpy as np
import pygame
import sys
import threading
from scipy.signal import butter, lfilter

class GuitarTuner:
    def __init__(self):
        self.kytara = pygame.image.load("images/guitar.png")
        self.pozadi = pygame.image.load("images/pozadi.jpg")
        self.pozadi2 = pygame.image.load("images/pozadi2.jpg")

        # Initialize Pygame
        pygame.init()
        self.clock = pygame.time.Clock()
        self.FPS = 60
        self.WIDTH = 1400
        self.HEIGHT = 900
        self.window = pygame.display.set_mode((self.WIDTH, self.HEIGHT))

        # Parameters for audio
        self.format = pyaudio.paInt16
        self.kanal = 1
        self.RATE = 23000
        self.frames_za_buffer = 4000
        self.sekundy = 2
        self.stop_thread = False
        self.amplitude_threshold = 500
        self.moving_average_window = 3
        self.freq_history = []

        # Parameters for the program
        self.end_it = 1
        self.font = pygame.font.Font("freesansbold.ttf", 32)
        self.font_1 = pygame.font.Font("freesansbold.ttf", 50)
        self.font_nota = pygame.font.Font("freesansbold.ttf", 170)
        self.font_akce = pygame.font.Font("freesansbold.ttf", 35)
        self.my_color = (11, 158, 227)

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        self.CONCERT_PITCH = 440

        self.Vsechny_Noty = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
        self.Relevanti_Noty = ["E3", "A2", "D3", "G3", "B3", "E4"]

        # Open stream
        self.stream = self.audio.open(format=self.format, channels=self.kanal,
                                      rate=self.RATE, input=True,
                                      frames_per_buffer=self.frames_za_buffer)

        self.peak_freq = 0
        self.nejbliz = ("", 0)
        self.akce = 0
        self.rozdil = ""

    def Nejblizsi_Relevanti(self, pitch):
        if pitch <= 0 or not np.isfinite(pitch):
            return "", 0.0
        i = int(np.round(np.log2(pitch / self.CONCERT_PITCH) * 12))
        closest_note = self.Vsechny_Noty[i % 12] + str(4 + (i + 9) // 12)
        closest_pitch = round(self.CONCERT_PITCH * 2 ** (i / 12), 2)
        if closest_note in self.Relevanti_Noty:
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

    def Lazeni(self, pf, nejbliz):
        if abs(pf - nejbliz) < 0.5:
            self.akce = 0
            return "In tune"
        elif pf < nejbliz:
            self.akce = 1
            return f"Too low by {round(nejbliz - pf, 3)} Hz"
        else:
            self.akce = 2
            return f"Too high by {round(pf - nejbliz, 3)} Hz"

    def akce_lazeni(self):
        if self.akce == 0:
            return "In tune"
        elif self.akce == 1:
            return "Utahnout strunu"
        else:
            return "Povolit strunu"

    def harmonic_product_spectrum(self, fft_values, downsample_factor=5):
        hps = np.copy(fft_values)
        for h in range(2, downsample_factor + 1):
            decimated = np.copy(fft_values[::h])
            decimated /= np.max(decimated) if np.max(decimated) != 0 else 1
            hps[:len(decimated)] *= decimated
        return hps

    def bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, data)

    def mozek(self):
        amplitude_history = []  # Rolling amplitude history
        stable_frequencies = []  # Stable frequencies
        stable_frames = 5

        while not self.stop_thread:
            data = self.stream.read(self.frames_za_buffer, exception_on_overflow=False)
            frames = np.frombuffer(data, dtype=np.int16)

            # Compute amplitude and check against threshold
            amplitude = np.max(np.abs(frames))
            amplitude_history.append(amplitude)
            if len(amplitude_history) > stable_frames:
                amplitude_history.pop(0)

            if np.mean(amplitude_history) > self.amplitude_threshold:
                frames = self.bandpass_filter(frames, 80, 1200, self.RATE)  # Apply band-pass filter
                window = np.hanning(len(frames))
                frames = frames * window
                frames = np.pad(frames, (0, len(frames)), 'constant')

                frequencies = np.fft.fftfreq(len(frames), 1.0 / self.RATE)
                fft_values = np.fft.fft(frames)

                hps = self.harmonic_product_spectrum(fft_values)
                new_peak_freq = round(abs(frequencies[np.argmax(hps)]), 3)

                # Apply moving average
                self.freq_history.append(new_peak_freq)
                if len(self.freq_history) > self.moving_average_window:
                    self.freq_history.pop(0)
                self.peak_freq = round(np.mean(self.freq_history), 3)

                # Immediate update if significant change
                if abs(new_peak_freq - self.peak_freq) > 5:  # Adjust threshold as needed
                    self.peak_freq = new_peak_freq

                print(f"Peak frequency: {self.peak_freq} Hz")
                self.nejbliz = self.Nejblizsi_Relevanti(self.peak_freq)
                print(f"Closest note: {self.nejbliz}")
                self.rozdil = self.Lazeni(self.peak_freq, self.nejbliz[1])
                print(self.rozdil)

        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

    def run(self):
        mozek_thread = threading.Thread(target=self.mozek)
        mozek_thread.start()

        ProgramBezi = True
        while ProgramBezi:
            while self.end_it == 1:
                areaStart = pygame.Rect(490, 400, 400, 150)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        ProgramBezi = False
                        self.stop_thread = True
                        mozek_thread.join()
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.MOUSEBUTTONDOWN and pygame.mouse.get_pressed()[0] is True:
                        if areaStart.collidepoint(pygame.mouse.get_pos()):
                            self.end_it = 0
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                        self.end_it = 0
                self.window.fill("white")
                self.window.blit(self.pozadi, (0, 0))
                text = self.font_1.render("Click here or", True, self.my_color)
                text2 = self.font_1.render("press Enter", True, self.my_color)
                text3 = self.font_1.render("to start", True, self.my_color)
                self.window.blit(text, (self.WIDTH / 2 - 160, self.HEIGHT / 2 - 50))
                self.window.blit(text2, (self.WIDTH / 2 - 140, self.HEIGHT / 2))
                self.window.blit(text3, (self.WIDTH / 2 - 100, self.HEIGHT / 2 + 50))
                self.window.blit(self.kytara, (self.WIDTH / 2 + 120, self.HEIGHT / 2 - 100))
                self.window.blit(pygame.transform.flip(self.kytara, True, False), (self.WIDTH / 2 - 500, self.HEIGHT / 2 - 100))
                pygame.display.flip()
                self.clock.tick(self.FPS)

            while self.end_it == 0:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        ProgramBezi = False
                        self.stop_thread = True
                        mozek_thread.join()
                        pygame.quit()
                        sys.exit()
                self.window.fill("black")
                self.window.blit(self.pozadi2, (0, 0))
                # Texts to display
                text = self.font.render(f"Peak freq: {self.peak_freq} Hz", True, "white")
                text2 = self.font.render(f"Lazeni: {self.Lazeni(self.peak_freq, self.nejbliz[1])}", True, "white")
                nejblizsitext = self.font_nota.render(f"{self.nejbliz[0]}", True, "black")
                akce_text = self.font_akce.render(f"{self.akce_lazeni()}", True, "black")
                rozdil_text = self.font_akce.render(self.rozdil, True, "black")
                # Display texts
                self.window.blit(nejblizsitext, (110, 120))
                self.window.blit(akce_text, (1020, 180))
                self.window.blit(text, (self.WIDTH / 2, self.HEIGHT / 2))
                self.window.blit(text2, (self.WIDTH / 2, self.HEIGHT / 2 + 100))
                self.window.blit(rozdil_text, (self.WIDTH / 2, self.HEIGHT / 2 + 200))

                pygame.display.flip()
                self.clock.tick(self.FPS)

if __name__ == "__main__":
    tuner = GuitarTuner()
    tuner.run()