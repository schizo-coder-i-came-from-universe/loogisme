import pyaudio
import numpy as np
import pygame
import sys
import threading
from scipy.signal import butter, lfilter

class GuitarTuner:
    def __init__(self):
        #obrazky
        self.kytara = pygame.image.load("images/guitar.png")
        self.pozadi = pygame.image.load("images/pozadi.jpg")
        self.pozadi2 = pygame.image.load("images/pozadi2.jpg")
        self.sipka = pygame.image.load("images/arrow.png")
        self.sipka = pygame.transform.scale(self.sipka, (90, 90))
        self.sipka_reversed = pygame.transform.flip(self.sipka, True, False)

        #zvuky
        pygame.mixer.init()
        self.E2_sound = pygame.mixer.Sound("sound/E2.ogg")
        self.A2_sound = pygame.mixer.Sound("sound/A2.ogg")
        self.D3_sound = pygame.mixer.Sound("sound/D3.ogg")
        self.G3_sound = pygame.mixer.Sound("sound/G3.ogg")
        self.B3_sound = pygame.mixer.Sound("sound/B3.ogg")
        self.E4_sound = pygame.mixer.Sound("sound/E4.ogg")

        # Initialize Pygame
        pygame.init()
        self.clock = pygame.time.Clock()
        self.FPS = 60
        self.WIDTH = 1400
        self.HEIGHT = 900
        self.window = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.end_it = 1

        # Parameters for audio
        self.format = pyaudio.paInt16
        self.kanal = 1
        self.RATE = 23000
        self.frames_za_buffer = 4000
        self.sekundy = 2
        self.stop_thread = False
        self.mozek_thread = None
        self.stream = None
        self.amplitude_threshold = 500
        self.moving_average_window = 3
        self.freq_history = []

        # Fonty juchu
        self.font = pygame.font.Font("freesansbold.ttf", 32)
        self.font_1 = pygame.font.Font("freesansbold.ttf", 50)
        self.font_nota = pygame.font.Font("freesansbold.ttf", 170)
        self.font_akce = pygame.font.Font("freesansbold.ttf", 35)
        self.my_color = (11, 158, 227)

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        self.CONCERT_PITCH = 440

        self.Vsechny_Noty = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
        self.Relevanti_Noty = ["E2", "A2", "D3", "G3", "B3", "E4"]

        # Open stream
        self.stream = self.audio.open(format=self.format, channels=self.kanal,
                                      rate=self.RATE, input=True,
                                      frames_per_buffer=self.frames_za_buffer)
        #promenne co se dale pouzivaji
        self.peak_freq = 0
        self.nejbliz = ("", 0)
        self.akce = 0
        self.rozdil = 0

        #Stoping el programmo
        self.button_rect = pygame.Rect(50, 800, 200, 50)
        self.tuning_active = True



        #Frequency of strings
        self.E2 = 82.41
        self.A2 = 110.00
        self.D3 = 146.83
        self.G3 = 196.00
        self.B3 = 246.94
        self.E4 = 329.63

        #dictionary of notes
        self.tuning_status = {
            "E2": False,
            "A2": False,
            "D3": False,
            "G3": False,
            "B3": False,
            "E4": False
        }
    def Calculate_middle(self, a, b):
        return (a + b) / 2

    def button_click(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.button_rect.collidepoint(event.pos):
                self.tuning_active = not self.tuning_active
                if self.tuning_active:  # Resume
                    self.stop_thread = False
                    if self.stream is None or not self.stream.is_active():
                        try:
                            self.stream = self.audio.open(format=self.format, channels=self.kanal,
                                                          rate=self.RATE, input=True,
                                                          frames_per_buffer=self.frames_za_buffer)
                            self.mozek_thread = threading.Thread(target=self.mozek)
                            self.mozek_thread.start()
                        except (OSError, ValueError) as e:
                            print(f"Error opening stream: {e}")
                else:  # Stop
                    self.stop_thread = True
                    if self.mozek_thread is not None:
                        self.mozek_thread.join()
                        self.mozek_thread = None
                    if self.stream is not None:
                        self.stream.stop_stream()
                        self.stream.close()
                        self.stream = None

    def Nejblizsi_Relevanti(self, pitch):
        if pitch <= 0 or not np.isfinite(pitch):
            return "", 0.0
        i = int(np.round(np.log2(pitch / self.CONCERT_PITCH) * 12))
        closest_note = self.Vsechny_Noty[i % 12] + str(4 + (i + 9) // 12)
        closest_pitch = round(self.CONCERT_PITCH * 2 ** (i / 12), 2)
        if closest_pitch <= self.E2:
            return "E2", 82.41
        elif self.Calculate_middle(self.E2,self.A2) < closest_pitch <= self.Calculate_middle(self.A2,self.D3):
            return "A2", 110.00
        elif self.Calculate_middle(self.A2,self.D3) < closest_pitch <= self.Calculate_middle(self.D3,self.G3):
            return "D3", 146.83
        elif self.Calculate_middle(self.D3,self.G3)< closest_pitch <= self.Calculate_middle(self.G3,self.B3):
            return "G3", 196.00
        elif self.Calculate_middle(self.G3,self.B3) < closest_pitch <= self.Calculate_middle(self.B3,self.E4):
            return "B3", 246.94
        else:
            return "E4", 329.63

    def Lazeni(self, pf, nejbliz):
        if abs(pf - nejbliz) < 0.5:
            self.akce = 0
            self.tuning_status[self.nejbliz[0]] = True
            return "In tune"
        elif pf < nejbliz:
            self.akce = 1
            self.rozdil = (round(nejbliz - pf,4))
            return f"Too low by {self.rozdil} Hz"
        else:
            self.akce = 2
            self.rozdil = (round(pf - nejbliz,4))
            return f"Too high by {self.rozdil} Hz"

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

    def display_tuning_status(self):
        y_offset = 440
        for string in self.Relevanti_Noty:
            status = self.tuning_status.get(string, False)
            status_text = "In tune" if status else "Out of tune"
            text_surface = self.font.render(f"{string}: {status_text}", True, "white")
            self.window.blit(text_surface, (50, y_offset))
            y_offset += 40



    def mozek(self):
        amplitude_history = []  # Rolling amplitude history
        stable_frames = 5

        while not self.stop_thread:
            if self.stream is not None and self.stream.is_active():
                try:
                    data = self.stream.read(self.frames_za_buffer, exception_on_overflow=False)
                    frames = np.frombuffer(data, dtype=np.int16)
                except (IOError, ValueError) as e:
                    print(f"Error reading stream: {e}")
                    break  # Exit loop on errors

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
                print(self.Lazeni(self.peak_freq, self.nejbliz[1]))

        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

    def run(self):
        ProgramBezi = True
        while ProgramBezi:
            while self.end_it == 1:
                areaStart = pygame.Rect(490, 400, 400, 150)
                areaCredits = pygame.Rect(20, 20, 200, 100)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.stop_thread = True
                        if self.mozek_thread is not None:
                            self.mozek_thread.join()
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.MOUSEBUTTONDOWN and areaStart.collidepoint(event.pos):
                        self.end_it = 0
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                        self.end_it = 0
                    if event.type == pygame.MOUSEBUTTONDOWN and areaCredits.collidepoint(event.pos):
                        self.end_it = 3

                self.window.fill("white")
                self.window.blit(self.pozadi, (0, 0))
                text = self.font_1.render("Click here or", True, self.my_color)
                text2 = self.font_1.render("press Enter", True, self.my_color)
                text3 = self.font_1.render("to start", True, self.my_color)
                textredits = self.font_1.render("Credits", True, "red")
                self.window.blit(text, (self.WIDTH / 2 - 160, self.HEIGHT / 2 - 50))
                self.window.blit(text2, (self.WIDTH / 2 - 140, self.HEIGHT / 2))
                self.window.blit(text3, (self.WIDTH / 2 - 100, self.HEIGHT / 2 + 50))
                self.window.blit(textredits, (30, 40))
                self.window.blit(self.kytara, (self.WIDTH / 2 + 120, self.HEIGHT / 2 - 100))
                self.window.blit(pygame.transform.flip(self.kytara, True, False),
                                 (self.WIDTH / 2 - 500, self.HEIGHT / 2 - 100))
                pygame.display.flip()
                self.clock.tick(self.FPS)

            if self.end_it == 0:
                if self.mozek_thread is None or not self.mozek_thread.is_alive():
                    self.mozek_thread = threading.Thread(target=self.mozek)
                    self.mozek_thread.start()

                while self.end_it == 0:
                    # area zvuky
                    areaE2 = pygame.Rect(50, 430, 50, 40)
                    areaA2 = pygame.Rect(50, 470, 50, 40)
                    areaD3 = pygame.Rect(50, 510, 50, 40)
                    areaG3 = pygame.Rect(50, 550, 50, 40)
                    areaB3 = pygame.Rect(50, 590, 50, 40)
                    areaE4 = pygame.Rect(50, 630, 50, 40)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.stop_thread = True
                            if self.mozek_thread is not None:
                                self.mozek_thread.join()
                            pygame.quit()
                            sys.exit()
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                            self.end_it = 1
                        if event.type == pygame.MOUSEBUTTONDOWN and areaE2.collidepoint(event.pos):
                            if self.stream is not None:
                                self.stream.stop_stream()
                            pygame.mixer.Sound.play(self.E2_sound)
                            pygame.time.delay(int(self.E2_sound.get_length() * 1000))  # Wait for the sound to finish
                            if self.stream is not None:
                                self.stream.start_stream()
                        if event.type == pygame.MOUSEBUTTONDOWN and areaA2.collidepoint(event.pos):
                            if self.stream is not None:
                                self.stream.stop_stream()
                            # Play the sound
                            pygame.mixer.Sound.play(self.A2_sound)
                            pygame.time.delay(int(self.A2_sound.get_length() * 1000))
                            # Restart the audio stream
                            if self.stream is not None:
                                self.stream.start_stream()
                        if event.type == pygame.MOUSEBUTTONDOWN and areaD3.collidepoint(event.pos):
                            if self.stream is not None:
                                self.stream.stop_stream()
                            # Play the sound
                            pygame.mixer.Sound.play(self.D3_sound)
                            pygame.time.delay(int(self.D3_sound.get_length() * 1000))
                            # Restart the audio stream
                            if self.stream is not None:
                                self.stream.start_stream()
                        if event.type == pygame.MOUSEBUTTONDOWN and areaG3.collidepoint(event.pos):
                            if self.stream is not None:
                                self.stream.stop_stream()
                            # Play the sound
                            pygame.mixer.Sound.play(self.G3_sound)
                            pygame.time.delay(int(self.G3_sound.get_length() * 1000))
                            # Restart the audio stream
                            if self.stream is not None:
                                self.stream.start_stream()
                        if event.type == pygame.MOUSEBUTTONDOWN and areaB3.collidepoint(event.pos):
                            if self.stream is not None:
                                self.stream.stop_stream()
                            # Play the sound
                            pygame.mixer.Sound.play(self.B3_sound)
                            pygame.time.delay(int(self.B3_sound.get_length() * 1000))
                            # Restart the audio stream
                            if self.stream is not None:
                                self.stream.start_stream()
                        if event.type == pygame.MOUSEBUTTONDOWN and areaE4.collidepoint(event.pos):
                            if self.stream is not None:
                                self.stream.stop_stream()
                            # Play the sound
                            pygame.mixer.Sound.play(self.E4_sound)
                            pygame.time.delay(int(self.E4_sound.get_length() * 1000))
                            # Restart the audio stream
                            if self.stream is not None:
                                self.stream.start_stream()
                        self.button_click(event)




                    self.window.fill("black")
                    self.window.blit(self.pozadi2, (0, 0))

                    pygame.draw.rect(self.window, "red", areaE2)
                    pygame.draw.rect(self.window, "red", areaA2)
                    pygame.draw.rect(self.window, "red", areaD3)
                    pygame.draw.rect(self.window, "red", areaG3)
                    pygame.draw.rect(self.window, "red", areaB3)
                    pygame.draw.rect(self.window, "red", areaE4)

                    # Display tuning status
                    self.display_tuning_status()

                    # Draw button
                    pygame.draw.rect(self.window, (0, 255, 0), self.button_rect)
                    button_text = self.font.render("Stop" if self.tuning_active else "Resume", True, "black")
                    self.window.blit(button_text, (self.button_rect.x + 10, self.button_rect.y + 10))

                    # Texts to display
                    text = self.font.render(f"Peak freq: {self.peak_freq} Hz", True, "white")
                    text2 = self.font.render(f"Lazeni: {self.Lazeni(self.peak_freq, self.nejbliz[1])}", True, "white")
                    nejblizsitext = self.font_nota.render(f"{self.nejbliz[0]}", True, "black")
                    akce_text = self.font_akce.render(f"{self.akce_lazeni()}", True, "black")
                    rozdil_text = self.font_1.render(f"{self.rozdil} Hz", True, "black")
                    hrane_text = self.font_1.render(f"{self.peak_freq}", True, "black")
                    nejblizsi_text = self.font_1.render(f"{self.nejbliz[1]}", True, "black")

                    # Display texts
                    self.window.blit(nejblizsitext, (110, 120))
                    self.window.blit(akce_text, (1020, 180))
                    self.window.blit(rozdil_text, (1040, 370))
                    self.window.blit(hrane_text, (1040, 560))
                    self.window.blit(nejblizsi_text, (1040, 750))

                    # Draw struck string
                    if self.nejbliz[0] == "E2":
                        pygame.draw.line(self.window, "red", (663, 591), (662, 900), 4)
                        self.window.blit(self.sipka, (420, 400))
                    elif self.nejbliz[0] == "A2":
                        pygame.draw.line(self.window, "red", (693, 591), (687, 900), 4)
                        self.window.blit(self.sipka, (420, 280))
                    elif self.nejbliz[0] == "D3":
                        pygame.draw.line(self.window, "red", (723, 591), (716, 900), 4)
                        self.window.blit(self.sipka, (420, 170))
                    elif self.nejbliz[0] == "G3":
                        pygame.draw.line(self.window, "red", (750, 591), (746, 900), 3)
                        self.window.blit(self.sipka_reversed, (960, 180))
                    elif self.nejbliz[0] == "B3":
                        pygame.draw.line(self.window, "red", (779, 591), (778, 900), 2)
                        self.window.blit(self.sipka_reversed, (960, 290))
                    elif self.nejbliz[0] == "E4":
                        pygame.draw.line(self.window, "red", (808, 591), (810, 900), 2)
                        self.window.blit(self.sipka_reversed, (960, 400))

                    pygame.display.flip()
                    self.clock.tick(self.FPS)
            #credits
            while self.end_it == 3:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        self.end_it = 1

                self.window.fill("black")
                self.window.blit(self.pozadi, (0, 0))

                #co budem rikat
                text = self.font.render("Lead programmer/designer:", True, self.my_color)
                text2 = self.font.render("des. Josef Kaspar", True, "darkorange")
                text3 = self.font.render("Special Assistant #1", True, self.my_color)
                text4 = self.font.render("GitHub Copilot", True, "darkorange")
                text5 = self.font.render("Special Assistant #2", True, self.my_color)
                text6 = self.font.render("Chat GPT", True, "darkorange")

                #kam to budem davat
                self.window.blit(text, (self.WIDTH / 2 - 200, 200))
                self.window.blit(text2, (self.WIDTH / 2 - 130, 250))
                self.window.blit(text3, (self.WIDTH / 2 - 150, 400))
                self.window.blit(text4, (self.WIDTH / 2 - 100, 450))
                self.window.blit(text5, (self.WIDTH / 2 - 150, 600))
                self.window.blit(text6, (self.WIDTH / 2 - 65, 650))
                pygame.display.flip()
                self.clock.tick(self.FPS)
if __name__ == "__main__":
    tuner = GuitarTuner()
    tuner.run()
