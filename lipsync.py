from math import log

import parselmouth as pm
import numpy as np
import matplotlib.pyplot as plt

import pygame
#####
file = "Test_DEMO.wav"
#####
def mean_formants(sound, threshold):
    formants_raw = sound.to_formant_burg(max_number_of_formants=2)
    intensities = sound.to_intensity()
    intensities.subtract(threshold)
    f1_stripped = np.array(
        [
            formants_raw.get_value_at_time(1, t)
            for t in formants_raw.xs()
            if intensities.get_value(t) >= 0
            and formants_raw.get_value_at_time(1, t) > 0
        ]
    )
    f2_stripped = np.array(
        [
            formants_raw.get_value_at_time(2, t)
            for t in formants_raw.xs()
            if intensities.get_value(t) >= 0
            and formants_raw.get_value_at_time(2, t) > 0
        ]
    )
    return (np.mean(f1_stripped), np.mean(f2_stripped))


def mean_pitch(sound, threshold):
    pitch = sound.to_pitch()
    intensities = sound.to_intensity()
    intensities.subtract(threshold)
    pitches_stripped = np.array(
        [
            pitch.get_value_at_time(t)
            for t in pitch.xs()
            if intensities.get_value(t) >= 0 and pitch.get_value_at_time(t) > 0
        ]
    )
    return np.mean(pitches_stripped)


def get_height(f1, f2, a_f1, a_f2, i_f1, i_f2, u_f1, u_f2):
    front = LinearModel(i_f1, i_f2, u_f1, u_f2)
    min_height = front.distanceToPoint(a_f1, a_f2)
    return max(0, min(1, 1 - front.distanceToPoint(log(f1), log(f2)) / min_height))


def get_roundness(f1, f2, a_f1, a_f2, i_f1, i_f2, u_f1, u_f2):
    unround = LinearModel(a_f1, a_f2, i_f1, i_f2)
    max_roundness = unround.distanceToPoint(u_f1, u_f2)
    return max(0, min(1, unround.distanceToPoint(log(f1), log(f2)) / max_roundness))


class LinearModel:
    def __init__(self, m, b):
        self.vertical = False
        self.m = m
        self.b = b

    def __init__(self, x1, y1, x2, y2):
        if abs(x2 - x1) < 0.001:
            self.vertical = True
            self.x = (x1 + x2) / 2
            self.backup = (y1 + y2) / 2
        else:
            self.vertical = False
            self.m = (y2 - y1) / (x2 - x1)
            self.b = -self.m * x1 + y1

    def calculate(self, x):
        if self.vertical:
            return self.backup
        else:
            return self.m * x + self.b

    def distanceToPoint(self, x, y):
        if self.vertical:
            return abs(x - self.x)
        else:
            return abs(-self.m * x + y - self.b) / (abs(self.m**2 + 1) ** 0.5)

sound = pm.Sound(file)

pygame.mixer.init()
pg_sound = pygame.mixer.Sound(file)

# Silence detection


silence = pm.Sound("silence.wav")
threshold = np.mean(silence.to_intensity().values.T) * 1.1

intensities = sound.to_intensity()
intensities.subtract(threshold)
# The value of intensities is below 0 if it is silent (ignoring ambient noise)


"""plt.plot(intensities.xs(),
         intensities.values.T)
plt.show()"""

# Set up models for vowel detection


a_low = pm.Sound("a_low.wav")
a_high = pm.Sound("a_high.wav")
i_low = pm.Sound("i_low.wav")
i_high = pm.Sound("i_high.wav")
u_low = pm.Sound("u_low.wav")
u_high = pm.Sound("u_high.wav")

a_low_f1, a_low_f2 = mean_formants(a_low, threshold)
a_high_f1, a_high_f2 = mean_formants(a_high, threshold)
i_low_f1, i_low_f2 = mean_formants(i_low, threshold)
i_high_f1, i_high_f2 = mean_formants(i_high, threshold)
u_low_f1, u_low_f2 = mean_formants(u_low, threshold)
u_high_f1, u_high_f2 = mean_formants(u_high, threshold)

a_low_pitch = mean_pitch(a_low, threshold)
a_high_pitch = mean_pitch(a_high, threshold)
i_low_pitch = mean_pitch(i_low, threshold)
i_high_pitch = mean_pitch(i_high, threshold)
u_low_pitch = mean_pitch(u_low, threshold)
u_high_pitch = mean_pitch(u_high, threshold)

avgpitch = (i_low_pitch * i_high_pitch) ** 0.5
avgf1 = (i_low_f1 * i_high_f1) ** 0.5
avgf2 = (i_low_f2 * i_high_f2) ** 0.5

a_f1 = LinearModel(log(a_low_pitch), log(a_low_f1), log(a_high_pitch), log(a_high_f1))
a_f2 = LinearModel(log(a_low_pitch), log(a_low_f2), log(a_high_pitch), log(a_high_f2))
i_f1 = LinearModel(log(i_low_pitch), log(i_low_f1), log(i_high_pitch), log(i_high_f1))
i_f2 = LinearModel(log(i_low_pitch), log(i_low_f2), log(i_high_pitch), log(i_high_f2))
u_f1 = LinearModel(log(u_low_pitch), log(u_low_f1), log(u_high_pitch), log(u_high_f1))
u_f2 = LinearModel(log(u_low_pitch), log(u_low_f2), log(u_high_pitch), log(u_high_f2))

# Formant detection


formants = sound.to_formant_burg(max_number_of_formants=2)

# Pitch contour setup


pitch = sound.to_pitch()

# Mouth setup
# Mouth size approaches either extreme with a logistic curve.
# Other factors work with exponential curves.


ticklength = intensities.dx
logistic_weight = ticklength / (ticklength + 0.02)
exp_weight = ticklength / (ticklength + 0.1)
mouth_size = [0.01]
height = [0.5]
roundness = [0.5]
for tick in intensities.xs():
    if pitch.get_value_at_time(tick) > 0:
        current_pitch = pitch.get_value_at_time(tick)
    else:
        current_pitch = avgpitch
    if formants.get_value_at_time(1, tick) > 0:
        f1 = formants.get_value_at_time(1, tick)
    else:
        f1 = avgf1
    if formants.get_value_at_time(2, tick) > 0:
        f2 = formants.get_value_at_time(2, tick)
    else:
        f2 = avgf2
    if intensities.get_value(tick) >= 0:
        mouth_size.append(
            mouth_size[-1] + mouth_size[-1] * (1 - mouth_size[-1]) * logistic_weight
        )
    else:
        mouth_size.append(
            mouth_size[-1] - mouth_size[-1] * (1 - mouth_size[-1]) * logistic_weight
        )
    height.append(
        height[-1] * (1 - exp_weight)
        + exp_weight
        * get_height(
            f1,
            f2,
            a_f1.calculate(log(current_pitch)),
            a_f2.calculate(log(current_pitch)),
            i_f1.calculate(log(current_pitch)),
            i_f2.calculate(log(current_pitch)),
            u_f1.calculate(log(current_pitch)),
            u_f2.calculate(log(current_pitch)),
        )
    )
    roundness.append(
        roundness[-1] * (1 - exp_weight)
        + exp_weight
        * get_roundness(
            f1,
            f2,
            a_f1.calculate(log(current_pitch)),
            a_f2.calculate(log(current_pitch)),
            i_f1.calculate(log(current_pitch)),
            i_f2.calculate(log(current_pitch)),
            u_f1.calculate(log(current_pitch)),
            u_f2.calculate(log(current_pitch)),
        )
    )
    # Clip variables

    mouth_size[-1] = max(0.05, min(mouth_size[-1], 0.95))
mouth_size.pop(0)
height.pop(0)
roundness.pop(0)

"""plt.plot(intensities.xs(), mouth_size, color = "red")
plt.plot(intensities.xs(), height, color = "green")
plt.plot(intensities.xs(), roundness, color = "blue")
plt.show()"""

# Display the mouth


pygame.init()

screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("Lipsync demo")

pg_sound.play()

running = True
start_ticks = pygame.time.get_ticks()
while running:
    pygame.time.delay(50)
    current_time = (pygame.time.get_ticks() - start_ticks) / 1000
    index = int(current_time / ticklength)

    # Draw Rectangle and Update Display

    screen.fill((0, 0, 0))
    pygame.draw.ellipse(screen, (255, 255, 0), pygame.Rect(50, 0, 300, 300))
    pygame.draw.ellipse(screen, (0, 0, 0), pygame.Rect(100, 50, 50, 100))
    pygame.draw.ellipse(screen, (0, 0, 0), pygame.Rect(250, 50, 50, 100))

    size = 100 * (mouth_size[index] if index < len(mouth_size) else 0)
    width_scale = 1 - 0.5 * roundness[index] if index < len(mouth_size) else 0
    height_scale = 1 - 0.8 * height[index] if index < len(mouth_size) else 0
    mouth_width = size * width_scale
    mouth_height = 0.6 * size * height_scale

    points = (
        (200 - mouth_width / 2, 200 - mouth_height / 2),
        (200 + mouth_width / 2, 200 - mouth_height / 2),
        (200 + mouth_width / 4, 200 + mouth_height / 2),
        (200 - mouth_width / 4, 200 + mouth_height / 2),
    )
    pygame.draw.polygon(
        screen,
        (255, 0, 0),
        points,
    )
    pygame.display.update()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
pygame.quit()
