import pygame
import numpy as np
import tensorflow as tf
import sys
from pygame import gfxdraw
from PIL import Image

pygame.init()

WIDTH, HEIGHT = 400, 400
WIN = pygame.display.set_mode((WIDTH, HEIGHT + 50))
pygame.display.set_caption("Ziffern Spiel")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)

FONT = pygame.font.SysFont("consolas", 20)
INFO_FONT = pygame.font.SysFont("consolas", 20)

clock = pygame.time.Clock()

model = tf.keras.models.load_model("models/usps_model.h5")

canvas = pygame.Surface((WIDTH, HEIGHT))
canvas.fill(BLACK)

prediction = None

def predict_digit(surface):
    data = pygame.surfarray.array3d(surface)
    data = np.transpose(data, (1, 0, 2))
    img = Image.fromarray(data)
    img = img.convert("L")
    img = img.resize((16, 16), resample=Image.Resampling.LANCZOS)
    img = np.asarray(img)
    img = img / 255.0
    img = img.reshape(1, 16, 16, 1)

    pred = model.predict(img, verbose=0) # type: ignore

    print(f"Prediction{np.argmax(pred)}, {np.max(pred)}")
    return np.argmax(pred), np.max(pred)

def draw_text(win, text, y):
    txt = FONT.render(text, True, GRAY)
    win.blit(txt, (10, HEIGHT + y))

def draw_info_text(win):
    lines = [
        "Zeichne eine Ziffer (0-9)",
        "Drücke ENTER zur Vorhersage",
        "Drücke SPACE um zu löschen"
    ]
    for i, line in enumerate(lines):
        txt = INFO_FONT.render(line, True, GRAY)
        win.blit(txt, (10, HEIGHT + 5 + i * 22))

def main():
    global prediction
    drawing = False
    running = True

    while running:
        clock.tick(60)
        WIN.fill(BLACK)
        WIN.blit(canvas, (0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    drawing = True

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    drawing = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    prediction =  None
                    prediction = predict_digit(canvas)
                elif event.key == pygame.K_SPACE:
                    canvas.fill(BLACK)
                    prediction = None

        if drawing:
            mx, my = pygame.mouse.get_pos()
            if my < HEIGHT:
                pygame.draw.circle(canvas, WHITE, (mx, my), 16)


        if prediction:
            draw_text(WIN, f"Vorhersage: {prediction[0]}", 0)
            draw_text(WIN, f"Sicherheit: {prediction[1]:.2f}", 16)
        else:
            draw_info_text(WIN)

        pygame.display.update()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()


