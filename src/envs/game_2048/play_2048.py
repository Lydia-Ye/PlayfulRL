import os
import pygame
import numpy as np
import random
from IPython.display import clear_output, display
from PIL import Image

# Initialize Pygame with a dummy display (for headless environments)
os.environ["SDL_VIDEODRIVER"] = "dummy"
pygame.init()

# Define constants
GRID_SIZE = 4
TILE_SIZE = 100
TILE_MARGIN = 10
WINDOW_SIZE = GRID_SIZE * (TILE_SIZE + TILE_MARGIN) + TILE_MARGIN

# Define colors
BACKGROUND_COLOR = (187, 173, 160)
EMPTY_TILE_COLOR = (205, 193, 180)
TILE_COLORS = {
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
}

# Initialize Pygame font
pygame.font.init()
FONT = pygame.font.Font(None, 72)
SMALL_FONT = pygame.font.Font(None, 36)

def draw_tile(screen, value, x, y):
    rect = pygame.Rect(x, y, TILE_SIZE, TILE_SIZE)
    pygame.draw.rect(screen, TILE_COLORS.get(value, EMPTY_TILE_COLOR), rect)
    if value:
        text = FONT.render(str(value), True, (119, 110, 101))
        text_rect = text.get_rect(center=(x + TILE_SIZE / 2, y + TILE_SIZE / 2))
        screen.blit(text, text_rect)

def draw_board(board, score):
    screen = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
    screen.fill(BACKGROUND_COLOR)
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            x = col * (TILE_SIZE + TILE_MARGIN) + TILE_MARGIN
            y = row * (TILE_SIZE + TILE_MARGIN) + TILE_MARGIN
            draw_tile(screen, board[row][col], x, y)

    # Draw the score
    score_text = SMALL_FONT.render(f"Score: {score}", True, (0, 0, 0))
    screen.blit(score_text, (10, 10))

    # Save the surface as an image and display it
    pygame.image.save(screen, "2048_board.png")
    try:
        clear_output(wait=True)
        display(Image.open("2048_board.png"))
    except Exception:
        pass

def add_new_tile(board):
    empty_cells = list(zip(*np.where(board == 0)))
    if empty_cells:
        cell = random.choice(empty_cells)
        board[cell] = 4 if random.random() > 0.9 else 2
    return board

def merge_left(board):
    new_board = np.zeros((4, 4), dtype=int)
    score = 0
    for i in range(4):
        row = board[i, :]
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, 4 - len(new_row)), 'constant')
        for j in range(3):
            if new_row[j] == new_row[j + 1]:
                new_row[j] *= 2
                score += new_row[j]  # Add to score
                new_row[j + 1:] = np.roll(new_row[j + 1:], -1)
                new_row[-1] = 0
        new_board[i, :] = new_row
    return new_board, score

def merge_right(board):
    board = np.fliplr(board)
    board, score = merge_left(board)
    board = np.fliplr(board)
    return board, score

def merge_up(board):
    board = board.T
    board, score = merge_left(board)
    board = board.T
    return board, score

def merge_down(board):
    board = board.T
    board, score = merge_right(board)
    board = board.T
    return board, score

def check_game_over(board):
    if np.any(board == 2048):
        return True, "You win!"
    if np.any(board == 0):
        return False, ""
    for direction in [merge_left, merge_right, merge_up, merge_down]:
        if not np.array_equal(board, direction(board.copy())[0]):
            return False, ""
    return True, "Game over!"

def play_game():
    board = np.zeros((4, 4), dtype=int)
    board = add_new_tile(board)
    board = add_new_tile(board)
    score = 0
    draw_board(board, score)

    while True:
        move = input("Enter move (w/a/s/d) or q to quit: \n").strip().lower()
        if move == "q":
            break
        if move not in ['w', 'a', 's', 'd']:
            print("Invalid move! Use w (up), a (left), s (down), d (right).")
            continue

        if move == 'w':
            new_board, move_score = merge_up(board.copy())
        elif move == 'a':
            new_board, move_score = merge_left(board.copy())
        elif move == 's':
            new_board, move_score = merge_down(board.copy())
        elif move == 'd':
            new_board, move_score = merge_right(board.copy())

        if np.array_equal(board, new_board):
            print("Move not valid!")
            continue

        score += move_score
        board = add_new_tile(new_board)
        draw_board(board, score)

        game_over, message = check_game_over(board)
        if game_over:
            print(message)
            break

if __name__ == "__main__":
    play_game() 