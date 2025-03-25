import pygame
import numpy as np
from opensimplex import OpenSimplex
from scipy.ndimage import median_filter
import threading
from queue import Queue

# Initialize pygame
pygame.init()

# Game constants
WIDTH, HEIGHT = 800, 600
CELL_SIZE = 20
MAZE_WIDTH, MAZE_HEIGHT = 60, 30  # In cells
PLAYER_SIZE = 15

# Colors
BACKGROUND = (0, 55, 0)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BROWN = (139, 69, 19)
RED = (255, 0, 0)
BLUE = (0, 0, 255)


# Maze caching system
maze_cache = {}
chunk_generation_queue = []
CHUNK_SIZE = 4  # Size of cached maze chunks

# Threaded generation system
generation_thread = None
generation_queue = Queue()
generation_lock = threading.Lock()
noise_gen = OpenSimplex(seed=42)  # Reuse this instance
loading_chunks = set()  # Tracks chunks being generated


def get_maze_chunk(chunk_x, chunk_y):
    """Get or queue generation of a maze chunk"""
    key = (chunk_x, chunk_y)

    # Return cached chunk if available
    if key in maze_cache:
        return maze_cache[key]

    # Queue for generation if not already queued
    if key not in chunk_generation_queue:
        chunk_generation_queue.append(key)

    # Return empty chunk as placeholder
    return np.zeros((CHUNK_SIZE, CHUNK_SIZE))


def generate_chunk(chunk_x, chunk_y):
    """Generate a single chunk (moved to separate function)"""
    noise = np.zeros((CHUNK_SIZE, CHUNK_SIZE))

    for x in range(CHUNK_SIZE):
        for y in range(CHUNK_SIZE):
            noise_x = (x + chunk_x * CHUNK_SIZE) * 0.05
            noise_y = (y + chunk_y * CHUNK_SIZE) * 0.05
            noise[x][y] = noise_gen.noise2(noise_x, noise_y)

    # Threshold and filter
    threshold = 0.2
    binary_noise = np.where(np.abs(noise) < threshold, 0, 1)
    filtered_noise = median_filter(binary_noise, size=4)

    with generation_lock:
        maze_cache[(chunk_x, chunk_y)] = filtered_noise


def worker():
    """Background worker for chunk generation"""
    while True:
        key = generation_queue.get()
        if key is None:  # Sentinel value to stop the thread
            break

        chunk_x, chunk_y = key
        generate_chunk(chunk_x, chunk_y)
        generation_queue.task_done()


def process_chunk_generation():
    """Process queued chunk generation using background thread"""
    global generation_thread

    # Start worker thread if not running
    if generation_thread is None:
        generation_thread = threading.Thread(target=worker)
        generation_thread.daemon = True
        generation_thread.start()

    # Move queued chunks to background thread
    while chunk_generation_queue:
        key = chunk_generation_queue.pop(0)
        generation_queue.put(key)


def get_maze_cell(x, y):
    """Get maze cell from cached chunks, treating placeholder chunks as empty space"""
    chunk_x = x // CHUNK_SIZE
    chunk_y = y // CHUNK_SIZE
    local_x = x % CHUNK_SIZE
    local_y = y % CHUNK_SIZE
    chunk = get_maze_chunk(chunk_x, chunk_y)

    # Treat placeholder chunks as empty space until generated
    if np.all(chunk == 0) and (chunk_x, chunk_y) in chunk_generation_queue:
        return 1  # Treat as empty space while waiting for generation
    return chunk[local_x][local_y]


class Player:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.speed = 8

    def move(self, dx, dy):
        new_x = self.x + dx * self.speed
        new_y = self.y + dy * self.speed

        # Convert position to grid coordinates
        grid_x = int(new_x // CELL_SIZE)
        grid_y = int(new_y // CELL_SIZE)

        # Check if new position is valid (not a wall)
        if get_maze_cell(grid_x, grid_y) == 1:
            return  # Hit a wall

        # Allow movement in any direction
        self.x = new_x
        self.y = new_y

    def draw(self, screen, camera_x, camera_y):
        # Draw player at fixed center position
        pygame.draw.rect(
            screen,
            RED,
            (
                WIDTH // 2 - PLAYER_SIZE // 2,
                HEIGHT // 2 - PLAYER_SIZE // 2,
                PLAYER_SIZE,
                PLAYER_SIZE,
            ),
        )


def main():
    try:
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Maze Game")
        clock = pygame.time.Clock()

        player = Player(CELL_SIZE + 10, CELL_SIZE + 10)  # Start position
    finally:
        # Clean up generation thread
        if generation_thread is not None:
            generation_queue.put(None)  # Signal thread to exit
            generation_thread.join()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Handle player input
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            player.move(-1, 0)
        if keys[pygame.K_RIGHT]:
            player.move(1, 0)
        if keys[pygame.K_UP]:
            player.move(0, -1)
        if keys[pygame.K_DOWN]:
            player.move(0, 1)

        # Process chunk generation
        process_chunk_generation()

        # Check win condition
        end_x = (MAZE_WIDTH - 2) * CELL_SIZE
        end_y = (MAZE_HEIGHT - 2) * CELL_SIZE
        if (
            player.x >= end_x
            and player.x <= end_x + CELL_SIZE
            and player.y >= end_y
            and player.y <= end_y + CELL_SIZE
        ):
            print("You won!")
            running = False

        # Draw everything
        screen.fill(BACKGROUND)

        # Calculate camera offset to keep player centered while respecting world boundaries
        camera_x = WIDTH // 2 - player.x - PLAYER_SIZE // 2
        camera_y = HEIGHT // 2 - player.y - PLAYER_SIZE // 2

        # Allow camera to follow player in all directions
        # No clamping needed for unlimited movement

        # Calculate visible area in grid coordinates
        visible_start_x = int((-camera_x) // CELL_SIZE) - 1
        visible_end_x = visible_start_x + (WIDTH // CELL_SIZE) + 2
        visible_start_y = int((-camera_y) // CELL_SIZE) - 1
        visible_end_y = visible_start_y + (HEIGHT // CELL_SIZE) + 2

        # Draw visible maze chunks
        for x in range(visible_start_x, visible_end_x):
            for y in range(visible_start_y, visible_end_y):
                if get_maze_cell(x, y) == 1:
                    screen_x = x * CELL_SIZE + camera_x
                    screen_y = y * CELL_SIZE + camera_y
                    pygame.draw.rect(
                        screen,
                        BROWN,
                        (screen_x, screen_y, CELL_SIZE, CELL_SIZE),
                    )

        # Draw end point (fixed at original position)
        end_x = (MAZE_WIDTH - 2) * CELL_SIZE
        end_y = (MAZE_HEIGHT - 2) * CELL_SIZE
        screen_end_x = end_x + camera_x
        screen_end_y = end_y + camera_y
        pygame.draw.rect(
            screen, GREEN, (screen_end_x, screen_end_y, CELL_SIZE, CELL_SIZE)
        )

        player.draw(screen, camera_x, camera_y)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
