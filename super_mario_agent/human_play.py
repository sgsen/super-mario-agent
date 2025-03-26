import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import pygame
import sys
import time
import numpy as np

# Scale factor for the display
SCALE = 3

# Initialize pygame with a properly sized window
pygame.init()
SCREEN_WIDTH = 256 * SCALE
SCREEN_HEIGHT = 240 * SCALE
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Super Mario Bros")

# Create the environment without human rendering
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', render_mode='rgb_array', apply_api_compatibility=True)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Print available actions
print("Available actions (SIMPLE_MOVEMENT):")
for i, action in enumerate(SIMPLE_MOVEMENT):
    print(f"{i}: {action}")

# Display controls
print("\n===== SUPER MARIO BROS CONTROLS =====")
print("→ Arrow key: Move right")
print("← Arrow key: Move left")
print("Z key: Jump (A button)")
print("X key: Run (B button)")
print("ESC: Quit game")
print("====================================\n")

# Reset the environment
obs, _ = env.reset()

# Game loop
action = 0  # Default to NOOP
running = True
clock = pygame.time.Clock()

while running:
    # Process pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
    
    # Get currently pressed keys
    keys = pygame.key.get_pressed()
    
    # Determine action based on key combinations
    if keys[pygame.K_RIGHT]:
        if keys[pygame.K_z] and keys[pygame.K_x]:
            action = 4  # Right + A + B
        elif keys[pygame.K_z]:
            action = 2  # Right + A (jump)
        elif keys[pygame.K_x]:
            action = 3  # Right + B (run)
        else:
            action = 1  # Right only
    elif keys[pygame.K_LEFT]:
        action = 6  # Left only
    elif keys[pygame.K_z]:
        action = 5  # A button only (jump in place)
    else:
        action = 0  # NOOP
    
    # Take the action in the environment
    obs, reward, done, truncated, info = env.step(action)
    print(f"action: {action}")
    print(f"info: {info}")
    print(f"reward: {reward}")
    
    # Convert observation to pygame surface and scale it
    obs_rgb = obs.transpose(1, 0, 2) if obs.shape[0] < obs.shape[1] else obs  # Make sure dimensions are right
    surf = pygame.surfarray.make_surface(obs_rgb)
    scaled_surf = pygame.transform.scale(surf, (SCREEN_WIDTH, SCREEN_HEIGHT))
    
    # Draw the scaled surface to the screen
    screen.blit(scaled_surf, (0, 0))
    pygame.display.flip()
    
    # Reset if Mario dies or completes the level
    if done:
        obs, _ = env.reset()
    
    # Control game speed
    clock.tick(60)  # 60 FPS

# Clean up
env.close()
pygame.quit()
sys.exit()