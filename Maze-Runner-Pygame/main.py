import pygame
from player import Player
from visualise import visualise,draw_reward_graph,show_q_equation_and_wait
import random
import numpy as np
import time

def main():
    pygame.init()
    WIDTH, HEIGHT = 1200, 800
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("My Pygame Window")
    player = Player(0,[8])
    episode_rewards = []
    visual_actions = []
    waiting = True

    # Main loop
    episode = 1000
    for _ in range(episode):
        player.reset()
        total_reward = 0
        visual_actions.clear()
        while player.state not in player.end:
            visualise(screen,player)
    
            if player.epsilon < np.random.rand() and np.max(player.q_table[player.state]):
                action = np.argmax(player.q_table[player.state])
            else:
                action = random.randint(0,3)

            prev_state = player.state
            player.safe_move(action)

            if player.epsilon > player.min_epsilon:
                player.epsilon *= player.alpha

            reward = player.reward(player.state)
            total_reward += reward

            show_q_equation_and_wait(screen,player,prev_state,action,reward)
            player.q_table[prev_state, action] += player.alpha * (reward + player.gamma * np.max(player.q_table[player.state]) - player.q_table[prev_state, action])
            draw_reward_graph(screen,episode_rewards,episode,player)
            time.sleep(0.05)
            # Update the display

            step_mode = True

            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                        waiting = True
            while waiting and step_mode:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_RSHIFT]:
                    step_mode = False
                for event in pygame.event.get():
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        waiting = False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            step_mode = False
                    
            
                        
            pygame.display.flip()
        episode_rewards.append(total_reward)

    # Quit Pygame
    pygame.quit()



    


    
if __name__ == "__main__":
    main()
