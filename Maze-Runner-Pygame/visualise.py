import pygame
import numpy as np
from constant import MAZE, COORDINATES,Action,Index


def sigmoid(x:int):
    return (1/(1+(np.e ** -x)))

def visualise(screen,player):
    screen.fill((0,0,0))

    reward_font = pygame.font.SysFont(None,36)


    #Initialising squares (rewards)
    for i in range(len(MAZE)):
        for j in range(len(MAZE[i])):
            pygame.draw.rect(screen, (200 * sigmoid(MAZE[i][j])+55, 200 * sigmoid(MAZE[i][j])+55,200 * sigmoid(MAZE[i][j]+55)), (10+(150*i), 10+(150*j), 150, 150))
            pygame.draw.rect(screen, (0,255,255), (10+(150*i), 10+(150*j), 150, 150),5)
            text_surface = reward_font.render(str(MAZE[i][j]),True,(100,100,255 * sigmoid(MAZE[i][j])))
            screen.blit(text_surface, (130+(150*i),130+(150*j)))

    #Initialising arrows (q-tables)
    '''
    arrow_path = "Maze-Runner-Pygame/assets/arrow.png"
    img = pygame.image.load(arrow_path).convert_alpha()
    img = pygame.transform.scale(img,(50,50))

    for i in range(len(MAZE)):
        for j in range(len(MAZE[i])-1):
            up = pygame.transform.rotate(img, 90)
            screen.blit(up, (60+(150*i),165+(150*j)))
            down = pygame.transform.rotate(img,-90)
            screen.blit(down,(60+(150*i),105+(150*j)))

    for i in range(len(MAZE)-1):
        for j in range(len(MAZE[i])):
            screen.blit(img,(105+(150*i),60+(150*j)))
            left = pygame.transform.rotate(img,180)
            screen.blit(left,(165+(150*i),60+(150*j)))

    '''


    q_font = pygame.font.SysFont(None,20)
    for i in range(MAZE.shape[0] * MAZE.shape[1]):
        x=COORDINATES[i][0]
        y=COORDINATES[i][1]

        q_value = round(player.q_table[i,Index.UP],3)
        text_surface = q_font.render(str(q_value),True,(50,255*sigmoid(q_value),50))
        screen.blit(text_surface, (x,y-50))

        q_value = round(player.q_table[i,Index.DOWN],3)
        text_surface = q_font.render(str(q_value),True,(50,255*sigmoid(q_value),50))
        screen.blit(text_surface, (x,y+50))

        q_value = round(player.q_table[i,Index.LEFT],3)
        text_surface = q_font.render(str(q_value),True,(50,255*sigmoid(q_value),50))
        screen.blit(text_surface, (x-40,y))

        q_value = round(player.q_table[i,Index.RIGHT],3)
        text_surface = q_font.render(str(q_value),True,(50,255*sigmoid(q_value),50))
        screen.blit(text_surface, (x+40,y))


    pygame.draw.circle(screen,(0,0,0),COORDINATES[player.state],10)

def draw_reward_graph(screen, rewards, episode, origin=(50, 500), size=(800, 200), window_size=100):
    """
    Draw a scrolling reward graph showing the last 'window_size' episodes.
    
    Parameters:
    - screen: PyGame screen surface
    - rewards: List of all rewards collected so far
    - episode: Current episode number (0-indexed or 1-indexed, specify in code)
    - origin: (x, y) position of graph
    - size: (width, height) of graph
    - window_size: Number of episodes to display in the graph
    """
    if len(rewards) < 2:
        return

    x0, y0 = origin
    w, h = size

    # Draw graph border
    pygame.draw.rect(screen, (255, 255, 255), (x0, y0, w, h), 2)

    # Get the last 'window_size' episodes (or all if less)
    start_episode = max(0, episode - window_size)
    

    # Calculate scaling for y-axis
    max_r = max(rewards)
    min_r = min(rewards)
    
    range_r = max_r - min_r
    scale_y = h / range_r
    
    # Draw reward line
    for i in range(len(rewards) - 1):
        # Calculate x positions based on episode number within the window
        # This makes episode 100 appear at x-coordinate 95 (for example)
        episode1 = start_episode + i
        episode2 = start_episode + i + 1
        
        # Scale episodes to fit within graph width
        x1 = x0 + (episode1 - start_episode) + w  / (window_size - 1) if window_size > 1 else x0
        x2 = x0 + (episode2 - start_episode) + w / (window_size - 1) if window_size > 1 else x0 + w
        
        # Clamp to graph boundaries
        x1 = max(x0, min(x0 + w, x1))
        x2 = max(x0, min(x0 + w, x2))
        
        # Calculate y positions
        y1 = y0 + h - (rewards[i] - min_r) * scale_y
        y2 = y0 + h - (rewards[i + 1] - min_r) * scale_y
        
        pygame.draw.line(screen, (0, 255, 0), (x1, y1), (x2, y2), 2)

def show_q_equation_and_wait(screen, player, prev_state, action, reward):
    """
    Print the Q-update equation and wait for mouse click.
    """
    # Prepare the equation string

    old_q_value = player.q_table[prev_state, action]
    max_next_q = np.max(player.q_table[player.state])
    updated_q = player.q_table[prev_state, action]+ player.alpha * (reward + player.gamma * np.max(player.q_table[player.state]) - player.q_table[prev_state, action])

    if action == Index.UP:
        a = "UP"
    elif action == Index.DOWN:
        a = "DOWN"
    elif action == Index.LEFT:
        a = "LEFT"
    else:
        a = "RIGHT"
    
    e_one = "Q(s,a) <= Q(s,a) + α[r + γ(max Q(s',a')) - Q(s,a)]"
    e_two = f"Q[{prev_state}, {a}] <= Q[{prev_state}, {a}] + {player.alpha}*[{reward} + {player.gamma}*max(Q[{player.state}]) - Q[{prev_state}, {a}])"
    e_three = f"Q[{prev_state},{a}] <= {old_q_value:.4f} + {player.alpha:.3f} * ({reward:.2f} + {player.gamma:.2f} * {max_next_q:.4f} - {old_q_value:.4f})"
    e_four = f"Q[{prev_state},{a}] <= {updated_q}"

    font = pygame.font.SysFont("segoeuisymbol", 28)
    text = font.render(e_one, True, (255, 255, 255))
    screen.blit(text, (500, 55))

    text = font.render(e_two,True,(255,255,255))
    screen.blit(text,(500,105))

    text = font.render(e_three,True,(255,255,255))
    screen.blit(text,(500,155))

    text = font.render(e_four,True,(255,255,255))
    screen.blit(text,(500,205))

    pygame.display.flip()