import pygame
import numpy as np
from constant import MAZE, COORDINATES,Action,Index,Color


def sigmoid(x:int):
    return (1/(1+(np.e ** -x)))

def visualise(screen,player):
    screen.fill((0,0,0))

    reward_font = pygame.font.SysFont(None,36)


    #Initialising squares (rewards)
    for i in range(len(MAZE)):
        for j in range(len(MAZE[i])):
            pygame.draw.rect(screen, (200 * sigmoid(MAZE[i][j])+55, 200 * sigmoid(MAZE[i][j])+55,200 * sigmoid(MAZE[i][j]+55)), (10+(150*j), 10+(150*i), 150, 150))
            pygame.draw.rect(screen, (0,255,255), (10+(150*j), 10+(150*i), 150, 150),5)
            text_surface = reward_font.render(str(MAZE[i][j]),True,(100,100,255 * sigmoid(MAZE[i][j])))
            screen.blit(text_surface, (130+(150*j),130+(150*i)))

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

def draw_reward_graph(screen, rewards, episode, player,origin=(50, 500), size=(800, 200), window_size=100):
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

        font = pygame.font.SysFont("segoeuisymbol", 28)


        image = font.render(f"{player.prev_actions}",True,Color.GREEN_R)
        screen.blit(image,(10,475))
        image = font.render(f"Random move chance: {player.epsilon}",True, Color.GREEN_R)
        screen.blit(image,(500,250))
        image = font.render(f"Reward {np.mean(rewards[-1:])} at epoch count {len(rewards)}",True, Color.GREEN_R)
        screen.blit(image,(500,275))
        image = font.render(f"Average Reward in the last 50 epochs: {np.mean(rewards[-50:])}", True, Color.GREEN_R)
        screen.blit(image, (500, 300))


        

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
    
    font = pygame.font.SysFont("segoeuisymbol", 28)
    x_pos = 500


    # --- Equation 1: Symbolic ---
    # Q(s,a) ← Q(s,a) + α[r + γ(max Q(s',a')) - Q(s,a)]


    parts_one = [
        ("Q(s,a) <= ", Color.WHITE),
        ("Q(s,a) ", Color.RED_OLD),
        ("+ ", Color.WHITE),
        ("α", Color.YELLOW_A),
        ("[", Color.WHITE),
        ("r", Color.GREEN_R),
        (" + ", Color.WHITE),
        ("γ", Color.CYAN_G),
        ("(max Q(s',a'))", Color.MAGENTA_MAX),
        (" - ", Color.WHITE),
        ("Q(s,a)", Color.RED_OLD),
        ("]", Color.WHITE)
    ]
    blit_multi_colored_text(screen, font, x_pos, 55, parts_one)

    # --- Equation 2: Variable Names/Values ---
    parts_two = [
        (f"Q[■{prev_state+1}, {a}] <= ", Color.WHITE),
        (f"Q[■{prev_state+1}, {a}] ", Color.RED_OLD),
        ("+ ", Color.WHITE),
        (f"{player.alpha}", Color.YELLOW_A),
        ("*[", Color.WHITE),
        (f"{reward}", Color.GREEN_R),
        (" + ", Color.WHITE),
        (f"{player.gamma}", Color.CYAN_G),
        (f"*max(Q[■{player.state+1}])", Color.MAGENTA_MAX),
        (" - ", Color.WHITE),
        (f"Q[■{prev_state+1}, {a}]", Color.RED_OLD),
        ("])", Color.WHITE)
    ]
    blit_multi_colored_text(screen, font, x_pos, 105, parts_two)

    # --- Equation 3: Actual Numbers ---
    # Note: I added ':.2f' formatting to keep it readable
    parts_three = [
        (f"Q[■{prev_state+1},{a}] <= ", Color.WHITE),
        (f"{old_q_value:.3f} ", Color.RED_OLD),
        ("+ ", Color.WHITE),
        (f"{player.alpha:.3f}", Color.YELLOW_A),
        (" * (", Color.WHITE),
        (f"{reward:.3f}", Color.GREEN_R),
        (" + ", Color.WHITE),
        (f"{player.gamma:.3f}", Color.CYAN_G),
        (" * ", Color.WHITE),
        (f"{max_next_q:.3f}",Color.MAGENTA_MAX),
        (" - ", Color.WHITE),
        (f"{old_q_value:.3f}", Color.RED_OLD),
        (")", Color.WHITE)
    ]
    blit_multi_colored_text(screen, font, x_pos, 155, parts_three)

    # --- Equation 4: Final Result ---
    # Assuming updated_q is calculated beforehand
    parts_four = [
        (f"Q[■{prev_state+1},{a}] = ", Color.WHITE),
        (f"{updated_q:.4f}", (100, 255, 100)) # Bright Green for final result
    ]
    blit_multi_colored_text(screen, font, x_pos, 205, parts_four)

def blit_multi_colored_text(screen, font, x, y, parts):
    """
    Draws text with different colors on the same line.
    parts: list of tuples -> [("text", (r,g,b)), ("text2", (r,g,b))]
    """
    current_x = x
    for text_content, color in parts:
        image = font.render(text_content, True, color)
        screen.blit(image, (current_x, y))
        current_x += image.get_width() # Move the cursor to the right