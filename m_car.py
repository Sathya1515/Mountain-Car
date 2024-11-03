import gym
import numpy as np
import os
import time

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_state_info(position, velocity):
    print("\nCurrent State:")
    print(f"Position: {position:.3f} (-1.2 to 0.6)")
    print(f"Velocity: {velocity:.3f} (-0.07 to 0.07)")
    print(f"Goal: Reach position 0.5")
    
def play_mountain_car():
    # Create environment with rendering
    env = gym.make('MountainCar-v0', render_mode="human")
    
    print("\nWelcome to Mountain Car!")
    print("\nGoal: Get the car to the top of the mountain (position 0.5)")
    print("\nControls:")
    print("0: Push car left")
    print("1: Do nothing")
    print("2: Push car right")
    print("\nTip: Build momentum by going back and forth!")
    input("\nPress Enter to start...")
    
    # Initialize the game
    state, _ = env.reset()
    total_reward = 0
    done = False
    truncated = False
    step = 0
    
    # Action meanings
    actions = {
        0: "Push Left",
        1: "No Push",
        2: "Push Right"
    }
    
    while not (done or truncated):
        clear_screen()
        
        # Print current state
        position, velocity = state
        print_state_info(position, velocity)
        
        # Get player input
        while True:
            try:
                print("\nAvailable Actions:")
                for i, action_name in actions.items():
                    print(f"{i}: {action_name}")
                action = int(input("\nChoose your action (0-2): "))
                if 0 <= action <= 2:
                    break
                print("Invalid action! Please choose 0, 1, or 2.")
            except ValueError:
                print("Please enter a valid number!")
        
        # Take action
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1
        
        # Print action result
        print(f"\nAction taken: {actions[action]}")
        print(f"Step: {step}")
        print(f"Reward: {reward}")
        print(f"Total Reward: {total_reward}")
        
        if position >= 0.5:
            print("\nCongratulations! You reached the goal!")
        
        time.sleep(0.1)  # Add slight delay to make game playable
        
    # Game over
    print("\nGame Over!")
    print(f"Final Score: {total_reward}")
    print(f"Total Steps: {step}")
    
    env.close()
    return total_reward

def main():
    while True:
        play_mountain_car()
        play_again = input("\nWould you like to play again? (yes/no): ").lower()
        if play_again != 'yes':
            print("Thanks for playing!")
            break

if __name__ == "__main__":
    main()