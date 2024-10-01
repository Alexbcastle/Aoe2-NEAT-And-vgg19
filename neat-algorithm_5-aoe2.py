import neat
import numpy as np
import pyautogui
import tensorflow as tf
from PIL import Image
import os
import time
import logging
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the VGG19 model (replace 'vgg19_model_3.keras' with your trained model)
model = tf.keras.models.load_model('vgg19_model_3.keras')


# Save NEAT population
def save_population(population, filename="neat_population.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump(population, f)
    logging.info(f"NEAT population saved to {filename}")


#load neat population from savefile
def load_neat_population(config_file, pickle_file="neat_population.pkl"):
    # Load the NEAT config
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    
    # Attempt to load the population from the pickle file
    try:
        with open(pickle_file, 'rb') as f:
            population = pickle.load(f)
        logging.info(f"NEAT population loaded from {pickle_file}")
    except FileNotFoundError:
        logging.error(f"File {pickle_file} not found. Starting fresh population.")
        population = neat.Population(config)  # Start a fresh population if the file isn't found
    
    # Add reporters to monitor progress (optional)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    
    return population
    

# Load NEAT population from checkpoints
def load_population(config, filename_prefix="neat_checkpoint-"):
    checkpoint_files = [f for f in os.listdir() if f.startswith(filename_prefix) and f.endswith('.pkl')]
    if checkpoint_files:
        # Load the latest checkpoint (assuming the naming convention is consistent)
        latest_checkpoint = sorted(checkpoint_files)[-1]  # Get the most recent checkpoint file
        with open(latest_checkpoint, 'rb') as f:
            population = pickle.load(f)
        logging.info(f"NEAT population loaded from {latest_checkpoint}")
        return population
    else:
        logging.info("No existing population found. Starting fresh.")
        return neat.Population(config)  # Start fresh if no saved population exists


# Load NEAT population from checkpoints
def load_populationtwo(config, filename_prefix="neat_checkpoint-"):
    checkpoint_files = [f for f in os.listdir() if f.startswith(filename_prefix)]
    
    if checkpoint_files:
        # Log the available checkpoints
        logging.info(f"Found checkpoint files: {checkpoint_files}")
        
        # Load the latest checkpoint (assuming the naming convention is consistent)
        latest_checkpoint = sorted(checkpoint_files)[-1]  # Get the most recent checkpoint file

        logging.info(f"Attempting to load checkpoint: {latest_checkpoint}")
        
        # Check if it's using NEAT's default checkpoint format (without .pkl extension)
        if latest_checkpoint.endswith(".pkl"):
            # If it's a pickle file, load using pickle (manual save)
            with open(latest_checkpoint, 'rb') as f:
                population = pickle.load(f)
            logging.info(f"NEAT population loaded from {latest_checkpoint}")
        else:
            # If it's NEAT's checkpoint file (without .pkl), use NEAT's restore function
            population = neat.Checkpointer.restore_checkpoint(latest_checkpoint)
            logging.info(f"NEAT checkpoint loaded from {latest_checkpoint}")
        
        return population
    else:
        logging.info("No existing checkpoint found. Starting fresh.")
        return neat.Population(config)  # Start fresh if no saved population exists
    


# Capture a screenshot, resize and preprocess it for VGG19 model
def capture_and_predict():
    try:
        # Define the area you want to capture (left, top, width, height)
        score_area = (1602, 825, 320, 80)  # Replace with your specific values
        screenshot = pyautogui.screenshot(region=score_area)
        
        screenshot = screenshot.resize((224, 224))  # Resize to the input size of VGG19
        screenshot = np.array(screenshot) / 255.0    # Normalize pixel values
        screenshot = np.expand_dims(screenshot, axis=0)  # Add batch dimension

        prediction = model.predict(screenshot)
        predicted_class = 1 if prediction > 0.5 else 0  # Winning: 1, Losing: 0

        print("Predicted Class:", "Winning" if predicted_class == 1 else "Losing")
        return predicted_class

    except Exception as e:
        print(f"Error capturing or predicting: {e}")
        return 0  # Default to losing if an error occurs


# Boltzmann exploration function
def boltzmann_exploration(action_probs, temperature):
    """Applies Boltzmann exploration to select actions with temperature-based randomness."""
    action_probs = np.array(action_probs)
    exp_values = np.exp(action_probs / temperature)  # Apply Boltzmann scaling
    probabilities = exp_values / np.sum(exp_values)  # Normalize to create a valid probability distribution
    return np.random.choice(len(action_probs), p=probabilities)  # Select an action based on probabilities


# Perform hotkey or mouse action based on NEAT output
def perform_action(action, genome):
    """Performs game actions based on NEAT neural network output and adds +1 fitness for each action."""
    hotkeys = ['q', 'w', 'e', 'r', 'h', 'a', '.', ',']  # Add more if necessary
    mouse_actions = ['left', 'right']
    movement_actions = [(100, 0), (-100, 0), (0, 100), (0, -100)]  # Right, Left, Down, Up

    # Calculate total number of actions expected
    total_actions = len(hotkeys) + len(mouse_actions) + len(movement_actions)

    # Check if the action length matches the expected total actions
    if len(action) != total_actions:
        logging.warning(f"Action length {len(action)} does not match the expected size of {total_actions}.")
        return  # Exit if action length is incorrect

    # Log all action probabilities for inspection
    logging.info(f"Action probabilities: {action}")

    # Track whether any action is taken
    action_taken = False

    # Execute hotkey presses and log them
    for i in range(len(hotkeys)):
        if action[i] > 0.5:
            pyautogui.press(hotkeys[i])
            logging.info(f"Hotkey '{hotkeys[i]}' pressed.")
            genome.fitness += 1  # Increment fitness for each hotkey press
            action_taken = True

    # Execute mouse actions and log them
    if action[len(hotkeys)] > 0.5:
        pyautogui.click(button='left')  # Simulate left mouse click
        logging.info("Left mouse click performed.")
        genome.fitness += 1  # Increment fitness for each mouse click
        action_taken = True

    if action[len(hotkeys) + 1] > 0.5:
        pyautogui.click(button='right')  # Simulate right mouse click
        logging.info("Right mouse click performed.")
        genome.fitness += 1  # Increment fitness for each mouse click
        action_taken = True

    # Execute mouse movements and log them
    for i, (x, y) in enumerate(movement_actions):
        if action[len(hotkeys) + len(mouse_actions) + i] > 0.5:
            pyautogui.move(x, y, duration=0.2)
            logging.info(f"Mouse moved to ({x}, {y}).")
            genome.fitness += 1  # Increment fitness for each mouse movement
            action_taken = True

    # If no action was performed, log it
    if not action_taken:
        logging.info("No action was performed based on the current NEAT output.")
    else:
        logging.info("Action(s) performed successfully based on NEAT output.")


# Restart game function
def restart_game():
    print("Restarting game...")
    print("Individual has been losing for more than 20 seconds while playing. Restarting...")
    pyautogui.click(x=1874, y=25)  # Click menu button
    time.sleep(1)
    pyautogui.click(x=1874, y=25)  # Click menu button
    time.sleep(3)
    pyautogui.click(x=907, y=679)  # Click Restart button
    time.sleep(1)
    pyautogui.click(x=907, y=679)  # Click Restart button
    time.sleep(3)
    pyautogui.click(x=851, y=602)  # Confirm Yes
    time.sleep(1)
    pyautogui.click(x=851, y=602)  # Confirm Yes
    time.sleep(3)  # Allow some time for restart


# NEAT fitness function
def fitness_function(genomes, config):
    for genome_id, genome in genomes:  # Iterate over each genome
        logging.info(f"\n--- Evaluating Genome {genome_id} ---")
        
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0  # Initialize fitness to 0 for each genome
        temperature = 4.0  # Initial temperature for Boltzmann exploration
        temperature_decay = 0.95  # Decay factor to decrease randomness over time

        losing_start_time = None  # Track when losing starts

        while True:  # Continuous loop until losing for more than 15 seconds
            predicted_class = capture_and_predict()

            # Get NEAT network output (simulate the network deciding a hotkey/mouse action)
            action_probs = net.activate([predicted_class])

            # Apply Boltzmann exploration to choose the action with some exploration
            selected_action_index = boltzmann_exploration(action_probs, temperature)

            # Make sure selected_action is a list or array of probabilities

            selected_action = np.zeros_like(action_probs)  # Initialize an array for the selected actions
            selected_action[selected_action_index] = 1  # Set the chosen action to 1

            # Perform the chosen action and update fitness for each action taken
            perform_action(selected_action, genome)  # Pass genome to perform_action for fitness update
        
            # Check if it's winning or losing
            if predicted_class == 1:
                genome.fitness += 10  # Reward for winning
                losing_start_time = None  # Reset losing start time
            else:
                genome.fitness -= 1  # Punish for losing
                if losing_start_time is None:
                    losing_start_time = time.time()  # Mark the time losing started

                # Check if losing for more than 20 seconds
                if time.time() - losing_start_time > 60:
                    restart_game()  # Restart the game
                    break  # Exit the loop after restarting

            # Decay temperature to reduce exploration over time
            temperature *= temperature_decay

            # Introduce delay to simulate real-time decision making
            time.sleep(0.5)

        
        logging.info(f"Current Fitness of Genome {genome_id}: {genome.fitness}")
    # Ensure to return fitness values for all genomes
    return [genome.fitness for _, genome in genomes]

# Custom SpeciesStatisticsReporter class
class SpeciesStatisticsReporter(neat.StatisticsReporter):
    """Custom class to collect species statistics."""
    def __init__(self):
        super().__init__()
        self.species_stats = {}

    def post_evaluate(self, config, population, species, best_genome):
        """Collect statistics after each evaluation."""
        species_data = species.species  # Get the dictionary of species

        # Use species ID as the key, and access members for fitness and size
        self.species_stats = {
            sid: {
                "fitness": np.mean([g.fitness for g in s.members.values()]),
                "size": len(s.members)
            }
            for sid, s in species_data.items()  # sid is the species ID (key), s is the Species object
        }

        logging.info(f"Species Statistics: {self.species_stats}")




# Run the NEAT algorithm
def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                          neat.DefaultSpeciesSet, neat.DefaultStagnation,
                          config_file)

    # Load previous population or create a new one
    population = load_populationtwo(config)    
    #population = neat.Checkpointer.restore_checkpoint('neat_checkpoint-99')


    # Add species statistics reporter to the population
    species_reporter = SpeciesStatisticsReporter()
    population.add_reporter(species_reporter)

    # Add a standard statistics reporter
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())

    # Save after each generation using a custom reporter
    population.add_reporter(neat.Checkpointer(generation_interval=1, time_interval_seconds=None, filename_prefix="neat_checkpoint-"))



    # Run NEAT for a number of generations
    winner = population.run(fitness_function, 100)  # Adjust the number of generations as needed

    # Save the final population
    save_population(population)

    logging.info(f"Best Genome: {winner}")
    logging.info(f"Winner Fitness: {winner.fitness}")


if __name__ == "__main__":
    # Set your NEAT configuration file
    run('neat_config.txt')
