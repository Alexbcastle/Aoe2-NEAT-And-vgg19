import pyautogui
import time
import os

# Set the directory to save screenshots
SAVE_DIR = 'screenshots scoreboard'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def take_screenshot(interval, duration):
    start_time = time.time()
    screenshot_count = 0
    
    while True:
        # Check if the specified duration has passed
        if time.time() - start_time > duration:
            break
        
        # Take a screenshot
        # Define the area you want to capture (left, top, width, height)
        
        # Score area is x = 1602, y = 825, 320 width and 80 height.
        #
        #
        score_area = (1602, 825, 320, 80)  # Replace x, y, width, height with your specific values
        #
        #

        screenshot = pyautogui.screenshot(region=score_area)
        # Save the screenshot with a unique filename
        screenshot_filename = os.path.join(SAVE_DIR, f'screenshot_{screenshot_count}.png')
        screenshot.save(screenshot_filename)
        
        print(f'Screenshot saved: {screenshot_filename}')
        
        # Increment the screenshot counter
        screenshot_count += 1
        
        # Wait for the specified interval before taking the next screenshot
        time.sleep(interval)

if __name__ == '__main__':
    # Set your desired interval (in seconds) and duration (in seconds)
    interval = 5  # Time between screenshots
    duration = 2400  # Total duration to take screenshots (10 minutes)

    print(f'Taking screenshots every {interval} seconds for {duration / 60} minutes...')
    take_screenshot(interval, duration)
    print('Screenshot capturing finished.')
