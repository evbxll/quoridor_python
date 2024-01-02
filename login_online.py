from time import *
from passwords import * 
from selenium import webdriver
from selenium.webdriver.common.by import By 
from selenium.webdriver.common.keys import Keys

# Define the login URL and target page URL


driver = webdriver.Chrome()

def login():
    login_url = 'https://en.boardgamearena.com/account'
    target_url = 'https://boardgamearena.com/welcome'

    driver.get(login_url)

    # Find the username/email input field by its ID
    username_input = driver.find_element(By.CSS_SELECTOR, '#username_input')

    # Clear the field (in case it has any default value)
    username_input.clear()

    # Enter your username or email
    username_input.send_keys(boardgamearena_usernmae)

    # Find the password input field by its ID
    password_input = driver.find_element(By.CSS_SELECTOR, '#password_input')

    # Clear the field (in case it has any default value)
    password_input.clear()

    # Enter your password
    password_input.send_keys(boardgamearena_password)

    # Submit the login form (assuming there's a submit button or you can use Keys.RETURN)
    password_input.send_keys(Keys.RETURN)

    # Wait for some time to allow the login to complete (you may need to adjust the time)
    driver.implicitly_wait(20)
    sleep(10)

    # Check if you have successfully logged in (you can check the current URL or page content)
    print(driver.current_url)
    if target_url == driver.current_url:
        print('Login successful')
    else:
        print('Login failed')


def new_game():
    return


def play_current_turn_game():
    driver.get('https://boardgamearena.com/gameinprogress')
    driver.implicitly_wait(20)
    password_input = driver.find_element(By.CSS_SELECTOR, '#password_input')
    sleep(100)

def quit():
    driver.quit()
    return