from engine import TetrisEngine


width, height = 10, 20 # standard tetris friends rules
env = TetrisEngine(width, height)

# Reset the environment
state, character, features = env.clear()


def agent():
    print("Please enter the rotation and column for the current tetromino:")
    rotation = int(input().strip())
    column = int(input().strip())
    print("Rotating to {} and moving to {}".format(rotation, column))
    return rotation, column

done = False
while not done:
    # Get an action from a theoretical AI agent
    print(str(env))
    print("Current tetromino / character is {}.".format(character))
    print("The current features are: {}".format(features))
    rotation, column = agent()

    # Sim step takes action and returns results
    features, reward, done, _ = env.step((rotation, column))
