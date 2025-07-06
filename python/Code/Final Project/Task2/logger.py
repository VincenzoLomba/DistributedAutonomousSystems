
from datetime import datetime

# A global variable used to keep track of the active task
activeTask = None

# A method to set the actual active task (and print a related log message)
def setActive(task):
    global activeTask
    activeTask = task.upper()
    log("Starting the execution of " + task + "...")

# A mathed to simply print a new (empty) line
def newLine(): log("") if activeTask else print("")

# The project log function
def log(word):
    now = datetime.now()
    prefix = "[" + activeTask + "][" + str(now.strftime("%H:%M:%S")) + "] "
    lines = str(word).split('\n')
    for line in lines: print(prefix + line)