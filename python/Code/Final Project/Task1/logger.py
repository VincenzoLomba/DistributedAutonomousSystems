
from datetime import datetime

# A global variable used to keep track of the active task
activeTask = None

# A method to set the actual active task (and print a related log message)
def setActive(task):
    newLine()
    global activeTask
    activeTask = task.upper()
    log("Starting the execution of " + task + "...")

# A mathed to simply print a new (empty) line
def newLine(): log("") if activeTask else print("")

# The project log function
def log(word):
    now = datetime.now()
    print("[" + activeTask + "][" + str(now.strftime("%H:%M:%S")) + "] " + word)