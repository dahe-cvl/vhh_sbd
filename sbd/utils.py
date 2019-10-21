

class STDOUT_TYPE:
    INFO = 1
    ERROR = 2


def printCustom(msg: str, type: int):
    if(type == 1):
        print("INFO: " + msg);
    elif(type == 2):
        print("ERROR: " + msg);
    else:
        print("FATAL ERROR: stdout type does not exist!")
        exit();