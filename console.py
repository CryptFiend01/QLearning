from reprint import output
import time

def showBox():
    with output(output_type='dict') as out:
        for i in range(100):
            out[1] = "+---+\n"
            out[2] = "|" + str(i) + "|\n"
            out[3] = "+---+\n"
            time.sleep(0.05)

showBox()
print
