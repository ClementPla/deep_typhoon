
def remove_nan(*tensor):
    for tens in tensor:
        tens[tens != tens] = 0