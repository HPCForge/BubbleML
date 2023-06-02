
def heatflux(temp, dfun, dy):
    return (dfun[0] < 0) * (1 - temp[0]) / dy
