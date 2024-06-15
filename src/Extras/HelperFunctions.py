def move_object(object , xyz):
    x = object.getX()
    y = object.getY()
    z = object.getZ()

    object.setX(x + xyz[0])
    object.setY(y + xyz[1])
    object.setZ(z + xyz[2])
        

def rotate_object(object , xyz):
    x = object.getH()
    y = object.getP()
    z = object.getR()

    object.setH(x + xyz[0])
    object.setP(y + xyz[1])
    object.setR(z + xyz[2])