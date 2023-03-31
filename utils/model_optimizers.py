from keras import optimizers


def SGD_loss():
    opt = optimizers.SGD(learning_rate=1e-02,
                         # decay=1e-6,
                         momentum=0.9,
                         nesterov=True
                         )
    # opt = optimizers.SGD(learning_rate=5,
    #                      decay=6,
    #                      momentum=0.9,
    #                      nesterov=True
    #                      )
    return opt

def Adam_loss():
    opt = optimizers.Adam(learning_rate=1e-02,
                          )

    return opt
