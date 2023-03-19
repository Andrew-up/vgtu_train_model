from keras import optimizers


def SGD_loss():
    opt = optimizers.SGD(learning_rate=1e-02,
                         decay=1e-6,
                         momentum=0.9,
                         nesterov=False
                         )
    # opt = optimizers.SGD(learning_rate=5,
    #                      decay=6,
    #                      momentum=0.9,
    #                      nesterov=True
    #                      )
    return opt
