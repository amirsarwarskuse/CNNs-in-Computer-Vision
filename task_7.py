import task_1


def main():
    # TODO implement your task 7 solution here
    pass


def prepare_test():
    # TODO: Load the model from task 1 and return its **backbone**. The backbone model will be fed a batch of images,
    #  i.e. a tensor of shape (B, 3, 32, 32), where B >= 2, and must return a tensor of shape (B, 576), i.e.
    #  the embedding extracted for the input images. Hint: if the backbone is stored inside your model with the
    #  name "backbone", you can simply leave the code below as is. Otherwise, please adjust.

    model = task_1.prepare_test()
    return model.backbone


if __name__ == '__main__':
    main()
