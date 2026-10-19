import torch

def check_1():
    from task_1 import prepare_test
    model = prepare_test()
    n = 2
    x = torch.rand(n, 3, 32, 32)
    y = model(x)
    assert y.shape == (n, 100)
    print('Task 1 check passed')

def check_2():
    from task_2 import prepare_test
    model = prepare_test()
    n = 2
    x = torch.rand(n, 3, 32, 32)
    y = model(x)
    assert y.shape == (n, 20)
    print('Task 2 check passed')

def check_3():
    from task_3 import prepare_test
    model = prepare_test()
    n = 2
    x = torch.rand(n, 3, 32, 32)
    yf, yc = model(x)
    assert yf.shape == (n, 100) and yc.shape == (n, 20)
    print('Task 3 check passed')

def check_4(margins):
    from task_4 import prepare_test
    n = 2
    x = torch.rand(n, 3, 32, 32)

    for m in margins:
        model_fine = prepare_test(m, True)
        model_coarse = prepare_test(m, False)

        for model in (model_fine, model_coarse):
            y = model(x)
            assert y.shape == (n, 576)

    print('Task 4 check passed')

def check_5():
    from task_5 import prepare_test
    model = prepare_test()
    n = 2
    x = torch.rand(n, 3, 32, 32)
    y = model(x)
    assert y.shape == (n, 576)
    print('Task 5 check passed')

def check_7():
    from task_7 import prepare_test
    model = prepare_test()
    n = 2
    x = torch.rand(n, 3, 32, 32)
    y = model(x)
    assert y.shape == (n, 576)
    print('Task 7 check passed')

if __name__ == '__main__':
    margins = (0.2, 0.5, 1.0)  # TODO adjust to a value for which you have stored the fine and coarse models for task 4
    check_1()
    check_2()
    check_3()
    check_4(margins)
    check_5()
    check_7()