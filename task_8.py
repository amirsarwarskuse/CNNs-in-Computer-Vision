import json
from pathlib import Path


def main():
    # TODO: implement the three sub-tasks and call your implementation here.
    #
    #  You should save your results in a JSON named `task_8.json' under the `./output` folder. The JSON should have
    #  the following structure:
    #
    #  "task_8.1": {
    #       "fine": xx.xx  (your top-1 accuracy) for fine-grained classification
    #       "coarse": xx.xx  (your top-1 accuracy) for coarse-grained classification
    #  },
    #  "task_8.2": {
    #       "support=x": {
    #           "k=y": zz.zz  (your top-1 accuracy with support size x and k-nn classifier with k=y)
    #       } here you need to provide results with the various support sizes and ks you used for your classifier
    #   },
    #  "task_8.3": {
    #       "k=x": yy.yy  (your top-1 accuracy with k-nn classifier with k=x)
    #   } here you need to provide results ks you used for your classifier
    #
    #   *IMPORTANT*: make sure your code runs in a self-contained manner, e.g. without absolute paths. You need to
    #   specify in your README what libraries we need to install to run your code


    task_1_res = dict(fine=None, coarse=None)
    task_2_res = None
    task_3_res = None

    all_res = {'task_8.1': task_1_res, 'task_8.2': task_2_res, 'task_8.3': task_3_res}

    Path('./output').mkdir(parents=True, exist_ok=True)

    with open('./output/task_8.json', 'w') as f:
        json.dump(all_res, f, indent=4)


if __name__ == '__main__':
    main()
