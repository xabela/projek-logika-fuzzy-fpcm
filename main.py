import numpy as np
import pandas as pd


def one_normalize(array_data):
    result = array_data / sum(array_data)
    return result


def fsc_clustering(dataframe, class_count=2, m=2, max_iter=1000, error_threshold=0.00001, debug=False):
    # n = jumlah data, 
    print(dataframe.shape)
    n, feature_count = dataframe.shape # multi-assignment

    u = np.absolute(np.random.randn(len(dataframe.index), class_count))
    print(u)
    for (row_index, row_u) in enumerate(u):
        u[row_index] = one_normalize(row_u)
    print(u)

    prev_obj_func_result = 0

    must_continue = True
    current_iter = 1
    while must_continue and current_iter <= max_iter:
        v = np.zeros((class_count, feature_count))
        r = np.zeros((class_count, feature_count))
        d_square = np.zeros((n, class_count))
        new_u = np.zeros((n, class_count))

        for class_index in range(class_count):
            for feature_index in range(feature_count):
                uik_m_array = [pow(u[k][class_index], m) for k in range(n)]
                sum_of_uik_m_array = sum(uik_m_array)
                v[class_index][feature_index] = sum([
                    uik_element * dataframe.iloc[k_index][feature_index]
                    for k_index, uik_element in enumerate(uik_m_array)
                ]) / sum_of_uik_m_array
                r[class_index][feature_index] = sum([
                    uik_element * np.absolute(dataframe.iloc[k_index][feature_index] - v[class_index][feature_index])
                    for k_index, uik_element in enumerate(uik_m_array)
                ]) / sum_of_uik_m_array

        for data_index in range(n):
            for class_index in range(class_count):
                d_square[data_index][class_index] = sum([
                    pow(abs(dataframe.iloc[data_index][feature_index] - v[class_index][feature_index]) - r[class_index][feature_index], 2)
                    for feature_index in range(feature_count)
                ])

        for data_index in range(n):
            for class_index in range(class_count):
                new_u[data_index][class_index] = 1 / sum([
                    pow(d_square[data_index][class_index] / d_square[data_index][j], 1 / (m - 1))
                    for j in range(class_count)
                ])

        obj_func_result = sum([
            sum([
                pow(u[data_index][class_index], m) * d_square[data_index][class_index]
                for data_index in range(n)
            ])
            for class_index in range(class_count)
        ])
        error = abs(prev_obj_func_result - obj_func_result)
        if debug:
            print(f'iteration {current_iter}, obj. funct = {obj_func_result}, error = {error}')

        current_iter += 1
        u = new_u
        prev_obj_func_result = obj_func_result
        must_continue = prev_obj_func_result is None or error > error_threshold

    print(u, "\n")

    for u_row in u:
        max_prob = max(u_row)

        for uik in u_row:
            print(" x  " if uik == max_prob else "   ", end="")
        print()


# path = input("Masukkan path file data: ")
path = "logfuz.xlsx"
df = pd.read_excel(path)
fsc_clustering(df, debug=True)
