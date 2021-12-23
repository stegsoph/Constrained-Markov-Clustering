from pathlib import Path
import pandas as pd


def flatten(t):
    list_flat = [item for sublist in t for item in sublist]
    return list_flat


def save_results(iteration, dataset_str, beta_vec, knns, restarts, 
                 labels_ann, labels_seq, NMI_ann, NMI_seq,
                 percentage, wrong_percentage=0, ClassLabels='All',
                 csv_string='Results.csv'):

    N_beta = len(beta_vec)
    type_list = flatten([['Annealing']*N_beta, ['Sequential']*N_beta])

    iteration_list = [iteration]*N_beta*2
    data_list = [dataset_str]*N_beta*2
    beta_list = flatten([beta_vec.tolist(),
                         beta_vec.tolist()])
    k_list = [knns]*N_beta*2
    restart_list = [restarts]*N_beta*2
    ClassLabels_list = [ClassLabels]*N_beta*2
    labels_list = flatten([[labels_ann[i] for i in range(N_beta)],
                      [labels_seq[i] for i in range(N_beta)]])
    NMI_list = flatten([[NMI_ann[i] for i in range(N_beta)],
                        [NMI_seq[i] for i in range(N_beta)]])
    percentage_list = [percentage]*N_beta*2
    wrong_percentage_list = [wrong_percentage]*N_beta*2


    my_file = Path("..","data","results",csv_string)
    if not my_file.is_file():
        print("CSV-File does not exist!")
        df = pd.DataFrame(columns=['Iteration', 'Dataset', 'Type', 'Beta', 'knns', 'Restarts', 
                                   'NMI', 'Labels',
                                   'Percentage', 'WrongPercentage', 'ClassLabels'])
    else:
        df = pd.read_csv(my_file)

    for i in range(N_beta*2):

        dict_index = {'Iteration': iteration_list[i],
                      'Dataset': data_list[i],
                      'Type': type_list[i],
                      'Beta': beta_list[i],
                      'knns': k_list[i],
                      'Restarts': restart_list[i],
                      'Percentage': percentage_list[i],
                      'WrongPercentage': wrong_percentage_list[i],
                      'ClassLabels': ClassLabels_list[i],
                      }

        index = df.index[(df[list(dict_index)] ==
                          pd.Series(dict_index)).all(axis=1)].tolist()

        if not(index):
            print("add row")
            NMI_label_dict = {'NMI': NMI_list[i], 'Labels': labels_list[i]}
            dict_add = dict(dict_index, **NMI_label_dict)
            df = df.append(dict_add, ignore_index=True)
        else:
            print("update row")
            df.drop(index, inplace=True)
            NMI_label_dict = {'NMI': NMI_list[i], 'Labels': labels_list[i]}
            dict_add = dict(dict_index, **NMI_label_dict)
            df = df.append(dict_add, ignore_index=True)

    df.to_csv(my_file, index=False)

    return df

