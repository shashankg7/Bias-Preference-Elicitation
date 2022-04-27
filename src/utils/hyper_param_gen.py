import os

hyper_param_file = open('./config/hyper_parms.txt', 'w')

user_lf_range = [5, 10, 20, 40]
#user_lf_range = [5, 10]
#item_lf_range: [5, 10, 20, 40]
reg_param_range = [0.000001, 0.00001, 0.0001, 0.001, 0.01]
#reg_param_range = [0.000001]

for user_lf in user_lf_range:
    for reg_param in reg_param_range:
        txt = '-user_lf ' + str(user_lf) + ' ' + '-reg_param ' + str(reg_param) + '\n'
        hyper_param_file.write(txt)
