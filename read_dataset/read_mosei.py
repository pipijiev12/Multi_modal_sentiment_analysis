import pickle
import numpy as np

if __name__ == '__main__':
    data = pickle.load(open('../cmumosi_cmumosei_iemocap_mult/iemocap_data.pkl','rb'))
    print(len(data['train']['labels']))
    print(len(data['test']['labels']))
    print(len(data['valid']['labels']))
    # for i in data['test']:
    #     print(i)
    # print(data['test']['labels'][0])
    # print(np.shape(data['train']['text']))
    # print(np.shape(data['test']['emotion']))
    # print(np.shape(data['valid']))
    # flag_1 = [[0,1],[1,0],[1,0],[1,0]]
    # flag_2 = [[1,0],[0,1],[1,0],[1,0]]
    # flag_3 = [[1,0],[1,0],[0,1],[1,0]]
    # flag_4 = [[1,0],[1,0],[1,0],[0,1]]
    # label_1 = 0
    # label_2 = 0
    # label_3 = 0
    # label_4 = 0
    # other = 0
    # sum = 0
    #
    #
    # print(len(data['train']['labels']))
    # for i in range(len(data['train']['labels'])):
    #     # print(data['train']['labels'][i])
    #     if (data['train']['labels'][i] == flag_1).all():
    #         label_1 += 1
    #     elif (data['train']['labels'][i] == flag_2).all():
    #         label_2 += 1
    #     elif (data['train']['labels'][i] == flag_3).all():
    #         label_3 += 1
    #     elif (data['train']['labels'][i] == flag_4).all():
    #         label_4 += 1
    #     else:
    #         other += 1
    #     sum += 1
    #
    # print(len(data['test']['labels']))
    # for i in range(len(data['test']['labels'])):
    #     # print(data['train']['labels'][i])
    #     if (data['test']['labels'][i] == flag_1).all():
    #         label_1 += 1
    #     elif (data['test']['labels'][i] == flag_2).all():
    #         label_2 += 1
    #     elif (data['test']['labels'][i] == flag_3).all():
    #         label_3 += 1
    #     elif (data['test']['labels'][i] == flag_4).all():
    #         label_4 += 1
    #     else:
    #         other += 1
    #     sum += 1
    #
    # print(len(data['valid']['labels']))
    # for i in range(len(data['valid']['labels'])):
    #     # print(data['train']['labels'][i])
    #     if (data['valid']['labels'][i] == flag_1).all():
    #         label_1 += 1
    #     elif (data['valid']['labels'][i] == flag_2).all():
    #         label_2 += 1
    #     elif (data['valid']['labels'][i] == flag_3).all():
    #         label_3 += 1
    #     elif (data['valid']['labels'][i] == flag_4).all():
    #         label_4 += 1
    #     else:
    #         other += 1
    #     sum += 1
    #
    #
    # print(label_1)
    # print(label_2)
    # print(label_3)
    # print(label_4)
    # print(other)
    # print(sum)