import torch

path = "/home/nearlab/Thesis_Anna_De_Luca/SuperGlue-TRAINING/warp_training/models_train1/model_epoch_163.pth"

model = Darknet(path)

model.load_state_dict(torch.load(path))


