
import os

proj_root_path = os.path.abspath(os.path.dirname(__file__))
print('project_root_path: ', proj_root_path)

img_dir = os.path.join(proj_root_path, 'data/cls_crop_result_on_real')

cls_dict = {
    'bg': 0,
    'seat_belt': 1
}


img_h = 64
img_w = 48
img_ch = 3

batch_size = 1

train_steps = 1000
save_n_iters = 1
eval_n_iters = 1


learning_rate = 1e-3
l2_loss_lambda = 1e-3






model_save_dir = os.path.join(proj_root_path, 'run_output')
summary_save_dir = os.path.join(proj_root_path, 'train_summary')
