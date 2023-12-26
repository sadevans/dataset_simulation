import os
from pathlib import Path
import re
import shutil

def global_collect_masks():
    root_dir = input('Введите путь: ')
    search_dir = input('Введите имя папки, в которой располагются маски: ')
    if len(search_dir) == 0: search_dir = 'mask_scunet'
    name = root_dir.split('/')[-1]
    # print( search_dir)
    
    save_bin_dir = os.path.join('./data', name) + '/bin_masks/'
    save_semantic_dir = os.path.join('./data', name) + '/semantic_masks/'

    os.makedirs(os.path.join('./data', name), exist_ok=True)
    os.makedirs(save_bin_dir, exist_ok=True)
    os.makedirs(save_semantic_dir, exist_ok=True)
    # all_dirs = os.listdir(root_dir)
    # print('ALL DIRS: ', os.listdir(root_dir))
    # root, dirs, files = os.walk(root_dir)
    # print(os.walk(root_dir))
    for root, dirs, files in os.walk(root_dir):
        # print('!!!!!!!!!!!!!!!!!!!!')
        # print(dirs)
        if root.split('/')[-1] == search_dir:
            # print(root, files)
            name_ = root.split('/')
            name_plus = re.sub(r'[^\w\s]', '', name_[-3]).replace(' ', '_') + '_' + re.sub(r'[^\w\s]', '', name_[-2]).replace(' ', '_') if name[-3] != name else re.sub(r'[^\w\s]', '', name_[-2]).replace(' ', '_')
            # print(name_plus)
            for file_name in files:
                source_file = os.path.join(root, file_name)
                dest_file = os.path.join(save_bin_dir, name_plus + '_' + file_name)
                shutil.copyfile(source_file, dest_file)
                # print(dest_file)

    return save_bin_dir, save_semantic_dir
            

if __name__ == '__main__':
    global_collect_masks()