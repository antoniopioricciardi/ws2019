import os, shutil


def create_clean_folders(paths_list):
    """
    Given a list of directory paths, checks for each one if it exists.
    If so, delete the content the folder, otherwise create it
    :param paths_list:
    :return:
    """
    assert isinstance(paths_list, list)
    for folder_path in paths_list:
        # if the folder does not exist, create it
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        # otherwise, delete all of its content
        else:
            '''
            If the directory exists, we need to remove its content.
            Therefore we have to check whether its content is a file or a folder.
            In the first case, just remove it.
            If it is a folder, we need to remove all of its content, too
            '''
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        print('deleting', file_path)
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # remove the folder and all of its subfolders
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

