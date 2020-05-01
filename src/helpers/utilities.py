import os


def create_file_name(output_folder_path, output_file_name, i,
                     placeholder_size):
    file_name, extension = \
        str.rsplit(output_file_name, ".", 1)
    final_file_name = f"{file_name}_{i:0{placeholder_size}}.{extension}"
    final_file_name = os.path.join(output_folder_path, final_file_name)

    return final_file_name
