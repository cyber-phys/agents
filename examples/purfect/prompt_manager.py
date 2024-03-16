import logging

# TODO: we should use this spec.
# https://github.com/malfoyslastname/character-card-spec-v2/blob/main/spec_v2.md

def read_prompt_file(file_path):
    """
    Reads a markdown file and returns its content as a string.

    :param file_path: Path to the markdown file.
    :return: Content of the file as a string, or None if an error occurs.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        logging.error("File not found: %s", file_path)
    except Exception as e:
        logging.error("An error occurred while reading %s: %s", file_path, e)

    return None