# Copyright 2023 Purfect, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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