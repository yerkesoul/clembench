from typing import Dict
import os
import json
import csv


def project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def game_dir(game_name: str) -> str:
    return os.path.join(project_root(), "games", game_name)


def results_dir(game_name: str) -> str:
    return os.path.join(project_root(), "results", game_name)


def load_json(file_name: str, game_name: str, is_results_file=False) -> Dict:
    data = load_file(file_name, game_name, file_ending=".json", is_results_file=is_results_file)
    data = json.loads(data)
    return data


def load_csv(file_name: str, game_name: str) -> Dict:
    # iso8859_2 was required for opening nytcrosswords.csv for clues in wordle
    rows = []
    fp = file_path(file_name, game_name)
    with open(fp, encoding='iso8859_2') as csv_file:
        data = csv.reader(csv_file, delimiter=',')
        # header = next(data)
        for row in data:
            rows.append(row)
    return rows


def load_template(file_name: str, game_name: str) -> str:
    return load_file(file_name, game_name, file_ending=".template")


def file_path(file_name: str, game_name: str = None, is_results_file=False) -> str:
    if is_results_file:
        return os.path.join(results_dir(game_name), file_name)
    if game_name:
        return os.path.join(game_dir(game_name), file_name)
    return os.path.join(project_root(), file_name)


def load_file(file_name: str, game_name: str = None, file_ending: str = None, is_results_file=False) -> str:
    if file_ending and not file_name.endswith(file_ending):
        file_name = file_name + file_ending
    fp = file_path(file_name, game_name, is_results_file)
    with open(fp, encoding='utf8') as f:
        data = f.read()
    return data


def store_game_results_file(data, file_name: str, game_name: str, sub_dir: str = None,
                            do_overwrite: bool = True) -> str:
    return store_file(data, file_name, results_dir(game_name), sub_dir, do_overwrite)


def store_game_file(data, file_name: str, game_name: str, sub_dir: str = None, do_overwrite: bool = True) -> str:
    return store_file(data, file_name, game_dir(game_name), sub_dir, do_overwrite)


def store_file(data, file_name: str, dir_path: str, sub_dir: str = None, do_overwrite: bool = True) -> str:
    """
    :param data: to store
    :param file_name: of the file to store
    :param dir_path: to the directory to store to
    :param sub_dir: optional subdirectories
    :param do_overwrite: default: True
    :return: the file path
    """
    if sub_dir:
        dir_path = os.path.join(dir_path, sub_dir)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    fp = os.path.join(dir_path, file_name)
    if not do_overwrite:
        if os.path.exists(fp):
            raise FileExistsError(fp)

    with open(fp, "w", encoding='utf-8') as f:
        if file_name.endswith(".json"):
            json.dump(data, f, ensure_ascii=False)
        else:
            f.write(data)
    return fp
