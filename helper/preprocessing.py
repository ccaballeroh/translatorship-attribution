"""Functions used to preprocess the corpora"""

import os
import re
from pathlib import Path
from typing import Dict, Generator, List, Set

from helper import ROOT

CORPORA = Path(fr"{ROOT}/Corpora/")


def remove_numbers(text: str) -> str:
    """Remove numbers between brackets and parentheses with hyphens or not.

    Using regex, finds and substitute with a blank space numbers of the form
    -[12]- or -(12)- or [12] or (12)
    
    Parameters:
    text: str - string where to remove numbers from

    Returns:
    clean_text: str - string without the numbers and brackets
    """
    pattern = re.compile(r"((-?\[\d+\]-?)|(-?\(\d+\)-?))")
    clean_text = pattern.sub(r" ", text)
    return clean_text


def collapse_spaces(text: str) -> str:
    """Substitutes two or more spaces with only one.

    Parameters:
    text: str - string where to do the replacement

    Returns:
    clean_text: str - string with two o more spaces substitued by only one
    """
    pattern = re.compile(r"\s+")
    clean_text = pattern.sub(r" ", text)
    return clean_text


def remove_special(text: str, REPLACE: Dict[str, str]) -> str:
    """Replaces special characters with conventional ones.

    Parameters:
    text: str - text where to replace the special characters
    REPLACE: Dict[str, str] - dictionary of mappinps of special characters to their
                              substitutions
    
    Returns:
    text: str - text with the special characters replaced
    """
    for char, subs in REPLACE.items():
        text = text.replace(char.lower(), subs)
    return text


def special_characters(INPUT_FOLDER: Path) -> List[str]:
    """Returns a list of special characters used in all files in a directory.
    
    A special character is all that has a Unicode greater than 127.
    This function is used only once to manually build the replacement dictionary.
    
    Parameters:
    INPUT_FOLDER: Path - Path object with directory where to read all files
    
    Returns:
    A list with all special characters
    """
    assert INPUT_FOLDER.exists(), f"Path: {INPUT_FOLDER} does not exist"
    chars: Set[str] = set()  # where to store all different characters
    for file in INPUT_FOLDER.iterdir():
        with file.open("r") as f:
            text = f.read()
        chars = chars.union(set(text))
    return list(filter(lambda char: True if ord(char) > 127 else False, chars))


def preprocess(INPUT: Path, OUTPUT: Path, REPLACE: Dict[str, str]) -> None:
    """Replaces two or more spaces and special characters in all files in a directory.

    If the output directory does not exist, it creates it. Checks if input directory
    exists.

    Parameters:
    INPUT: Path - directory where to read from the files
    OUTPUT: Path - directory where to write the processed files
    REPLACE: dict - mapping of special characters to replace
    
    Returns:
    None
    """
    assert INPUT.exists(), f"Path: {INPUT} does not exist"
    if not OUTPUT.exists():
        OUTPUT.mkdir()
        print(f"Directory {OUTPUT} created!")
    for filename in INPUT.iterdir():
        with filename.open("r", encoding="cp1252") as file:
            file_content = file.read()
        file_content = collapse_spaces(remove_numbers(file_content))
        file_content = remove_special(file_content, REPLACE)
        with open(
            OUTPUT / (filename.stem + "_proc.txt"), "w", encoding="UTF-8"
        ) as file:
            file.write(file_content)
    return None


def remove_front_back_matter(filename: Path, OUTPUT: Path) -> None:
    """Remove legal information from Project Gutenberg files.
    
    Reads the file with 'filename' and outputs the same file with
    the "proc" word appended at the end of the filename in the
    'OUTPUT', but without the lines at the beginning and
    at the end of the original file containing legal information
    from Project Gutenberg.
    
    Parameters:
    filename: Path - name of the file to process
    out_folder: Path - name of the outout folder
    
    Returns:
    None
    """
    assert filename.exists(), f"File {filename} does not exist!"
    if not OUTPUT.exists():
        OUTPUT.mkdir()
        print(f"Directory {OUTPUT} created!")
    lines = []
    write = False
    with open(filename, "r", encoding="UTF-8") as f:
        for line in f:
            if line.strip().startswith("*** START OF"):
                write = True
                continue
            elif line.strip().startswith("*** END OF"):
                write = False
                break
            if write:
                lines.append(line)
    with open(OUTPUT / (filename.stem + "_proc.txt"), "w", encoding="UTF-8") as g:
        for line in lines:
            g.write(line)
    return None


def chunks(filename: Path, CHUNK_SIZE: int = 5000) -> Generator[str, None, None]:
    """Generator that yields the following chunk of the file.
    
    The output is a string with the following chunk size
    CHUNK_SIZE of the file 'filename' in the folder 'input folder'.
    
    Parameters:
    filename: Path - Path object of file to process
    CHUNK_SIZE: int - size of chunk
    
    yields:
    string of size of CHUNK_SIZE
    """
    assert filename.exists(), f"file {filename} does not exist."
    SIZE = filename.stat().st_size  # filesize
    with open(filename, "r", encoding="UTF-8") as f:
        for _ in range(SIZE // CHUNK_SIZE):
            # reads the lines that amount to the Chunksize
            # and yields a string
            yield "".join(f.readlines(CHUNK_SIZE))


def remove_temp(TEMP: Path) -> None:
    """Deletes folder with affix "temp" along all its contents.

    Parameters:
    TEMP: Path - Directory to delete

    Returns:
    None
    """
    assert TEMP.name.endswith("temp"), f"Directory does not seem temporary"
    for file in TEMP.iterdir():
        file.unlink()
    else:
        TEMP.rmdir()
        print(f"Removed {TEMP} and all its contents!")
    return None


def quixote() -> None:
    """Preprocess all files of Quixote corpus.
    
    Removes numbers between brackes, parentheses and hyphens and replace special
    characters with a hard-coded mapping constructed after inspecting all the
    special characters used in the corpus.
    
    Parameters:
    None

    Returns:
    None
    """
    INPUT_FOLDER = CORPORA / "Raw_Quixote/"
    OUTPUT_FOLDER = CORPORA / "Proc_Quixote/"
    REPLACE = dict(  # manually constructed after calling special_characters(...)
        zip(
            ["à", "é", "’", "«", "ë", "“", "‘", "ù", "ü", "”", "—", "û", "â", "ç", "è"],
            ["a", "e", "'", '"', "e", '"', "'", "u", "u", '"', "-", "u", "a", "z", "e"],
        )
    )
    try:
        preprocess(INPUT_FOLDER, OUTPUT_FOLDER, REPLACE)
    except AssertionError as e:
        print(e)
    return None


def ibsen() -> None:
    """Preprocess all files of Ibsen corpus.
    
    Removes front and back matter of files since were downloaded from
    Project Gutenmberg. Then the files are segmented in chunks of a
    predefined size. Those files are then preprocessed to remove numbers
    between brackes, parentheses and hyphens and replace special characters
    with a hard-coded mapping constructed after inspecting all the special
    characters used in the corpus. The temporary folder used for the
    intermediate steps is deleted by the end.
    
    Parameters:
    None

    Returns:
    None
    """
    INPUT_FOLDER = CORPORA / "Raw_Ibsen/"
    TEMP_FOLDER = CORPORA / "Proc_Ibsen_temp/"
    for file in INPUT_FOLDER.iterdir():
        remove_front_back_matter(file, TEMP_FOLDER)
    for file in [file for file in TEMP_FOLDER.iterdir() if file.suffix == ".txt"]:
        str_gen = chunks(file, CHUNK_SIZE=5000)
        num = 0
        for chunk in str_gen:
            num += 1
            with open(TEMP_FOLDER / (file.stem + f"_part{num:03}.txt"), "w") as f:
                f.write(chunk)
        file.unlink()
    INPUT_FOLDER = TEMP_FOLDER
    OUTPUT_FOLDER = CORPORA / "Proc_Ibsen/"
    REPLACE = dict(  # manually constructed after running special_characters(...)
        zip(
            ["ê", "ü", "é", "â", "ú", "ó", "ö", "ë", "[", "]"],
            ["e", "u", "e", "a", "u", "o", "o", "e", "(", ")"],
        )
    )
    preprocess(INPUT_FOLDER, OUTPUT_FOLDER, REPLACE)
    remove_temp(TEMP_FOLDER)


if __name__ == "__main__":
    quixote()
    ibsen()
