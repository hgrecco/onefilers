
import collections
import hashlib

try:
    import dill as pickle
except ImportError:
    print('dill not installed, please do it by running\npip install dill')
    raise

try:
    import pandas as pd
except ImportError:
    print('pandas not installed, please do it by running\npip install pandas')
    raise


#: Filename for data summary
SUMMARY_FILENAME = 'summary.pandas'

#: Filename for exception summary
ERRORS_FILENAME = 'errors.pandas'

#: Filename for hash summary
HASH_FILENAME = 'pda.txt'


class PDAError(Exception):
    pass


def _hasher(obj):
    return hashlib.sha1(pickle.dumps(obj)).hexdigest()


def is_namedtuple_instance(x):
    """Return True if x is an instance of a named tuple.
    """
    return (isinstance(x, tuple) and
            isinstance(getattr(x, '__dict__', None), collections.Mapping) and
            getattr(x, '_fields', None) is not None)


def process_folder(path, funcs):
    """Process a folder capturing the information in a a DataFrame.

    Parameters
    ----------
    path : pathlib.Path
        folder to process
    funcs : List of (string, callable, callable)
        Each triplet is (level_name, folder_func, file_func)
        level_name: a description of the folder level.
        folder_func: a callable to process folders found in this level.
                     Same signature as process_folder.
        file_func: a callable to process files found in this level.
                   Same signature as process_folder

    Returns
    -------
    data : DataFrame
    errors : DataFrame
    """

    (level_name, folder_func, file_func), funcs = funcs[0], funcs[1:]

    all_df = []
    all_edf = []

    for p in path.iterdir():

        df = edf = None
        if p.is_dir():
            if folder_func is not None:
                df, edf = folder_func(p, funcs)
        else:
            if file_func is not None:
                df, edf = file_func(p, funcs)

        if df is not None:
            if level_name in df.columns:
                raise PDAError("Column name '%s' is already taken in data dataframe" % level_name)

            df[level_name] = [p.name, ] * len(df)
            all_df.append(df)

        if edf is not None:
            if level_name in edf.columns:
                raise PDAError("Column name '%s' is already taken in error dataframe" % level_name)

            edf[level_name] = [p.name, ] * len(edf)
            all_edf.append(edf)

    if all_df:
        try:
            out_df = pd.concat(all_df)
        except:
            raise PDAError('Could not concatenate dataframes. '
                           'Is the output of all analyzing functions homogenoeus?')
    else:
        out_df = None

    if all_edf:
        out_edf = pd.concat(all_edf)
    else:
        out_edf = None

    return out_df, out_edf


def process_folder_store(path, funcs):
    """Process a folder capturing the information in a a DataFrame.
    Store each a summary file in each path.

    Parameters
    ----------
    path : pathlib.Path
        folder to process
    funcs : List of (string, callable, callable)
        Each triplet is (level_name, folder_func, file_func)
        level_name: a description of the folder level.
        folder_func: a callable to process folders found in this level.
                     Same signature as process_folder.
        file_func: a callable to process files found in this level.
                   Same signature as process_folder

    Returns
    -------
    data : DataFrame
    errors : DataFrame
    """

    out_df, out_edf = process_folder(path, funcs)

    if out_df is not None:
        out_df.to_pickle(str(path.joinpath(SUMMARY_FILENAME)))
    if out_edf is not None:
        out_edf.to_pickle(str(path.joinpath(ERRORS_FILENAME)))

    return out_df, out_edf


def process_folder_cache(path, funcs):
    """Process a folder capturing the information in a a DataFrame.
    Store each a summary file in each path and load is if available.

    Parameters
    ----------
    path : pathlib.Path
        folder to process
    funcs : List of (string, callable, callable)
        Each triplet is (level_name, folder_func, file_func)
        level_name: a description of the folder level.
        folder_func: a callable to process folders found in this level.
                     Same signature as process_folder.
        file_func: a callable to process files found in this level.
                   Same signature as process_folder

    Returns
    -------
    data : DataFrame
    errors : DataFrame
    """

    # We only use the cache if the hash file exists and matches
    # the current value

    current_hash = _hasher(funcs)

    p = path.joinpath(HASH_FILENAME)

    if p.exists():
        stored_hash = p.read_text(encoding='utf-8')

        if stored_hash == current_hash:

            p = path.joinpath(SUMMARY_FILENAME)
            if p.exists():
                df = pd.read_pickle(str(p))
            else:
                df = None

            p = path.joinpath(ERRORS_FILENAME)
            if p.exists():
                edf = pd.read_pickle(str(p))
            else:
                edf = None

            return df, edf

    out_df, out_edf = process_folder_store(path, funcs)

    path.joinpath(HASH_FILENAME).write_text(current_hash,
                                            encoding='utf-8')

    return out_df, out_edf


def _expand_namedtuple_in_dict(d, **extra):
    """Helper function to expand named tuples in dict.

    For each field in tha namedtuple corresponding to key
    a new item is added to the dict with key_field=value.field

    """
    out = {}
    for key, value in d.items():
        if is_namedtuple_instance(value):
            for k in value._fields:
                out[key + '_' + k] = getattr(value, k)
        else:
            out[key] = value

    for k, v in extra.items():
        out[k] = v

    return out


def to_dataframe(func, *args, **kwargs):
    """Helper function to convert a function that returns a dictionary
    into a function that returns a DataFrame

    Parameters
    ----------
    func : callable (pathlib.Path, *args, **kwargs) -> dict
    args : arguments passed to func on each path
    kwargs : arguments passed to

    Returns
    -------
    callable

    """

    def _inner(p, funcs):

        try:
            tmp = func(p, *args, **kwargs)
        except Exception as ex:
            return None, pd.DataFrame.from_dict([dict(path=str(p), exc=str(ex))])

        if tmp is None:
            return None, None

        if isinstance(tmp, dict):
            out = [_expand_namedtuple_in_dict(tmp, path=str(p))]
        elif isinstance(tmp, list):
            out = [_expand_namedtuple_in_dict(el, path=str(p))
                   for el in tmp]
        else:
            raise PDAError("to_dataframe valid input types are dict and list, not '%s'" % type(tmp))

        return pd.DataFrame.from_dict(out), None

    return _inner


if __name__ == '__main__':

    ## A small example
    import pathlib

    import pandas as pd

    # The analyzing function, notice that we return a dict.
    def analyze_file(p):
        if p.suffix != '.txt':
            return None

        print(p)

        with p.open('r', encoding='utf-8') as fi:
            return {'chars': len(fi.read())}

    # This defines the folder hierarchy
    funcs = [('date', process_folder, None),
             ('group', process_folder, None),
             ('fov', process_folder, None),
             ('_', None, to_dataframe(analyze_file))]

    print('Hash: %s' % _hasher(funcs))

    root = pathlib.Path('/path/to/folder')
    df, edf = process_folder(root, funcs)

    print('df')
    print('--')
    print(df)
    print()
    print('edf')
    print('---')
    print(edf)


