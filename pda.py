
import pandas as pd


#: Filename for data summary
FILENAME = 'summary.pandas'

#: Filename for exception summary
ERRORS_FILENAME = 'errors.pandas'


def is_namedtuple_instance(x):
    """Return True if x is an instance of a named tuple.
    """
    return (isinstance(x, tuple) and
            isinstance(getattr(x, '__dict__', None), collections.Mapping) and
            getattr(x, '_fields', None) is not None)


def process_folder(path, funcs, use_cache=True, save_cache=True):
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
    use_cache : bool
    save_cache : bool

    Returns
    -------
    data : DataFrame
    errors : DataFrame
    """

    (level_name, folder_func, file_func), funcs = funcs[0], funcs[1:]

    if use_cache:
        df = edf = None

        try:
            p = path.joinpath(FILENAME)
            if p.exists():
                df = pd.read_pickle(str(p))

            p = path.joinpath(FILENAME)
            if edf.exists():
                edf = pd.read_pickle(str(p))

            return df, edf

        except Exception:
            pass

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
                raise Exception('Column name %s is already taken in data dataframe' % level_name)

            df[level_name] = [p.name, ] * len(df)
            all_df.append(df)

        if edf is not None:
            if level_name in edf.columns:
                raise Exception('Column name %s is already taken in error dataframe' % level_name)

            edf[level_name] = [p.name, ] * len(edf)
            all_edf.append(edf)

    if all_df:
        df = pd.concat(all_df)
        if save_cache:
            df.to_pickle(str(path.joinpath(FILENAME)))
    else:
        df = None

    if all_edf:
        edf = pd.concat(all_edf)
        if save_cache:
            edf.to_pickle(str(path.joinpath(ERRORS_FILENAME)))
    else:
        edf = None

    return df, edf


def dictfunc_to_dataframe(func, *args, **kwargs):
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
        df = edf = None
        try:
            tmp = func(p, *args, **kwargs)

            if tmp is None:
                return None, None

            out = {}
            for key, value in tmp.items():
                if is_namedtuple_instance(value):
                    for k in value._fields:
                        out[key + '_' + k] = getattr(value, k)
                else:
                    out[key] = value

            out['path'] = str(p)
            df = pd.DataFrame.from_dict([out])

        except Exception as ex:
            edf = pd.DataFrame.from_dict({path: str(p), exc: str(ex)})

        return df, edf

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
             ('_', None, dictfunc_to_dataframe(analyze_file))]

    root = pathlib.Path('/path/to/folder')
    df, edf = process_folder(root, funcs)

    print(df)
    print(edf)
