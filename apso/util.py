def meaningful_locals(d):
    """
    Returns a dict "d" containing local variables. This is useful for
    saving the state of a workspace to disk, much like you would do
    in Matlab. We have to filter this dict for a few objects that are
    technically local variables but can't be saved to disk.

    The variables can be loaded back into the workspace by executing:

        for key in d.keys():
            exec(key + " = d['%s']" % key)

    ^ That statement will only work if called within the python read-
    evaluate-print-loop (REPL). The statement can't used in a predefined function.
    """
    d = {key: val for (key, val) in d.items() if '__' not in key}
    d = {key: val for (key, val) in d.items() if 'module' not in str(val)}
    d = {key: val for (key, val) in d.items() if 'function' not in str(val)}
    d = {key: val for (key, val) in d.items() if '<_io.Buffered' not in str(val)}
    return d
