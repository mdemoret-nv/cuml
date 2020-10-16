import importlib
import inspect
import os
import sys
import numbers



def add_all_to_module(module_name):
    try:
        m = importlib.import_module(module_name)

        if (hasattr(m, "__all__")):
            return

        def is_match(x):
            if (isinstance(x, numbers.Number)):
                return True

            if (isinstance(x, str)):
                return True

            if (not hasattr(x, "__module__") or not hasattr(x, "__qualname__")):
                return False

            return m.__name__ + "." in ".".join((x.__module__, x.__qualname__))

        matching = list(dict(inspect.getmembers(m, is_match)).keys())

        # Remove dunder
        matching = [x for x in matching if not (x.startswith("__") and x.endswith("__"))]

        if (len(matching) == 0):
            return

        with open(m.__file__, "a") as f:
            f.write("\n__all__ = [\n")

            for match in sorted(matching):
                f.write("    \"{}\",\n".format(match))

            f.write("]\n")
    except:
        pass


if __name__ == "__main__":

    module_arg_name = sys.argv[1]

    module_arg = importlib.import_module(module_arg_name)

    # First, find all __init__.py files in subdirectories of this package
    root_dir = os.path.dirname(module_arg.__file__)

    root_relative = os.path.dirname(root_dir)

    # Now loop
    for root, _, files in os.walk(root_dir):

        if "__init__.py" in files:

            module_name = os.path.relpath(root, root_relative).replace(
                os.sep, ".")

            add_all_to_module(module_name=module_name)


