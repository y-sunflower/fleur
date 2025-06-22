from functools import wraps


def vectorize(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if any(isinstance(arg, list) for arg in args):
            max_len = max(len(arg) if isinstance(arg, list) else 1 for arg in args)
            new_args_list = []
            for i in range(max_len):
                new_args = []
                for arg in args:
                    if isinstance(arg, list):
                        new_args.append(arg[i % len(arg)])
                    else:
                        new_args.append(arg)
                new_args_list.append(func(*new_args, **kwargs))
            return new_args_list
        return func(*args, **kwargs)

    return wrapper


@vectorize
def add(a, b):
    return a + b


@vectorize
def concat(a, b):
    return f"{a}-{b}"


print(add(1, 2))
print(add([1, 2, 3], 10))
print(add([1, 2], [10, 20]))
print(concat("x", "a"))
print(concat("x", ["a", "b", "c"]))
