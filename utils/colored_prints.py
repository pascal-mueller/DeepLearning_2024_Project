def print_info(message):
    print(f"\033[92m[INFO]\033[0m {message}")  # Green


def print_warning(message):
    print(f"\033[93m[WARNING]\033[0m {message}")  # Yellow


def print_error(message):
    print(f"\033[91m[ERROR]\033[0m {message}")  # Red


def print_green(text):
    green_color_code = "\033[92m"
    reset_color_code = "\033[0m"
    print(f"{green_color_code}{text}{reset_color_code}")
