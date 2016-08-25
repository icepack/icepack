
import os
import sys
import urllib.request

def _progress_bar_hook(block_num, block_size, total_size):
    read_so_far = block_num * block_size
    if total_size > 0:
        percent = read_so_far * 100 / total_size
        s = "\r%5.1f%% %*d / %d" % (
        percent, len(str(total_size)), read_so_far, total_size)
        sys.stderr.write(s)
        if read_so_far >= total_size:
            sys.stderr.write("\n")

    else:
        sys.stderr.write("read %d\n" % (read_so_far,))


def fetch(url, filename, show_progress = True):
    hook = _progress_bar_hook if show_progress else None
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename, hook)
