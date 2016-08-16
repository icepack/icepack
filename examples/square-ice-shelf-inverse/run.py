
import shutil
import glob
import os
import os.path
from subprocess import call

args = {"basic":
        ["-output", "theta", "-tol", "1.0e-4",
         "-regularization", "none",
         "-temp-profile", "parabolic"],
        "noisy":
        ["-output", "theta", "-sigma", "7.5", "-tol", "1.0e-3",
         "-regularization", "none", "-temp-profile", "parabolic"],
        "regularized":
        ["-output", "theta", "-sigma", "7.5", "-tol", "1.0e-3",
         "-regularization", "square-gradient", "-temp-profile", "parabolic"],
        "boxy-sg":
        ["-output", "theta", "-tol", "1.0e-3",
         "-regularization", "square-gradient", "-temp-profile", "square"],
        "boxy-tv":
        ["-output", "theta", "-tol", "1.0e-3",
         "-regularization", "total-variation", "-temp-profile", "square"]}

if __name__ == "__main__":
    for name, arg_list in args.items():
        with open("output.txt", "w") as output_file:
            call(["build/square_ice_shelf_inverse"] + arg_list,
                 stdout = output_file)

        if not os.path.exists(name):
            os.makedirs(name)

        for f in glob.glob(r'*.ucd'):
            shutil.move(f, name)

        shutil.move("output.txt", name)
