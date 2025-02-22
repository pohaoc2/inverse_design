import tkinter as tk
from inverse_design.gui.abc_precomputed_gui import ABCPrecomputedGUI


def main():
    root = tk.Tk()
    app = ABCPrecomputedGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
