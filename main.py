import tensorflow as tf
from model import model
from dataset import test, accuracy_scale
import tkinter as tk
from tkinter import simpledialog


def main():
    window = tk.Tk()
    window.title("Neural network editor")

    test_button = tk.Button(window, text="Button 1", command=lambda: test())
    accuracy_button = tk.Button(window, text="Button 2", command=lambda: accuracy_scale())
    add_layer_button = tk.Button(window, text="Button 3",
                                 command=lambda: model.add_layer(
                                     simpledialog.askinteger("Input", "Number of neurons:") or 1))
    get_config_button = tk.Button(window, text="Button 4", command=lambda: print(model.get_config()))

    for button in (test_button, accuracy_button, add_layer_button, get_config_button):
        button.config(relief=tk.RAISED, bd=3, padx=10, pady=5, bg="blue", fg="white", font=("Arial", 12), width=30)
        button.config(activebackground="green", activeforeground="white", highlightcolor="blue", highlightthickness=3)

    test_button.pack(pady=10)
    accuracy_button.pack(pady=10)
    add_layer_button.pack(pady=10)
    get_config_button.pack(pady=10)

    window.mainloop()


if __name__ == '__main__':
    main()
