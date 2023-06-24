import tensorflow as tf
from model import model
from dataset import test, accuracy_scale
import tkinter as tk
from tkinter import simpledialog


def main():
    window = tk.Tk()
    window.title("Прототип редактора нейронної мережі")

    test_button = tk.Button(window, text="Тестувати нейронну мережу", command=lambda: test())
    accuracy_button = tk.Button(window, text="Оцінка точності передбачення", command=lambda: accuracy_scale())
    add_layer_button = tk.Button(window, text="Додати шар",
                                 command=lambda: model.add_layer(
                                     simpledialog.askinteger("Додати шар", "Введіть число нейронів(до 50):") or 1))
    get_config_button = tk.Button(window, text="Переглянути конфігурацію мережі",
                                  command=lambda: print(model.get_config()))
    get_layer_button = tk.Button(window, text="Переглянути шар",
                                 command=lambda: print(model.get_layer(
                                     simpledialog.askinteger("Переглянути шар", "Введіть номер шару:") or 0)))
    delete_layer_button = tk.Button(window, text="Видалити шар",
                                    command=lambda: model.remove_layer(
                                        simpledialog.askinteger("Видалити шар", "Введіть номер шару:") or 0))
    set_dropout_rate_button = tk.Button(window, text="Встановити швидкість відпадання",
                                        command=lambda: model.set_dropout_rate(
                                            simpledialog.askfloat("Встановити швидкість відпадання",
                                                                  "Введіть швидкість відпадання:") or 0.2))

    for button in (test_button, accuracy_button, add_layer_button, get_config_button, get_layer_button,
                   delete_layer_button, set_dropout_rate_button):
        button.config(relief=tk.RAISED, bd=3, padx=10, pady=5, bg="blue", fg="white", font=("Arial", 12), width=30)
        button.config(activebackground="green", activeforeground="white", highlightcolor="blue", highlightthickness=3)

    test_button.pack(pady=10)
    accuracy_button.pack(pady=10)
    add_layer_button.pack(pady=10)
    get_config_button.pack(pady=10)
    get_layer_button.pack(pady=10)
    delete_layer_button.pack(pady=10)
    set_dropout_rate_button.pack(pady=10)

    window.mainloop()


if __name__ == '__main__':
    main()
