from model import model
from dataset import test, accuracy_scale
import tkinter as tk
from tkinter import simpledialog

window = tk.Tk()

edit_network_screen = tk.Frame(window)
main_screen = tk.Frame(window)
accuracy_screen = tk.Frame(window)


def main_screen_show():
    main_screen.pack()
    edit_network_screen.pack_forget()
    accuracy_screen.pack_forget()


def edit_network_show():
    main_screen.pack_forget()
    edit_network_screen.pack()
    accuracy_screen.pack_forget()


def accuracy_screen_show():
    main_screen.pack_forget()
    edit_network_screen.pack_forget()
    accuracy_screen.pack()


def main():
    main_screen_show()
    window.geometry("500x500")
    window.title("Прототип редактора нейронної мережі")
    window.resizable(False, False)
    # Main menu buttons
    test_button = tk.Button(main_screen, text="Тестувати нейронну мережу", command=lambda: test())
    accuracy_button = tk.Button(main_screen, text="Оцінка точності передбачення", command=lambda: accuracy_scale())
    get_config_button = tk.Button(main_screen, text="Переглянути конфігурацію мережі",
                                  command=lambda: print(model.get_config()))
    edit_network_button = tk.Button(main_screen, text="Редагувати нейронну мережу", command=lambda: edit_network_show())

    # Edit network buttons
    add_layer_button = tk.Button(edit_network_screen, text="Додати шар",
                                 command=lambda: model.add_layer(
                                     simpledialog.askinteger("Додати шар", "Введіть число нейронів(до 50):") or 1))
    get_layer_button = tk.Button(edit_network_screen, text="Переглянути шар",
                                 command=lambda: print(model.get_layer(
                                     simpledialog.askinteger("Переглянути шар", "Введіть номер шару:") or 0)))
    delete_layer_button = tk.Button(edit_network_screen, text="Видалити шар",
                                    command=lambda: model.remove_layer(
                                        simpledialog.askinteger("Видалити шар", "Введіть номер шару:") or 0))
    set_dropout_rate_button = tk.Button(edit_network_screen, text="Встановити швидкість відпадання",
                                        command=lambda: model.set_dropout_rate(
                                            simpledialog.askfloat("Встановити швидкість відпадання",
                                                                  "Введіть швидкість відпадання:") or 0.2))
    # Setting style for buttons
    for button in (test_button, accuracy_button, add_layer_button, get_config_button, get_layer_button,
                   delete_layer_button, set_dropout_rate_button, edit_network_button):
        button.config(relief=tk.RAISED, bd=3, padx=10, pady=5, bg="blue", fg="white", font=("Arial", 12), width=30)
        button.config(activebackground="green", activeforeground="white", highlightcolor="blue", highlightthickness=3)

    # Packing main buttons
    test_button.pack(pady=10)
    accuracy_button.pack(pady=10)
    get_config_button.pack(pady=10)
    edit_network_button.pack(pady=10)

    # Packing edit network buttons
    add_layer_button.pack(pady=10)
    get_layer_button.pack(pady=10)
    delete_layer_button.pack(pady=10)
    set_dropout_rate_button.pack(pady=10)

    # Loop program to run without closing automatically
    window.mainloop()


if __name__ == '__main__':
    main()
