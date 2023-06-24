import tkinter

from model import model
from dataset import test, accuracy_scale, epochs
import tkinter as tk
from tkinter import simpledialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

window = tk.Tk()

edit_network_screen = tk.Frame(window)
main_screen = tk.Frame(window)
accuracy_screen = tk.Frame(window)
test_result_screen = tk.Frame(window)

back_button = tk.Button(text="Назад", command=lambda: main_screen_show())
back_button.config(relief=tk.RAISED, bd=3, padx=10, pady=5, bg="red", fg="white", font=("Arial", 12), width=30)
back_button.config(activebackground="green", activeforeground="white", highlightcolor="blue", highlightthickness=3)


def display_correct_result():
    data = {
        "[5.8 2.7 3.9 1.2]": "Iris versicolor",
        "[4.7 3.2 1.6 0.2]": "Iris setosa",
        "[7.7 2.6 6.9 2.3]": "Iris virginica",
        "[4.8 3 1.4 0.1]": "Iris setosa",
        "[6.7 2.5 5.8 1.8]": "Iris virginica",
    }
    answers = []

    for key, value in data.items():
        answers.append(value)

    simpledialog.messagebox.showinfo("Правильні відповіді", "\n".join(answers))


def accuracy_scale_graph():
    train_acc_values, val_acc_values = accuracy_scale()
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(range(1, epochs + 1), train_acc_values, label='Точність навчання')
    ax.plot(range(1, epochs + 1), val_acc_values, label='Точність перевірки')
    ax.set_xlabel('Епоха')
    ax.set_ylabel('Точність')
    ax.set_title('Точність навчання та перевірки')
    ax.legend()

    canvas = FigureCanvasTkAgg(fig, master=accuracy_screen)
    canvas.draw()

    canvas.get_tk_widget().pack(side=tk.LEFT, padx=10, pady=10)


def main_screen_show():
    main_screen.pack()
    edit_network_screen.pack_forget()
    accuracy_screen.pack_forget()
    test_result_screen.pack_forget()
    back_button.pack_forget()


def edit_network_show():
    main_screen.pack_forget()
    edit_network_screen.pack()
    accuracy_screen.pack_forget()
    test_result_screen.pack_forget()
    back_button.pack(pady=10)


def accuracy_screen_show():
    main_screen.pack_forget()
    edit_network_screen.pack_forget()
    test_result_screen.pack_forget()
    accuracy_screen.pack()
    back_button.pack(pady=10)
    accuracy_scale_graph()


def test_result_screen_show():
    main_screen.pack_forget()
    edit_network_screen.pack_forget()
    accuracy_screen.pack_forget()
    test_result_screen.pack()
    back_button.pack(pady=10)


def layer_units_setter():
    layer_index = simpledialog.askinteger("Номер шару", "Введіть номер шару") or None
    layer_units = simpledialog.askinteger("Кількість нейронів", "Введіть кількість нейронів у шарі") or None
    model.set_layer_units(layer_index, layer_units)


def main():
    main_screen_show()
    window.geometry("500x500")
    window.title("Прототип редактора нейронної мережі")
    # window.resizable(False, False)

    # Main menu buttons
    test_button = tk.Button(main_screen, text="Тестувати нейронну мережу", command=lambda: test_result_screen_show())
    accuracy_button = tk.Button(main_screen, text="Оцінка точності передбачення",
                                command=lambda: accuracy_screen_show())
    get_config_button = tk.Button(main_screen, text="Переглянути конфігурацію мережі",
                                  command=lambda: tk.simpledialog.messagebox.showinfo("Конфігурація мережі",
                                                                                      model.get_config()))
    edit_network_button = tk.Button(main_screen, text="Редагувати нейронну мережу", command=lambda: edit_network_show())

    # Edit network buttons
    add_layer_button = tk.Button(edit_network_screen, text="Додати шар",
                                 command=lambda: model.add_layer(
                                     simpledialog.askinteger("Додати шар", "Введіть число нейронів:") or None))
    get_layer_button = tk.Button(edit_network_screen, text="Переглянути шар",
                                 command=lambda: simpledialog.messagebox.showinfo("Кількість нейронів вибраного шару",
                                                                                  (model.get_layer(
                                                                                      simpledialog.askinteger(
                                                                                          "Переглянути шар",
                                                                                          "Введіть номер шару:") or None))))
    delete_layer_button = tk.Button(edit_network_screen, text="Видалити шар",
                                    command=lambda: model.remove_layer(
                                        simpledialog.askinteger("Видалити шар", "Введіть номер шару:") or None))
    set_dropout_rate_button = tk.Button(edit_network_screen, text="Встановити швидкість відпадання",
                                        command=lambda: model.set_dropout_rate(
                                            simpledialog.askfloat("Встановити швидкість відпадання",
                                                                  "Введіть швидкість відпадання:") or None))
    set_layer_units_button = tk.Button(edit_network_screen, text="Встановити кількість нейронів",
                                       command=lambda: layer_units_setter())

    # Test result screen
    test_result_label = tk.Label(test_result_screen, text="Результат тестування:")
    test_result_label.config(font=("Arial", 14))
    test_result_label.pack(pady=10)
    test_result = tk.Message(test_result_screen, text=test())
    test_result.config(width=500, font=("Arial", 16))
    test_result.pack(pady=10)
    check_correct_answers_button = tk.Button(test_result_screen, text="Перевірити правильність відповідей",
                                             command=lambda: display_correct_result())

    # Setting style for buttons
    buttons = (test_button, accuracy_button, add_layer_button, get_config_button, get_layer_button,
               delete_layer_button, set_dropout_rate_button, edit_network_button, set_layer_units_button,
               check_correct_answers_button)

    for button in buttons:
        button.config(relief=tk.RAISED, bd=3, padx=10, pady=5, bg="blue", fg="white", font=("Arial", 12), width=30)
        button.config(activebackground="green", activeforeground="white", highlightcolor="blue", highlightthickness=3)

    # Packing main screen buttons
    test_button.pack(pady=10)
    accuracy_button.pack(pady=10)
    get_config_button.pack(pady=10)
    edit_network_button.pack(pady=10)

    # Packing edit network screen buttons
    add_layer_button.pack(pady=10)
    get_layer_button.pack(pady=10)
    delete_layer_button.pack(pady=10)
    set_dropout_rate_button.pack(pady=10)
    set_layer_units_button.pack(pady=10)

    # Packing test result screen buttons
    check_correct_answers_button.pack(pady=10)

    # Loop program to run without closing automatically
    window.mainloop()


if __name__ == '__main__':
    main()
