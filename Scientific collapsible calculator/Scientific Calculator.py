
import tkinter as tk
import math

input_value = ""

win = tk.Tk()

win.geometry("400x392")
win.resizable(100, 100)
win.title("Calculator")

display_text = tk.StringVar()

popup_extended = False


def square_button_action():
    global input_value
    input_num = int(input_value)
    result = input_num ** 2
    display_text.set(str(result))
    input_value = str(result)

def inverse_button_action():
    global input_value
    input_num = int(input_value)
    result = input_num ** (-1)
    display_text.set(str(result))
    input_value = str(result)

def squareroot_button_action():
    global input_value
    input_num = int(input_value)
    result = input_num ** (1/2)
    display_text.set(str(result))
    input_value = str(result)

def cuberoot_button_action():
    global input_value
    input_num = int(input_value)
    result = input_num **(1/3)
    display_text.set(str(result))
    input_value = str(result)

def power3_button_action():
    global input_value
    input_num = int(input_value)
    result = input_num ** 3
    display_text.set(str(result))
    input_value = str(result)

def power5_button_action():
    global input_value
    input_num = int(input_value)
    result = input_num ** 5
    display_text.set(str(result))
    input_value = str(result)

def power7_button_action():
    global input_value
    input_num = int(input_value)
    result = input_num ** 7
    display_text.set(str(result))
    input_value = str(result)

def power11_button_action():
    global input_value
    input_num = int(input_value)
    result = input_num ** 11
    display_text.set(str(result))
    input_value = str(result)

def power13_button_action():
    global input_value
    input_num = int(input_value)
    result = input_num ** 13
    display_text.set(str(result))
    input_value = str(result)

def sqroot_button_action():
    global input_value
    input_num = int(input_value)
    result = input_num ** (1 / 2)
    display_text.set(str(result))
    input_value = str(result)
def factorial_button_action():
    global input_value
    result = 1
    input_num = int(input_value)
    for i in range(1, input_num + 1):
        result *= i
    display_text.set(str(result))
    input_value = str(result)
# Function to toggle the popup extension
def toggle_popup():
    global popup_extended
    if not popup_extended:
        win.geometry("800x392")  # Adjust the width as needed
    else:
        win.geometry("400x392")  # Original width
    popup_extended = not popup_extended

def click_button_action(item):
    global input_value
    input_value = input_value + str(item)

    display_text.set(input_value)


def clear_button_action():
    global input_value
    input_value = ""
    display_text.set("")

def log_button_action():
    global input_value
    try:
        num = float(input_value)
        result = math.log10(num)
        input_value = str(result)
        display_text.set(input_value)
    except ValueError:

        input_value = ""
        display_text.set("Error")
def back_button_action():
    global input_value
    input_value = input_value[:-1]
    display_text.set(input_value)


def equal_button_action():
    global input_value
    result = str(eval(input_value))
    display_text.set(result)
    input_value = ""

def trigonometric_button_action(func):
    global input_value
    try:
        input_num = float(input_value)
        degrees = input_num  # Input is already in degrees
        radians = math.radians(degrees)  # Convert degrees to radians
        result = func(radians)  # Use the provided trigonometric function (math.sin, math.cos, math.tan)
        decimal_places = 6  # Adjust the number of decimal places as needed
        rounded_result = round(result, decimal_places)
        display_text.set(str(rounded_result))
        input_value = str(rounded_result)
    except ValueError:
        input_value = ""
        display_text.set("Error")

def mod_button_action():
    global input_value
    try:
        input_num = float(input_value)
        result = input_num % 2
        display_text.set(str(result))
        input_value = str(result)
    except ValueError:
        input_value = ""
        display_text.set("Error")


input_frame = tk.Frame(win, width=312, height=50, bd=0, highlightbackground="black", highlightcolor="green",
                       highlightthickness=3)

input_frame.pack(side=tk.TOP)

input_field = tk.Entry(input_frame, font=('Sans Serif', 20, 'bold'), textvariable=display_text, width=28, bg="#eee", bd=0,
                       justify=tk.RIGHT)

input_field.grid(row=0, column=0)
input_field.pack(ipady=15)

btns_frame = tk.Frame(win, width=312, height=272.5, bg="green")
btns_frame.pack(side=tk.LEFT)

options_btn = tk.Button(btns_frame, text="",font=('Sans Serif', 11 ), fg="black", width=10, height=3, bd=0, bg="#eee", cursor="hand2")

options_btn.grid(row=2, column=0,padx=2,pady=2)

minus_btn = tk.Button(btns_frame, text="-",font=('Sans Serif', 11 ), fg="black", width=10, height=3, bd=0, bg="#eee", cursor="hand2",
                  command=lambda: click_button_action("-"))

minus_btn.grid(row=2, column=1,padx=2,pady=2)

clear_btn = tk.Button(btns_frame, text="C",font=('Sans Serif', 11), fg="black", width=10, height=3, bd=0, bg="#eee", cursor="hand2",
                      command=lambda: clear_button_action())

clear_btn.grid(row=2, column=2, padx=2, pady=2)

div_btn = tk.Button(btns_frame, text="/",font=('Sans Serif', 11), fg="black", width=10, height=3, bd=0, bg="#eee", cursor="hand2",
                    command=lambda: click_button_action("/"))

div_btn.grid(row=2, column=3, padx=2, pady=2)

btn_1 = tk.Button(btns_frame, text="1",font=('Sans Serif', 11), fg="black", width=10, height=3, bd=0, bg="#fff", cursor="hand2",
                  command=lambda: click_button_action(1))

btn_1.grid(row=3, column=0, padx=2, pady=2)

btn_2 = tk.Button(btns_frame, text="2",font=('Sans Serif', 11), fg="black", width=10, height=3, bd=0, bg="#fff", cursor="hand2",
                  command=lambda: click_button_action(2))

btn_2.grid(row=3, column=1, padx=2, pady=2)

btn_3 = tk.Button(btns_frame, text="3",font=('Sans Serif', 11), fg="black", width=10, height=3, bd=0, bg="#fff", cursor="hand2",
                  command=lambda: click_button_action(3))

btn_3.grid(row=3, column=2, padx=2, pady=2)

multiply_btn = tk.Button(btns_frame, text="*",font=('Sans Serif', 11), fg="black", width=10, height=3, bd=0, bg="#eee", cursor="hand2",
                         command=lambda: click_button_action("*"))

multiply_btn.grid(row=3, column=3, padx=2, pady=2)

btn_4 = tk.Button(btns_frame, text="4",font=('Sans Serif', 11), fg="black", width=10, height=3, bd=0, bg="#fff", cursor="hand2",
                  command=lambda: click_button_action(4))

btn_4.grid(row=4, column=0, padx=2, pady=2)

btn_5 = tk.Button(btns_frame, text="5",font=('Sans Serif', 11), fg="black", width=10, height=3, bd=0, bg="#fff", cursor="hand2",
                  command=lambda: click_button_action(5))

btn_5.grid(row=4, column=1, padx=2, pady=2)

btn_6 = tk.Button(btns_frame, text="6",font=('Sans Serif', 11), fg="black", width=10, height=3, bd=0, bg="#fff", cursor="hand2",
                  command=lambda: click_button_action(6))

btn_6.grid(row=4, column=2, padx=2, pady=2)

popup_btn = tk.Button(btns_frame, text="↦",font=('Sans Serif', 11), fg="black", width=10, height=3, bd=0, bg="#eee", cursor="hand2",
                      command=lambda: toggle_popup())

popup_btn.grid(row=4, column=3, padx=2, pady=2)

# 4th row

btn_7 = tk.Button(btns_frame, text="7",font=('Sans Serif', 11), fg="black", width=10, height=3, bd=0, bg="#fff", cursor="hand2",
                  command=lambda: click_button_action(7))

btn_7.grid(row=5, column=0, padx=2, pady=2)

btn_8 = tk.Button(btns_frame, text="8",font=('Sans Serif', 11), fg="black", width=10, height=3, bd=0, bg="#fff", cursor="hand2",
                  command=lambda: click_button_action(8))

btn_8.grid(row=5, column=1, padx=2, pady=2)

btn_9 = tk.Button(btns_frame, text="9",font=('Sans Serif', 11), fg="black", width=10, height=3, bd=0, bg="#fff", cursor="hand2",
                  command=lambda: click_button_action(9))

btn_9.grid(row=5, column=2, padx=2, pady=2)

plus_btn = tk.Button(btns_frame, text="+",font=('Sans Serif', 11), fg="black", width=10, height=3, bd=0, bg="#eee", cursor="hand2",
                     command=lambda: click_button_action("+"))

plus_btn.grid(row=5, column=3, padx=2, pady=2)

point_btn = tk.Button(btns_frame, text=".",font=('Sans Serif', 11), fg="black", width=10, height=3, bd=0, bg="#eee", cursor="hand2",
                      command=lambda: click_button_action("."))

point_btn.grid(row=6, column=0, padx=2, pady=2)

btn_0 = tk.Button(btns_frame, text="0",font=('Sans Serif', 11), fg="black", width=10, height=3, bd=0, bg="#fff", cursor="hand2",
                  command=lambda: click_button_action(0))

btn_0.grid(row=6, column=1, padx=2, pady=2)

back_btn = tk.Button(btns_frame, text="<--",font=('Sans Serif', 11), fg="black", width=10, height=3, bd=0, bg="#fff", cursor="hand2",
                     command=lambda: back_button_action())

back_btn.grid(row=6, column=2, padx=2, pady=2)

equals_btn = tk.Button(btns_frame, text="=",font=('Sans Serif', 11), fg="black", width=10, height=3, bd=0, bg="#eee", cursor="hand2",
                       command=lambda: equal_button_action())

equals_btn.grid(row=6, column=3, padx=2, pady=2)

#----------------------------------------------------------------------------------------

sin_btn = tk.Button(btns_frame, text="sin",font=('Sans Serif', 11 ), fg="black", width=10, height=3, bd=0, bg="#eee", cursor="hand2",
                    command=lambda: trigonometric_button_action(math.sin))

sin_btn.grid(row=2, column=4,padx=2,pady=2)

cos_btn = tk.Button(btns_frame, text="cos",font=('Sans Serif', 11 ), fg="black", width=10, height=3, bd=0, bg="#eee", cursor="hand2",
                  command=lambda: trigonometric_button_action(math.cos))

cos_btn.grid(row=2, column=5,padx=2,pady=2)

tan_btn = tk.Button(btns_frame, text="tan",font=('Sans Serif', 11), fg="black", width=10, height=3, bd=0, bg="#eee", cursor="hand2",
                      command=lambda: trigonometric_button_action(math.tan))

tan_btn.grid(row=2, column=6, padx=2, pady=2)

sqr_btn = tk.Button(btns_frame, text="x^2",font=('Sans Serif', 11), fg="black", width=10, height=3, bd=0, bg="#eee", cursor="hand2",
                    command=lambda: square_button_action())

sqr_btn.grid(row=2, column=7, padx=2, pady=2)

sqroot = tk.Button(btns_frame, text="√x",font=('Sans Serif', 11), fg="black", width=10, height=3, bd=0, bg="#fff", cursor="hand2",
                  command=lambda: sqroot_button_action())

sqroot.grid(row=3, column=4, padx=2, pady=2)

cuberoot = tk.Button(btns_frame, text="∛x",font=('Sans Serif', 11), fg="black", width=10, height=3, bd=0, bg="#fff", cursor="hand2",
                  command=lambda: cuberoot_button_action())

cuberoot.grid(row=3, column=5, padx=2, pady=2)

mod = tk.Button(btns_frame, text="%",font=('Sans Serif', 11), fg="black", width=10, height=3, bd=0, bg="#fff", cursor="hand2",
                  command=lambda: mod_button_action())

mod.grid(row=3, column=6, padx=2, pady=2)

fraction = tk.Button(btns_frame, text="1/x",font=('Sans Serif', 11), fg="black", width=10, height=3, bd=0, bg="#eee", cursor="hand2",
                         command=lambda: inverse_button_action())

fraction.grid(row=3, column=7, padx=2, pady=2)

factorial = tk.Button(btns_frame, text="!",font=('Sans Serif', 11), fg="black", width=10, height=3, bd=0, bg="#fff", cursor="hand2",
                  command=lambda: factorial_button_action())

factorial.grid(row=4, column=4, padx=2, pady=2)

cube = tk.Button(btns_frame, text="x^3",font=('Sans Serif', 11), fg="black", width=10, height=3, bd=0, bg="#fff", cursor="hand2",
                  command=lambda: power3_button_action())

cube.grid(row=4, column=5, padx=2, pady=2)

power5 = tk.Button(btns_frame, text="x^5",font=('Sans Serif', 11), fg="black", width=10, height=3, bd=0, bg="#fff", cursor="hand2",
                  command=lambda: power5_button_action())

power5.grid(row=4, column=6, padx=2, pady=2)

power7 = tk.Button(btns_frame, text="x^7",font=('Sans Serif', 11), fg="black", width=10, height=3, bd=0, bg="#eee", cursor="hand2",
                      command=lambda: power7_button_action())

power7.grid(row=4, column=7, padx=2, pady=2)

# 4th row

power11 = tk.Button(btns_frame, text="x^11",font=('Sans Serif', 11), fg="black", width=10, height=3, bd=0, bg="#fff", cursor="hand2",
                  command=lambda: power11_button_action())

power11.grid(row=5, column=4, padx=2, pady=2)

power13 = tk.Button(btns_frame, text="x^13",font=('Sans Serif', 11), fg="black", width=10, height=3, bd=0, bg="#fff", cursor="hand2",
                  command=lambda: power13_button_action())

power13.grid(row=5, column=5, padx=2, pady=2)

log = tk.Button(btns_frame, text="log",font=('Sans Serif', 11), fg="black", width=10, height=3, bd=0, bg="#fff", cursor="hand2",
                  command=lambda: log_button_action())

log.grid(row=5, column=6, padx=2, pady=2)

btn1 = tk.Button(btns_frame, text="",font=('Sans Serif', 11), fg="black", width=10, height=3, bd=0, bg="#eee", cursor="hand2")

btn1.grid(row=5, column=7, padx=2, pady=2)

btn2 = tk.Button(btns_frame, text="",font=('Sans Serif', 11), fg="black", width=10, height=3, bd=0, bg="#eee", cursor="hand2")

btn2.grid(row=6, column=4, padx=2, pady=2)

btn3 = tk.Button(btns_frame, text="",font=('Sans Serif', 11), fg="black", width=10, height=3, bd=0, bg="#fff", cursor="hand2")

btn3.grid(row=6, column=5, padx=2, pady=2)

btn4 = tk.Button(btns_frame, text="",font=('Sans Serif', 11), fg="black", width=10, height=3, bd=0, bg="#fff", cursor="hand2")

btn4.grid(row=6, column=6, padx=2, pady=2)

btn5 = tk.Button(btns_frame, text="",font=('Sans Serif', 11), fg="black", width=10, height=3, bd=0, bg="#eee", cursor="hand2")

btn5.grid(row=6, column=7, padx=2, pady=2)

win.mainloop()