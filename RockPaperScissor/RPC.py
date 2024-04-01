from tkinter import *
import numpy as np

root = Tk()
root.geometry('400x400')
root.resizable(0,0)
root.title('Yogesh\'s Rock,Paper,Scissors')
root.config(bg ='seashell3')

user_take = StringVar()
Label(root, text='Choose one: Rock, Paper, Scissors', font=('arial 15 bold',16), bg='seashell3').place(x=30,y=90)
Entry(root, font='arial 15', textvariable=user_take, bg='antiquewhite2').place(x=90,y=140)

Result = StringVar()

user_score = StringVar()
comp_score = StringVar()
Entry(root, font='arial 15', textvariable=user_score, bg='antiquewhite2',width=3).place(x=30,y=10)
Label(root, text='Your Score', font=('arial 15 bold',12), bg='seashell3').place(x=10,y=45)
Entry(root, font='arial 15', textvariable=comp_score, bg='antiquewhite2', width=3).place(x=335,y=10)
Label(root, text='Computer Score', font=('arial 15 bold',12), bg='seashell3').place(x=270,y=45)

def play():

        comp_pick = np.random.randint(1, 4)
        if comp_pick == 1:
            comp_pick = 'rock'
        elif comp_pick == 2:
            comp_pick = 'paper'
        elif comp_pick == 3:
            comp_pick = 'scissors'

        user_pick = user_take.get()
        num_user=0
        num_comp=0
        if user_pick == comp_pick:
            Result.set('tie,you both select same')
            user_score.set(num_user)
            comp_score.set(num_comp)

        elif user_pick == 'rock' and comp_pick == 'paper':
            Result.set('you loose,computer select paper')
            num_comp += 1
            if num_user==0:
                num_user=0
                comp_score.set(num_comp)
                user_score.set(num_user)

        elif user_pick == 'rock' and comp_pick == 'scissors':
            Result.set('you win,computer select scissors')
            num_user += 1
            num_comp=0
            user_score.set(num_user)
            comp_score.set(num_comp)

        elif user_pick == 'paper' and comp_pick == 'scissors':
            Result.set('you loose,computer select scissors')
            num_comp += 1
            num_user=0
            comp_score.set(num_comp)
            user_score.set(num_user)

        elif user_pick == 'paper' and comp_pick == 'rock':
            Result.set('you win,computer select rock')
            num_user += 1
            num_comp=0
            user_score.set(num_user)
            comp_score.set(num_comp)

        elif user_pick == 'scissors' and comp_pick == 'rock':
            Result.set('you loose,computer select rock')
            num_user=0
            num_comp += 1
            comp_score.set(num_comp)
            user_score.set(num_user)

        elif user_pick == 'scissors' and comp_pick == 'paper':
            Result.set('you win ,computer select paper')
            num_user += 1
            num_comp=0
            user_score.set(num_user)
            comp_score.set(num_comp)
        else:
            Result.set('invalid: choose any one -- rock, paper, scissors')

def Reset():
    Result.set("")
    user_take.set("")
    user_score.set('0')
    comp_score.set('0')



def Exit():
    root.destroy()

Entry(root, font='arial 10 bold', textvariable=Result, bg='antiquewhite2', width=50, ).place(x=30, y=250)
Button(root, font='arial 13 bold', text='PLAY', padx=5, bg='seashell4',width=7, command=play).place(x=160, y=190)
Button(root, font='arial 13 bold', text='RESET', padx=5, bg='seashell4',width=7, command=Reset).place(x=70, y=310)
Button(root, font='arial 13 bold', text='EXIT', padx=5, bg='seashell4',width=7, command=Exit).place(x=240, y=310)

root.mainloop()