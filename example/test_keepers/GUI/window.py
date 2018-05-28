import tkinter as tk

root = tk.Tk()
frame = tk.Frame(root)
frame.pack()

title = tk.Label(frame, text = "HFO KEEPER PROGRAM", font=("Helvetica", 26),width = 80)
title.pack()

button = tk.Button(frame, text="High Level Agent")
button.pack()



root.mainloop()
