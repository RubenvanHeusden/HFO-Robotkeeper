from tkinter import *
from tkinter import filedialog
import sys
#from fixed_agent import *


return_values = {"experiment":"","f_name":"" , "learning_rate":0, 
                "num_trials":0, "pre_train":0, "update_freq":0, 
                "batch_size":0, "num_distances":0, "num_angles":0}

master = Tk()
master.title("Goalkeeper Menu")
variable = StringVar(master)
options = {"Transfer Learning": 0, "No Transfer Learning": 0}
variable.set(options.keys()[0]) # default value


exp_label = Label(master, text = "Type of experiment")
exp_label.pack()

w = OptionMenu(master, variable, *options.keys())
w.pack()



def submit_values():
    return_values["experiment"] = variable.get()
    return_values["learning_rate"] = float(learning_rate_txt.get())
    return_values["num_trials"] = int(trials_txt.get())
    return_values["pre_train"] = int(pre_train_txt.get())
    return_values["update_freq"] = int(update_freq_txt.get())
    return_values["batch_size"] = int(batch_size_txt.get())  
    
    return_values["num_angles"] = int(num_angles_txt.get())
    return_values["num_distances"] = int(num_distances_txt.get()) 
    master.quit()

type_var = StringVar(master)
type_options = ["Run", "Test", "Load"]
type_var.set(type_options[0])

def type_selection(var):
    if var == "Load":
        return_values["f_name"] = filedialog.askopenfilename(initialdir="/home/student/Desktop",title="Select training file")
    else:
        return_values["f_name"] = var

type_label = Label(master, text = "run/test/load a model")
type_label.pack()


w2 = OptionMenu(master, type_var, *type_options, command=type_selection)
w2.pack()


type_label = Label(master, text = "learning rate")
type_label.pack()

learning_rate_txt = Entry(master)
learning_rate_txt.pack()
learning_rate_txt.delete(0, END)
learning_rate_txt.insert(0, "0.0001")


trials_label = Label(master, text = "# of trials")
trials_label.pack()
trials_txt = Entry(master)
trials_txt.pack()
trials_txt.delete(0, END)
trials_txt.insert(0, "10000")

pretrain_label = Label(master, text = "# of pre-train steps")
pretrain_label.pack()
pre_train_txt = Entry(master)
pre_train_txt.pack()
pre_train_txt.delete(0, END)
pre_train_txt.insert(0, "4000")

update_label = Label(master, text = "train_frequency")
update_label.pack()
update_freq_txt = Entry(master)
update_freq_txt.pack()
update_freq_txt.delete(0, END)
update_freq_txt.insert(0, "200")

batch_label = Label(master, text = "# of batches to update with")
batch_label.pack()
batch_size_txt = Entry(master)
batch_size_txt.pack()
batch_size_txt.delete(0, END)
batch_size_txt.insert(0, "16")

angles_label = Label(master, text = "# of angle bins")
angles_label.pack()
num_angles_txt = Entry(master)
num_angles_txt.pack()
num_angles_txt.delete(0, END)
num_angles_txt.insert(0, "10")

distances_label = Label(master, text = "# of distance bins")
distances_label.pack()
num_distances_txt = Entry(master)
num_distances_txt.pack()
num_distances_txt.delete(0, END)
num_distances_txt.insert(0, "10")




button = Button(master, text="OK", command=submit_values)
button.pack()

master.mainloop()

def run_experiment(params):
    transfer = (return_values["experiment"][0] == "T")
    learning_rate = return_values["learning_rate"]   
    num_trials = return_values["num_trials"] 
    pre_train_steps = return_values["pre_train"]
    update_freq = return_values["update_freq"]
    batch_size = return_values["batch_size"]   
    num_distances = return_values["num_distances"] 
    num_angles = return_values["num_angles"] 
    
    if return_values["f_name"] == "Run":
        pass
    elif return_values["f_name"] == "Test":
        pass



# call main.py with the appropriate arguments to start the experiment

run_experiment(return_values)











