from tkinter import *
import sys
import os
import yaml

class Application(Frame):
    def RunREMM(self):
        print("I am Running! /n  -----------------")
        print("Parcel Year is ", self.year.get())
        print("Random Seed is ", self.seed.get())
        fname = "configs/settings.yaml"
        stream = open(fname, 'r')
        data = yaml.load(stream, Loader=yaml.FullLoader)
        data['build_networks']['max_distance'] = self.seed.get()
        with open(fname, 'w') as yaml_file:
            yaml_file.write(yaml.dump(data, default_flow_style=False))

        os.system('python RunREMM.py')

    def createWidgets(self):
        self.welcom=Label(self, text="Welcome To REMM GUI!")
        self.welcom.grid(row=0, column=1)

        self.year0=Label(self, text="Output Parcel Year: ")
        self.year0.grid(row=1, column=1)
        self.year=Entry(self)
        self.year.grid(row=1, column=2)
        self.seed0=Label(self, text="Random Seed: ")
        self.seed0.grid(row=2, column=1)
        self.seed=Entry(self)
        self.seed.grid(row=2, column=2)
        self.tdm0=Label(self, text="TDM Folder: ")
        self.tdm0.grid(row=3, column=1)
        self.tdm=Entry(self)
        self.tdm.grid(row=3, column=2)
        self.input0=Label(self, text="Input Folder: ")
        self.input0.grid(row=4, column=1)
        self.input=Entry(self)
        self.input.grid(row=4, column=2)
        self.output0=Label(self, text="Output Folder: ")
        self.output0.grid(row=5, column=1)
        self.output=Entry(self)
        self.output.grid(row=5, column=2)
        self.gpi0=Label(self, text="Output GPI Folder: ")
        self.gpi0.grid(row=6, column=1)
        self.gpi=Entry(self)
        self.gpi.grid(row=6, column=2)
        self.qaqc0=Label(self, text="Output QAQC Folder: ")
        self.qaqc0.grid(row=7, column=1)
        self.qaqc=Entry(self)
        self.qaqc.grid(row=7, column=2)

        self.yearx=Label(self, text="")
        self.yearx.grid(row=8, column=1)

        self.run = Button(self)
        self.run["text"] = "Click Here to Runn REMM!",
        self.run["command"] = self.RunREMM
        self.run.grid(row=9, column=1)

        self.runqaqc = Button(self)
        self.runqaqc["text"] = "Click Here to QAQC!",
        self.runqaqc["command"] = self.RunREMM
        self.runqaqc.grid(row=10, column=1)

        self.QUIT = Button(self)
        self.QUIT["text"] = "QUIT"
        self.QUIT["fg"]   = "red"
        self.QUIT["command"] =  self.quit
        self.QUIT.grid(row=11, column=1)
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()
        self.createWidgets()

root = Tk()
app = Application(master=root)
app.mainloop()
root.destroy()
