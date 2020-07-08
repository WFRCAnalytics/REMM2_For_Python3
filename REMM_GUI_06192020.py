from tkinter import *
import tkinter.filedialog as filedialog
import sys
import os
import yaml

 

class Application(Frame):
    def input_remm(self):
        input_path = filedialog.askdirectory(parent=root,title="Select the folder of REMM Input")
        self.inputf.delete(1, END)  # Remove current text in entry
        self.inputf.insert(0, input_path)  # Insert the 'path'
        print("The Input folder is",input_path)

    def output_remm(self):
        output_path = filedialog.askdirectory(parent=root,title="Select the folder of REMM Output")
        self.outputf.delete(1, END)  # Remove current text in entry
        self.outputf.insert(0, output_path)  # Insert the 'path'
        print("The Output folder is",output_path)

    def tdmf_remm(self):
        tdmf_path = filedialog.askdirectory(parent=root,title="Select the folder of TDM for REMM")
        self.tdmf.delete(1, END)  # Remove current text in entry
        self.tdmf.insert(0, tdmf_path)  # Insert the 'path'
        print("The TDM for REMM folder is",tdmf_path)

    def qaqcf_remm(self):
        qaqcf_path = filedialog.askdirectory(parent=root,title="Select the folder of QAQC for REMM")
        self.qaqcf.delete(1, END)  # Remove current text in entry
        self.qaqcf.insert(0, qaqcf_path)  # Insert the 'path'
        print("The QAQC for REMM folder is",qaqcf_path)

    def gpif_remm(self):
        gpif_path = filedialog.askdirectory(parent=root,title="Select the folder of GPI for REMM")
        self.gpif.delete(1, END)  # Remove current text in entry
        self.gpif.insert(0, gpif_path)  # Insert the 'path'
        print("The GPI for REMM folder is",gpif_path)

    def RunREMM(self):
        print("I am Running! /n  -----------------")

        fname = "configs/settings_original.yaml"
        stream = open(fname, 'r')
        data = yaml.load(stream, Loader=yaml.FullLoader)

        data['0scenario_name'] = self.scenarion.get()

        if len(self.horizony.get())>0 and int(self.horizony.get()) > 2015:
              data['1horizon_year'] = self.horizony.get()
        else:
              data['1horizon_year'] = 2061

        data['4tdm_folder'] = self.tdmf.get()  #for quick check redundent variable
        if len(self.tdmf.get())> 0:
              data['tdm']['main_dir']  = self.tdmf.get()+'/REMM/REMM_'
              data['tdm']['input_dir'] = self.tdmf.get()+'/1_Inputs/2_SEData/REMM'
              data['tdm']['bat_dir']   = self.tdmf.get()+'/REMM'
        else:
              data['tdm']['main_dir']  = 'E:/_TDMv8.3_REMM/REMM/REMM_'
              data['tdm']['input_dir'] = 'E:/_TDMv8.3_REMM/1_Inputs/2_SEData/REMM'
              data['tdm']['bat_dir']   = 'E:/_TDMv8.3_REMM/REMM'

        if len(self.seed.get())==0:
            data['7Randomseed']=1
        else:
            is_int = True
            try:
                  # convert to integer
                  int(str)
            except ValueError:
                  is_int = False
            # print result
            if is_int:
                  data['7Randomseed']=int(self.seed.get())
            else:
                  print('Random Seed should be either empty or an integer')

        data['2Remm_input_folder'] = self.inputf.get()
        data['3Remm_output_folder'] = self.outputf.get()
        data['5QAQCoutput_folder'] = self.qaqcf.get()
        data['6GPI_output_folder'] = self.gpif.get()

        data['8RunGPIModel'] = self.gpiy.get()
        if self.tdmy11.get()==1:
              data['tdm']['run_years'].append(2015)
        if self.tdmy21.get()==1:
              data['tdm']['run_years'].append(2022)
        if self.tdmy31.get()==1:
              data['tdm']['run_years'].append(2033)
        if self.tdmy41.get()==1:
              data['tdm']['run_years'].append(2045)
        if self.tdmy51.get()==1:
              data['tdm']['run_years'].append(2055)

        fnameo = "configs/settings.yaml"
        with open(fnameo, 'w') as yaml_file:
            yaml_file.write(yaml.dump(data, default_flow_style=False))

        os.system('python RunREMM.py')

    def createWidgets(self):
        self.welcom=Label(self, text="Welcome To ")
        self.welcom.grid(row=0, column=1)
        self.welcom=Label(self, text="REMM GUI!")
        self.welcom.grid(row=0, column=2)

        self.scenarion0=Label(self, text="Scenario Name: ")
        self.scenarion0.grid(row=1, column=1)
        self.scenarion=Entry(self)
        self.scenarion.grid(row=1, column=2)
        self.horizony0=Label(self, text="Horizon Year: ")
        self.horizony0.grid(row=2, column=1)
        self.horizony=Entry(self)
        self.horizony.grid(row=2, column=2)

        self.inputf0=Label(self, text="Input Folder: ")
        self.inputf0.grid(row=3, column=1)
        self.inputf=Entry(self)
        self.inputf.grid(row=3, column=2)
        self.inputfb=Button(self)
        self.inputfb["text"] = "Browse",
        self.inputfb["command"] = self.input_remm
        self.inputfb.grid(row=3, column=3)

        self.outputf0=Label(self, text="Output Folder: ")
        self.outputf0.grid(row=4, column=1)
        self.outputf=Entry(self)
        self.outputf.grid(row=4, column=2)
        self.outputfb=Button(self)
        self.outputfb["text"] = "Browse",
        self.outputfb["command"] = self.output_remm
        self.outputfb.grid(row=4, column=3)

        self.tdmf0=Label(self, text="TDM Folder: ")
        self.tdmf0.grid(row=5, column=1)
        self.tdmf=Entry(self)
        self.tdmf.grid(row=5, column=2)
        self.tdmfb=Button(self)
        self.tdmfb["text"] = "Browse",
        self.tdmfb["command"] = self.tdmf_remm
        self.tdmfb.grid(row=5, column=3)

        self.qaqcf0=Label(self, text="Output QAQC Folder: ")
        self.qaqcf0.grid(row=6, column=1)
        self.qaqcf=Entry(self)
        self.qaqcf.grid(row=6, column=2)
        self.qaqcfb=Button(self)
        self.qaqcfb["text"] = "Browse",
        self.qaqcfb["command"] = self.qaqcf_remm
        self.qaqcfb.grid(row=6, column=3)

        self.gpif0=Label(self, text="Output GPI Folder: ")
        self.gpif0.grid(row=7, column=1)
        self.gpif=Entry(self)
        self.gpif.grid(row=7, column=2)
        self.gpifb=Button(self)
        self.gpifb["text"] = "Browse",
        self.gpifb["command"] = self.gpif_remm
        self.gpifb.grid(row=7, column=3)

        self.seed0=Label(self, text="Random Seed: ")
        self.seed0.grid(row=8, column=1)
        self.seed=Entry(self)
        self.seed.grid(row=8, column=2)
        self.gpi0=Label(self, text="Run Regionwide GPI?: ")
        self.gpi0.grid(row=9, column=1)
        self.gpiy=IntVar()
        self.gpiyc=Checkbutton(self, text="Yes",variable=self.gpiy)
        self.gpiyc.grid(row=9, column=2, sticky=W)

        self.runtdm0=Label(self, text="Run TDM in Following Years? ")
        self.runtdm0.grid(row=10, column=1)
        self.tdmy10=Label(self, text="2015: ")
        self.tdmy10.grid(row=11, column=1)
        self.tdmy11 = IntVar()
        self.tdmy11c=Checkbutton(self, text="Yes",variable=self.tdmy11)
        self.tdmy11c.grid(row=11, column=2, sticky=W)
        self.tdmy20=Label(self, text="2022: ")
        self.tdmy20.grid(row=12, column=1)
        self.tdmy21 = IntVar()
        self.tdmy21c=Checkbutton(self, text="Yes",variable=self.tdmy21)
        self.tdmy21c.grid(row=12, column=2, sticky=W)
        self.tdmy30=Label(self, text="2027: ")
        self.tdmy30.grid(row=13, column=1)
        self.tdmy31 = IntVar()
        self.tdmy31c=Checkbutton(self, text="Yes",variable=self.tdmy31)
        self.tdmy31c.grid(row=13, column=2, sticky=W)
        self.tdmy40=Label(self, text="2035: ")
        self.tdmy40.grid(row=14, column=1)
        self.tdmy41 = IntVar()
        self.tdmy41c=Checkbutton(self, text="Yes",variable=self.tdmy41)
        self.tdmy41c.grid(row=14, column=2, sticky=W)
        self.tdmy50=Label(self, text="2045: ")
        self.tdmy50.grid(row=15, column=1)
        self.tdmy51 = IntVar()
        self.tdmy51c=Checkbutton(self, text="Yes",variable=self.tdmy51)
        self.tdmy51c.grid(row=15, column=2, sticky=W)

        self.yearx=Label(self, text="")
        self.yearx.grid(row=16, column=1)

        self.run = Button(self)
        self.run["text"] = "Click Here to Runn REMM!",
        self.run["command"] = self.RunREMM
        self.run.grid(row=17, column=1)

        self.runqaqc = Button(self)
        self.runqaqc["text"] = "Click Here to QAQC!",
        self.runqaqc["command"] = self.RunREMM
        self.runqaqc.grid(row=18, column=1)

        self.QUIT = Button(self)
        self.QUIT["text"] = "QUIT"
        self.QUIT["fg"]   = "red"
        self.QUIT["command"] =  self.quit
        self.QUIT.grid(row=19, column=1)
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()
        self.createWidgets()

root = Tk()
app = Application(master=root)
app.mainloop()
root.destroy()
