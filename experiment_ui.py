import tkinter as tk
import random

items = ["Yellow_tool","Multimeter","Tape","Ninja","Screwdriver","Hot_glue"]
item_counts = {item: {'detected': 0, 'not_detected': 0} for item in items}

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.run_counter = 0  # add a counter
        self.create_widgets()

    def create_widgets(self):
        self.item_button = tk.Button(self)
        self.item_button["text"] = "New run"
        self.item_button["command"] = self.new_item
        self.item_button.pack(side="top")

        self.item_name = tk.StringVar()
        self.item_label = tk.Label(self, textvariable=self.item_name)
        self.item_label.pack(side="top")

        # Display the counter
        self.run_counter_str = tk.StringVar()
        self.run_counter_label = tk.Label(self, textvariable=self.run_counter_str)
        self.run_counter_label.pack(side="top")

        self.detected_button = tk.Button(self)
        self.detected_button["text"] = "Detected"
        self.detected_button["command"] = self.detected
        self.detected_button.pack(side="top")

        self.not_detected_button = tk.Button(self)
        self.not_detected_button["text"] = "Not Detected"
        self.not_detected_button["command"] = self.not_detected
        self.not_detected_button.pack(side="top")

        self.quit = tk.Button(self, text="Finish run", fg="red",
                              command=self.finish_run)
        self.quit.pack(side="bottom")

    def new_item(self):
        self.current_item = random.choice(items)
        self.item_name.set(self.current_item)
        self.run_counter += 1  # increment the counter
        self.run_counter_str.set('Run count: ' + str(self.run_counter))  # update the counter display

    def detected(self):
        if self.current_item:
            item_counts[self.current_item]['detected'] += 1

    def not_detected(self):
        if self.current_item:
            item_counts[self.current_item]['not_detected'] += 1

    def finish_run(self):
        total_detections = 0
        total_non_detections = 0
        with open('run_counts_gru_extra_data.txt', 'w') as f:
            for item, counts in item_counts.items():
                detections = counts["detected"]
                non_detections = counts["not_detected"]
                total_runs = detections + non_detections

                total_detections += detections
                total_non_detections += non_detections

                if total_runs > 0:
                    accuracy = detections / total_runs
                else:
                    accuracy = 0
                f.write(f'{item}: Detected={detections}, Not Detected={non_detections}, Accuracy={accuracy * 100}%\n')

            if total_detections + total_non_detections > 0:
                total_accuracy = total_detections / (total_detections + total_non_detections)
            else:
                total_accuracy = 0
            f.write(
                f'Overall: Total Detected={total_detections}, Total Not Detected={total_non_detections}, Overall Accuracy={total_accuracy * 100}%\n')

        self.master.destroy()


root = tk.Tk()
app = Application(master=root)
app.mainloop()

