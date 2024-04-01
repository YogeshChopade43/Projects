import tkinter as tk
import csv

class Bed:
    def __init__(self, canvas, x, y, width, height, bed_number):
        self.canvas = canvas
        self.rect = canvas.create_rectangle(x, y, x + width, y + height, fill="green", outline="black")
        self.label = canvas.create_text((x + x + width) / 2, (y + y + height) / 2, text=f"Bed {bed_number}",
                                        font=("Helvetica", 10, "bold"))
        self.occupied = False
        self.bed_number = bed_number
        self.patient_name = None

    def occupy(self, patient_name):
        if not self.occupied:
            self.canvas.itemconfig(self.rect, fill="red")
            self.occupied = True
            self.patient_name = patient_name
            self.canvas.itemconfig(self.label, text=f"Bed {self.bed_number}\n{self.patient_name}")

    def free(self):
        if self.occupied:
            self.canvas.itemconfig(self.rect, fill="green")
            self.occupied = False
            self.patient_name = None
            self.canvas.itemconfig(self.label, text=f"Bed {self.bed_number}")

    def highlight(self):
        self.canvas.itemconfig(self.rect, outline="blue", width=2)

    def unhighlight(self):
        self.canvas.itemconfig(self.rect, outline="", width=1)

class HospitalManagementSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Hospital Bed Management System")

        self.canvas = tk.Canvas(root, width=650, height=500, bg="white")
        self.canvas.pack()

        self.create_beds()
        self.selected_bed = None

        # Create a frame for buttons and input fields
        input_frame = tk.Frame(root)
        input_frame.pack()

        # Create a label for the patient name input
        self.patient_name_label = tk.Label(input_frame, text="Name of the Patient:",font=("Helvetica", 12),width=20, height=5)
        self.patient_name_label.pack(side="left")

        # Create an entry widget for patient name input
        self.patient_name_entry = tk.Entry(input_frame)
        self.patient_name_entry.pack(side="left")

        # Create a "Save" button with increased size
        self.save_button = tk.Button(input_frame, text="Save", command=self.save_to_csv, height=1, width=8)
        self.save_button.pack(side="left")

        # Create "Allot Bed" and "Empty Bed" buttons with increased size
        self.allot_button = tk.Button(root, text="Allot Bed", command=self.allot_bed, height=2, width=10)
        self.allot_button.pack()

        self.empty_button = tk.Button(root, text="Empty Bed", command=self.empty_bed, height=2, width=10)
        self.empty_button.pack()

        # Create a new window for real-time bed information
        self.info_window = tk.Toplevel(root)
        self.info_window.title("Bed Information")
        self.create_bed_info_canvas()  # Create the canvas and scrollbar

        # Initialize the bed information labels
        self.info_labels = []
        self.update_bed_info()

    def create_beds(self):
        self.beds = []
        for i in range(5):
            for j in range(4):
                x = 50 + j * 150
                y = 50 + i * 100
                bed_number = i * 4 + j + 1
                bed = Bed(self.canvas, x, y, 100, 50, bed_number)
                self.beds.append(bed)

    def allot_bed(self):
        if self.selected_bed and not self.selected_bed.occupied:
            patient_name = self.patient_name_entry.get()
            if patient_name:
                self.selected_bed.occupy(patient_name)
                self.update_logs()
                self.update_bed_info()

    def empty_bed(self):
        if self.selected_bed and self.selected_bed.occupied:
            self.selected_bed.free()
            self.update_logs()
            self.update_bed_info()

    def bed_clicked(self, event):
        x, y = event.x, event.y
        for bed in self.beds:
            coords = self.canvas.coords(bed.rect)
            if (coords[0] <= x <= coords[2]) and (coords[1] <= y <= coords[3]):
                if self.selected_bed:
                    self.selected_bed.unhighlight()
                self.selected_bed = bed
                self.selected_bed.highlight()
                self.patient_name_entry.delete(0, tk.END)

    def save_to_csv(self):
        file_name = "bed_occupancy.csv"
        with open(file_name, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Bed Number", "Status", "Patient Name"])
            for bed in self.beds:
                status = "Occupied" if bed.occupied else "Empty"
                writer.writerow([bed.bed_number, status, bed.patient_name if bed.patient_name else "N/A"])

    def update_logs(self):
        # Update logs here
        pass

    def update_bed_info(self):
        # Clear the previous bed information labels
        for label_id in self.info_labels:
            self.info_canvas.delete(label_id)

        # Display updated bed information
        for bed in self.beds:
            x = 20
            y = (bed.bed_number - 1) * 30
            status = "Occupied" if bed.occupied else "Empty"
            patient_name = bed.patient_name if bed.occupied else "N/A"
            text_color = "red" if bed.occupied else "green"
            info_label = self.info_canvas.create_text(x, y,
                                                      text=f"Bed {bed.bed_number}: {status}\nPatient: {patient_name}",
                                                      anchor="w", fill=text_color)
            self.info_labels.append(info_label)

    def create_bed_info_canvas(self):
        # Create a new frame for bed information
        info_frame = tk.Frame(self.info_window)
        info_frame.pack(fill=tk.BOTH, expand=True)

        # Create a canvas for displaying bed information
        self.info_canvas = tk.Canvas(info_frame, width=700, bg="white")
        self.info_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create a vertical scrollbar
        scrollbar = tk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.info_canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Configure the canvas to use the scrollbar
        self.info_canvas.config(yscrollcommand=scrollbar.set)

        # Bind the canvas to configure the scroll region
        self.info_canvas.bind("<Configure>", self.configure_scroll_region)

    def configure_scroll_region(self, event):
        self.info_canvas.config(scrollregion=self.info_canvas.bbox("all"))

if __name__ == "__main__":
    root = tk.Tk()
    app = HospitalManagementSystem(root)
    app.canvas.bind("<Button-1>", app.bed_clicked)
    root.title("Hospital Bed Management System")
    root.geometry("1000x600")
    root.configure(bg="white")

    app.allot_button.configure(text="Allot Bed", font=("Helvetica", 12, "bold"), bg="blue", fg="white")
    app.empty_button.configure(text="Empty Bed", font=("Helvetica", 12, "bold"), bg="blue", fg="white")

    root.mainloop()
