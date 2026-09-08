import threading
import tkinter as tk

from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

from modules.face import verify_faces


class FaceVerificationGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("DocScreen - Face Verification")
        self.root.geometry("950x650")
        self.root.resizable(False, False)

        self.document_path = None
        self.person_path = None
        self.threshold = tk.DoubleVar(value=0.40)
        self.status = tk.StringVar(value="Ready")

        self.build_ui()

    def build_ui(self):

        title = ttk.Label(self.root, text="FACE VERIFICATION", font=("Arial", 18, "bold"))
        title.pack(pady=(15, 5))

        subtitle = ttk.Label(self.root, text="Compare a document face with a reference person image")
        subtitle.pack(pady=(0, 15))

        images = ttk.Frame(self.root)
        images.pack(fill="x", padx=20)

        document_frame = ttk.LabelFrame(images, text="Document / Passport", padding=10)
        document_frame.pack(side="left", fill="both", expand=True, padx=10)

        person_frame = ttk.LabelFrame(images, text="Reference / Person", padding=10)
        person_frame.pack(side="right", fill="both", expand=True, padx=10)

        self.document_label = ttk.Label(document_frame, text="No image selected", anchor="center")
        self.document_label.pack(pady=10)

        self.person_label = ttk.Label(person_frame, text="No image selected", anchor="center")
        self.person_label.pack(pady=10)

        ttk.Button(document_frame, text="Select Document", command=self.select_document).pack(pady=5)
        ttk.Button(person_frame, text="Select Person", command=self.select_person).pack(pady=5)

        settings = ttk.LabelFrame(self.root, text="Verification Settings", padding=10)
        settings.pack(fill="x", padx=30, pady=15)

        ttk.Label(settings, text="Threshold:").pack(side="left", padx=(10, 5))

        ttk.Entry(settings, textvariable=self.threshold, width=8).pack(side="left")

        ttk.Label(settings, text="(default: 0.40)").pack(side="left", padx=5)

        self.verify_button = ttk.Button(
            settings,
            text="VERIFY FACES",
            command=self.start_verification
        )
        self.verify_button.pack(side="right", padx=10, ipadx=20, ipady=5)

        # ---------------------------------------------------------
        # Verification Result
        # ---------------------------------------------------------

        results = ttk.LabelFrame(self.root, text="Verification Result", padding=15)
        results.pack(fill="x", padx=30, pady=5)

        score_frame = ttk.Frame(results)
        score_frame.pack(side="left", expand=True, padx=80)

        ttk.Label(
            score_frame,
            text="Similarity Score",
            font=("Arial", 11)
        ).pack()

        self.similarity_label = ttk.Label(
            score_frame,
            text="--",
            font=("Arial", 22, "bold")
        )
        self.similarity_label.pack(pady=5)

        match_frame = ttk.Frame(results)
        match_frame.pack(side="right", expand=True, padx=80)

        ttk.Label(
            match_frame,
            text="Verification",
            font=("Arial", 11)
        ).pack()

        self.match_label = ttk.Label(
            match_frame,
            text="--",
            font=("Arial", 22, "bold")
        )
        self.match_label.pack(pady=5)

        ttk.Label(
            self.root,
            textvariable=self.status,
            relief="sunken",
            anchor="w"
        ).pack(fill="x", side="bottom")

    # -------------------------------------------------------------
    # Select Document
    # -------------------------------------------------------------

    def select_document(self):

        path = filedialog.askopenfilename(
            title="Select Document / Passport Image",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")
            ]
        )

        if not path:
            return

        self.document_path = path
        self.show_image(path, self.document_label)
        self.status.set("Document image selected")

    # -------------------------------------------------------------
    # Select Person
    # -------------------------------------------------------------

    def select_person(self):

        path = filedialog.askopenfilename(
            title="Select Reference / Person Image",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")
            ]
        )

        if not path:
            return

        self.person_path = path
        self.show_image(path, self.person_label)
        self.status.set("Reference image selected")

    # -------------------------------------------------------------
    # Display Image
    # -------------------------------------------------------------

    def show_image(self, path, label):

        image = Image.open(path)
        image.thumbnail((350, 300))

        photo = ImageTk.PhotoImage(image)

        label.configure(image=photo, text="")
        label.image = photo

    # -------------------------------------------------------------
    # Start Verification
    # -------------------------------------------------------------

    def start_verification(self):

        if not self.document_path:
            messagebox.showerror(
                "Missing Image",
                "Please select the document/passport image."
            )
            return

        if not self.person_path:
            messagebox.showerror(
                "Missing Image",
                "Please select the reference/person image."
            )
            return

        try:
            threshold = float(self.threshold.get())
        except ValueError:
            messagebox.showerror(
                "Invalid Threshold",
                "Threshold must be a number."
            )
            return

        self.verify_button.config(state="disabled")

        self.similarity_label.config(text="...")
        self.match_label.config(text="PROCESSING...")

        self.status.set("Verifying faces...")

        threading.Thread(
            target=self.verify,
            args=(threshold,),
            daemon=True
        ).start()

    # -------------------------------------------------------------
    # Face Verification
    # -------------------------------------------------------------

    def verify(self, threshold):

        try:
            result = verify_faces(
                self.document_path,
                self.person_path,
                threshold
            )

            similarity = result["similarity"]
            match = result["match"]

            self.root.after(
                0,
                self.show_result,
                similarity,
                match
            )

        except Exception as e:

            self.root.after(
                0,
                self.verification_error,
                str(e)
            )

    # -------------------------------------------------------------
    # Display Result
    # -------------------------------------------------------------

    def show_result(self, similarity, match):

        self.similarity_label.config(
            text=f"{similarity:.4f}"
        )

        self.match_label.config(
            text="MATCH" if match else "NO MATCH"
        )

        self.status.set("Verification complete")

        self.verify_button.config(state="normal")

    # -------------------------------------------------------------
    # Error
    # -------------------------------------------------------------

    def verification_error(self, error):

        self.similarity_label.config(text="--")
        self.match_label.config(text="ERROR")

        self.status.set("Verification failed")

        self.verify_button.config(state="normal")

        messagebox.showerror(
            "Verification Error",
            error
        )


if __name__ == "__main__":

    root = tk.Tk()
    FaceVerificationGUI(root)
    root.mainloop()